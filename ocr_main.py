import argparse, base64, json, mimetypes, os, re, sys
from pathlib import Path
from dotenv import load_dotenv
from jsonschema import validate
from openai import OpenAI
from paddleocr import PaddleOCR
from src.privacy_masking import mask_sensitive_regions

load_dotenv()
EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

def image_data_url(path):
    mime = mimetypes.guess_type(path.name)[0]
    if mime not in {"image/png", "image/jpeg", "image/webp", "image/bmp"}:
        raise ValueError(f"Unsupported image type: {path}")
    return "data:" + mime + ";base64," + base64.b64encode(path.read_bytes()).decode("utf-8")

def parse_json(text):
    text = re.sub(r"^```(?:json)?\\s*", "", text.strip(), flags=re.I)
    text = re.sub(r"\\s*```$", "", text)
    return json.loads(text)

def azure_client():
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "").strip().rstrip("/")
    key = os.getenv("AZURE_OPENAI_API_KEY", "").strip()
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "").strip()
    missing = [k for k,v in [("AZURE_OPENAI_ENDPOINT",endpoint),("AZURE_OPENAI_API_KEY",key),("AZURE_OPENAI_DEPLOYMENT",deployment)] if not v]
    if missing:
        raise RuntimeError("Fill these values in .env: " + ", ".join(missing))
    base_url = endpoint if endpoint.endswith("/openai/v1") else endpoint + "/openai/v1/"
    return OpenAI(api_key=key, base_url=base_url), deployment

def as_list(v):
    return v.tolist() if hasattr(v, "tolist") else (list(v) if v is not None else [])

def ocr_image(engine, path):
    results = engine.predict(str(path))
    found = []
    for result in results or []:
        payload = getattr(result, "json", None)
        if callable(payload): payload = payload()
        if not isinstance(payload, dict): payload = result if isinstance(result, dict) else {}
        if isinstance(payload.get("res"), dict): payload = payload["res"]
        texts = as_list(payload.get("rec_texts", []))
        scores = as_list(payload.get("rec_scores", []))
        boxes = as_list(payload.get("rec_boxes", []))
        for i, txt in enumerate(texts):
            if txt is None or not str(txt).strip(): continue
            box = as_list(boxes[i]) if i < len(boxes) else []
            box = [int(round(float(x))) for x in box[:4]] if len(box) >= 4 else None
            score = float(scores[i]) if i < len(scores) and scores[i] is not None else None
            found.append({"text":str(txt), "bounding_box":box, "confidence":round(score,4) if score is not None else None})
    return found

def interpret(client, deployment, image, detections, prompt, schema, max_tokens):
    ocr_text = json.dumps(
        {
            "image_file": image.name,
            "coordinates": "pixel [x1,y1,x2,y2], origin top-left",
            "detections": detections
        },
        ensure_ascii=False
    )

    response = client.chat.completions.create(
        model=deployment,
        messages=[
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Interpret the screenshot AND these OCR detections. Return only the required JSON.\nOCR DETECTIONS:\n" + ocr_text
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_data_url(image),
                            "detail": "high"
                        }
                    }
                ]
            }
        ],
        max_completion_tokens=max_tokens
    )

    # DEBUG: Inspect the response returned by Azure
    print("\n--- AZURE RESPONSE DEBUG ---", flush=True)
    print("Response ID:", response.id, flush=True)
    print("Model:", response.model, flush=True)
    print("Number of choices:", len(response.choices), flush=True)
    print(
        "Usage:",
        response.usage.model_dump() if response.usage else None,
        flush=True
    )

    for i, choice in enumerate(response.choices):
        print("Finish reason:", choice.finish_reason, flush=True)
        print("Content:", repr(choice.message.content), flush=True)
        print("Refusal:", repr(choice.message.refusal), flush=True)
        print("Full message:", choice.message.model_dump(), flush=True)

    print("--- END AZURE RESPONSE DEBUG ---\n", flush=True)

    if not response.choices:
        raise RuntimeError("Azure returned no choices")

    content = response.choices[0].message.content

    if not content:
        raise RuntimeError("Azure model returned an empty response")

    data = parse_json(content)
    validate(instance=data, schema=schema)
    return data


def safe(s): return re.sub(r"[^A-Za-z0-9._-]+","_",s).strip("._") or "image"

def main():
    ap = argparse.ArgumentParser(description="Local PaddleOCR + Azure Vision LLM SAP screenshot extraction")
    ap.add_argument("--input",required=True,help="One image or a folder")
    ap.add_argument("--output",default="outputs/ocr_vision")
    ap.add_argument("--max-tokens",type=int,default=12000)
    args = ap.parse_args()
    here = Path(__file__).resolve().parent
    config_dir = here / "configuration_files"
    prompt = (config_dir/"system_prompt_ocr.txt").read_text(encoding="utf-8")
    schema = json.loads((config_dir/"schema_ocr.json").read_text(encoding="utf-8"))
    client, deployment = azure_client()
    source = Path(args.input).expanduser()
    if source.is_file(): images = [source] if source.suffix.lower() in EXTS else []
    elif source.is_dir(): images = sorted(p for p in source.rglob("*") if p.is_file() and p.suffix.lower() in EXTS)
    else: raise FileNotFoundError(source)
    if not images: raise RuntimeError(f"No supported images found: {source}")
    out = Path(args.output).expanduser()
    raw_dir, json_dir = out/"ocr_raw", out/"json"
    raw_dir.mkdir(parents=True,exist_ok=True); json_dir.mkdir(parents=True,exist_ok=True)
    print("Loading PaddleOCR; first run may download model weights...")
    engine = PaddleOCR(
        lang="en",
        enable_mkldnn=False
    )
    failures = []
    for n,img in enumerate(images,1):
        print(f"[{n}/{len(images)}] OCR: {img.name}")
        try:
            # Create a separate masked copy
            masked_path = out / "masked_images" / f"{img.stem}_masked.png"
            masked_path.parent.mkdir(parents=True, exist_ok=True)

            mask_sensitive_regions(
                image_path=img,
                output_path=masked_path,
                ocr_engine=engine,
            )

            if not masked_path.is_file() or masked_path.stat().st_size == 0:
                raise RuntimeError(
                    f"Masking failed for {img.name}. "
                    "The original image will not be processed."
            )
            detections = ocr_image(engine,masked_path)
            raw_path = raw_dir/(safe(img.stem)+"_ocr.json")
            raw_path.write_text(json.dumps({"image":img.name,"detections":detections},ensure_ascii=False,indent=2),encoding="utf-8")
            print(f"    {len(detections)} OCR detections. Sending image + OCR to Azure...")
            result = interpret(client, deployment, masked_path, detections,prompt, schema, args.max_tokens)
            out_path = json_dir/(safe(img.stem)+".json")
            if out_path.exists(): out_path = json_dir/(safe(img.parent.name)+"__"+safe(img.stem)+".json")
            out_path.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
            print(f"    Saved: {out_path}")
        except Exception as e:
            failures.append((str(img),str(e))); print(f"    FAILED: {e}",file=sys.stderr)
    print(f"\\nFinished: {len(images)-len(failures)}/{len(images)} succeeded.")
    print(f"Final JSON: {json_dir}\\nRaw OCR: {raw_dir}")
    if failures:
        for name,err in failures: print(f"- {name}: {err}",file=sys.stderr)
        sys.exit(2)

if __name__ == "__main__": main()
