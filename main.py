import argparse
import base64
import json
import mimetypes
import os
import re
import sys
from pathlib import Path
from src.privacy_masking import mask_sensitive_regions

from dotenv import load_dotenv
from jsonschema import validate
from openai import OpenAI

load_dotenv()
SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif"}

def image_to_data_url(path: Path) -> str:
    mime, _ = mimetypes.guess_type(path.name)
    if mime not in {"image/png", "image/jpeg", "image/webp", "image/gif"}:
        raise ValueError(f"Unsupported image type: {path}")
    encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{encoded}"

def parse_json(text: str) -> dict:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    return json.loads(text)

def make_client():
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "").strip().rstrip("/")
    key = os.getenv("AZURE_OPENAI_API_KEY", "").strip()
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "").strip()
    missing = [n for n, v in [
        ("AZURE_OPENAI_ENDPOINT", endpoint),
        ("AZURE_OPENAI_API_KEY", key),
        ("AZURE_OPENAI_DEPLOYMENT", deployment)
    ] if not v]
    if missing:
        raise RuntimeError("Fill these values in .env: " + ", ".join(missing))
    base_url = endpoint if endpoint.endswith("/openai/v1") else endpoint + "/openai/v1/"
    return OpenAI(api_key=key, base_url=base_url), deployment

def analyze(client, deployment, image_path, prompt, schema, max_tokens):
    image_url = image_to_data_url(image_path)
    response = client.chat.completions.create(
        model=deployment,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": [
                {"type": "text", "text": (
                    f"Analyze {image_path.name}. Return only the required JSON. "
                    "Use the original image dimensions for bounding boxes."
                )},
                {"type": "image_url", "image_url": {"url": image_url, "detail": "high"}}
            ]}
        ],
        max_completion_tokens=max_tokens
    )
    content = response.choices[0].message.content
    if not content:
        raise RuntimeError("The model returned an empty response.")
    result = parse_json(content)
    validate(instance=result, schema=schema)
    return result

def main():
    parser = argparse.ArgumentParser(description="Vision-only SAP screenshot extraction")
    parser.add_argument("--input", required=True, help="One image or a folder of images")
    parser.add_argument("--output", default="outputs", help="Output folder for JSON files")
    parser.add_argument("--max-tokens", type=int, default=12000)
    args = parser.parse_args()

    here = Path(__file__).resolve().parent
    config_dir = here / "configuration_files"
    prompt = (config_dir / "system_prompt.txt").read_text(encoding="utf-8")
    schema = json.loads((config_dir / "schema.json").read_text(encoding="utf-8"))
    client, deployment = make_client()

    source = Path(args.input).expanduser()
    if source.is_file():
        images = [source] if source.suffix.lower() in SUPPORTED_EXTENSIONS else []
    elif source.is_dir():
        images = sorted(p for p in source.rglob("*")
                        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS)
    else:
        raise FileNotFoundError(source)

    if not images:
        raise RuntimeError("No supported images found. Use PNG, JPG, JPEG, WEBP, or GIF.")

    output_dir = Path(args.output).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    failures = []

    for i, image in enumerate(images, 1):
        print(f"[{i}/{len(images)}] {image.name}")
        try:
            masked_path = output_dir / "masked_images" / f"{image.stem}_masked.png"
            masked_path.parent.mkdir(parents=True, exist_ok=True)

            mask_sensitive_regions(
                image_path=image,
                output_path=masked_path,
            )

            if not masked_path.is_file() or masked_path.stat().st_size == 0:
                raise RuntimeError("Masking failed. Image will not be sent to Azure.")
            
            result = analyze(client, deployment, image, prompt, schema, args.max_tokens)
            out = output_dir / (re.sub(r"[^A-Za-z0-9._-]+", "_", image.stem) + ".json")
            if out.exists():
                out = output_dir / (re.sub(r"[^A-Za-z0-9._-]+", "_", image.parent.name)
                                    + "__" + re.sub(r"[^A-Za-z0-9._-]+", "_", image.stem) + ".json")
            out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"    Saved {out}")
        except Exception as exc:
            failures.append((str(image), str(exc)))
            print(f"    FAILED: {exc}", file=sys.stderr)

    print(f"\nDone: {len(images)-len(failures)}/{len(images)} succeeded.")
    if failures:
        for name, error in failures:
            print(f"- {name}: {error}", file=sys.stderr)
        sys.exit(2)

if __name__ == "__main__":
    main()
