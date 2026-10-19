
import subprocess
import sys
from pathlib import Path

# Folder containing your SAP screenshots
INPUT_DIR = Path(
    r"C:\Sumedh\coding\sap_visual_assistant\data\raw_data\SAP_Screenshot_Samples\AI_AMS_SAP_Screenshot_Samples"
)

# Folder where JSON outputs will be saved
OUTPUT_DIR = Path(r"C:\Sumedh\coding\sap_visual_assistant\outputs")

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Run the existing main.py on the entire folder
    command = [
        sys.executable,
        "main.py",
        "--input", str(INPUT_DIR),
        "--output", str(OUTPUT_DIR)
    ]

    subprocess.run(command, check=True)

if __name__ == "__main__":
    main()
