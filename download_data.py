"""
Download the UCI Household Power Consumption dataset.

This script downloads the dataset directly from the UCI Machine Learning Repository
and saves it to data/raw/ in the correct format for this project.

Run this once before running any notebooks:
    python download_data.py
"""

import os
import urllib.request
import zipfile

RAW = os.path.join("data", "raw")
os.makedirs(RAW, exist_ok=True)

URL = "https://archive.ics.uci.edu/static/public/235/individual+household+electric+power+consumption.zip"
ZIP_PATH = os.path.join(RAW, "uci_download.zip")
TXT_PATH = os.path.join(RAW, "household_power_consumption.txt")
CSV_PATH = os.path.join(RAW, "household_power_consumption.csv")

if os.path.exists(CSV_PATH) or os.path.exists(TXT_PATH):
    print("UCI dataset already present. Nothing to download.")
else:
    print("Downloading UCI Household Power Consumption dataset (~127 MB)...")
    print("This may take several minutes depending on your internet speed.")

    def show_progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(100, downloaded * 100 // total_size)
            bar = "#" * (pct // 5) + "-" * (20 - pct // 5)
            print(f"\r  [{bar}] {pct}%  ({downloaded // 1_000_000} MB / {total_size // 1_000_000} MB)", end="", flush=True)

    urllib.request.urlretrieve(URL, ZIP_PATH, reporthook=show_progress)
    print("\nDownload complete. Extracting...")

    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        z.extractall(RAW)

    os.remove(ZIP_PATH)

    if os.path.exists(TXT_PATH) and not os.path.exists(CSV_PATH):
        import shutil
        shutil.copy(TXT_PATH, CSV_PATH)
        print(f"Saved as: {CSV_PATH}")

    print("UCI dataset ready.")

print("\nAll data files present. You can now run the notebooks in order:")
print("  02_preprocessing.ipynb")
print("  03_eda.ipynb")
print("  04_models.ipynb")
print("  05_evaluation.ipynb")
