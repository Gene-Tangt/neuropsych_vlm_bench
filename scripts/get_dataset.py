"""Download the test images from Google Drive into datasets/."""

import shutil
import zipfile

import gdown

URL = "https://drive.google.com/drive/folders/1qcAQBB9C1vf3PdaSPer4kNVBOrC7ORf4?usp=sharing"

gdown.download_folder(URL, use_cookies=False, quiet=False)
with zipfile.ZipFile("neuropsych_vlm_bench_datasets/datasets.zip") as archive:
    archive.extractall(".")
shutil.rmtree("neuropsych_vlm_bench_datasets")
print("Dataset saved to datasets/")
