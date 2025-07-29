import gdown
import zipfile
import os
import shutil

def get_dataset():
    """
    Downloads the dataset from Google Drive using file ID, extracts it to 'datasets' folder,
    and removes the zip file.
    """
    url = "https://drive.google.com/drive/folders/1qcAQBB9C1vf3PdaSPer4kNVBOrC7ORf4?usp=sharing"
    print("Downloading dataset from Google Drive...")
    gdown.download_folder(url, use_cookies=False, quiet=False)

    print("Extracting dataset...")
    with zipfile.ZipFile("neuropsych_vlm_bench_datasets/datasets.zip", "r") as zip_ref:
        zip_ref.extractall(".")

    print("Removing zip file...")
    shutil.rmtree("neuropsych_vlm_bench_datasets")

    print("Dataset downloaded successfully. Dataset is located in 'datasets' folder.")


if __name__ == "__main__":
    get_dataset()
