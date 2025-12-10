import os
import zipfile

zip_path = "data/HRSID_raw/HRSID_jpg.zip"
dest_dir = "data/HRSID_raw"

if os.path.exists(zip_path):
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(dest_dir)
    print("Unzipped successfully.")
else:
    print("ZIP file not found.")
