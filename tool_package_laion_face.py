import zipfile
from glob import glob

zf = zipfile.ZipFile("./dataset_laion_face_no_target_v2.zip", 'w')

# Add all the normal files:
for file in [
    "gradio_face2image.py",
    "laion_face_common.py",
    "laion_face_dataset.py",
    "README_laion_face.md",
    "tool_generate_face_poses.py",
    "tool_download_face_targets.py",
    "train_laion_face.py",
    "train_laion_face_sd15.py",
    "training/laion-face-processed/prompt.jsonl",
    "training/laion-face-processed/metadata.json",
]:
    zf.write(file, arcname=file)

# Add source images:
for source_image in glob("./training/laion-face-processed/source/*.jpg"):
    zf.write(source_image, arcname=source_image)

zf.close()
