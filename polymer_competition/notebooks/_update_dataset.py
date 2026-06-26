"""Update the Kaggle code dataset with the fixed descriptors.py."""
import shutil, tempfile, zipfile, os, json
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()

# Create temp dir and copy relevant files
tmp = Path(tempfile.mkdtemp())
src = Path(r'D:\Parth\Poly\polymer_competition')

# Copy files preserving relative paths
for f in src.rglob('*'):
    if f.is_file() and '.git' not in str(f) and '__pycache__' not in str(f):
        rel = f.relative_to(src)
        dest = tmp / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, dest)

print(f'Copied files to {tmp}')
print(f'Total files: {len(list(tmp.rglob("*")))}')

# Create dataset metadata
os.chdir(str(tmp))
# Remove old kernel-metadata.json if present
old_meta = tmp / 'kernel-metadata.json'
if old_meta.exists():
    old_meta.unlink()

from kaggle.models.dataset_new_request import DatasetNewRequest
from kaggle.models.dataset_upload_file import DatasetUploadFile

# Create new version
api.dataset_create_version(str(tmp), 'Fixed descriptors.py - skip EState hang, add timeout')
print('Dataset version created')
