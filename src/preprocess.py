"""Resize images to 224x224 and split into train/val/test (80/10/10)."""
import random
from pathlib import Path
import numpy as np
from PIL import Image

IMG_SIZE = (224, 224)

def preprocess_image(img):
    img = img.convert("RGB").resize(IMG_SIZE)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr

def split_files(files, seed=42):
    files = sorted(files)
    random.Random(seed).shuffle(files)
    n = len(files)
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)
    return {
        "train": files[:n_train],
        "val": files[n_train:n_train + n_val],
        "test": files[n_train + n_val:],
    }

def main():
    raw = Path("data/raw")
    processed = Path("data/processed")
    for cls in ["cats", "dogs"]:
        files = [p for p in (raw / cls).iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
        splits = split_files(files)
        for split, file_list in splits.items():
            dest = processed / split / cls
            dest.mkdir(parents=True, exist_ok=True)
            count = 0
            for p in file_list:
                try:
                    arr = preprocess_image(Image.open(p))
                    Image.fromarray((arr * 255).astype(np.uint8)).save(dest / (p.stem + ".jpg"))
                    count += 1
                except Exception:
                    continue
            print(split, cls, count)

if __name__ == "__main__":
    main()
