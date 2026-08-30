"""Pick N random images per class from the full dataset into data/raw/."""
import argparse, random, shutil
from pathlib import Path

def main():
    p = argparse.ArgumentParser()
    p.add_argument("source")
    p.add_argument("--n", type=int, default=2000)
    p.add_argument("--out", default="data/raw")
    args = p.parse_args()
    src, out = Path(args.source), Path(args.out)
    for cls in ["cat", "dog"]:
        folders = [d for d in src.rglob("*") if d.is_dir() and cls in d.name.lower()]
        if not folders:
            raise SystemExit("no folder found for " + cls)
        images = [q for q in folders[0].iterdir() if q.suffix.lower() in (".jpg", ".jpeg", ".png")]
        random.seed(42)
        sample = random.sample(images, min(args.n, len(images)))
        dest = out / (cls + "s")
        dest.mkdir(parents=True, exist_ok=True)
        count = 0
        for q in sample:
            try:
                shutil.copy(q, dest / q.name); count += 1
            except OSError:
                continue
        print(cls, count)

if __name__ == "__main__":
    main()
