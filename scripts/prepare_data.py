# scripts/prepare_data.py
import os, csv, argparse
from sklearn.model_selection import train_test_split

def make_manifest(raw_dir, out_dir, val_size=0.1, test_size=0.1, seed=42):
    os.makedirs(out_dir, exist_ok=True)
    items = []
    for label in os.listdir(raw_dir):
        cls_dir = os.path.join(raw_dir, label)
        if not os.path.isdir(cls_dir): continue
        for fn in os.listdir(cls_dir):
            if fn.lower().endswith((".jpg","jpeg","png")):
                items.append((os.path.join(cls_dir, fn), label))
    train, rest = train_test_split(items, test_size=val_size+test_size, random_state=seed, stratify=[l for _,l in items])
    val_pct = val_size / (val_size+test_size)
    val, test = train_test_split(rest, test_size=val_pct, random_state=seed, stratify=[l for _,l in rest])

    for split, rows in [("train", train), ("val", val), ("test", test)]:
        path = os.path.join(out_dir, f"{split}.csv")
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["path","label"])
            w.writerows(rows)
        print(f"Wrote {len(rows)} rows to {path}")

if __name__=="__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--raw_dir",     required=True)
    p.add_argument("--out_dir",     required=True)
    p.add_argument("--val_size",    type=float, default=0.1)
    p.add_argument("--test_size",   type=float, default=0.1)
    args = p.parse_args()
    make_manifest(args.raw_dir, args.out_dir, args.val_size, args.test_size)
