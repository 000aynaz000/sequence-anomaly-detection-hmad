"""Recreate the notebook input from the official ECG5000 .ts files."""
from pathlib import Path
from urllib.request import urlretrieve
import hashlib
import numpy as np

ROOT = Path(__file__).resolve().parent / "ECG5000_prepared"
RAW = ROOT / "raw"
FILES = {"ECG5000_TRAIN.ts": "a1465f2d667b157427fde903eaf2e533",
         "ECG5000_TEST.ts": "c2a9aa8f2141a293493c68e8eea9a06c"}
BASE = "https://zenodo.org/records/11186692/files/"


def read_ts(path):
    rows, labels = [], []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith(("#", "@")):
            continue
        series, label = line.rsplit(":", 1)
        rows.append([float(value) for value in series.split(",")])
        labels.append(int(label))
    return np.asarray(rows, dtype=np.float64), np.asarray(labels, dtype=np.int64)


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    for filename, checksum in FILES.items():
        path = RAW / filename
        if not path.exists():
            print("Downloading", filename)
            urlretrieve(BASE + filename + "?download=1", path)
        if hashlib.md5(path.read_bytes()).hexdigest() != checksum:
            raise ValueError(f"Checksum mismatch: {path}. Replace with the official file.")
    train, train_class = read_ts(RAW / "ECG5000_TRAIN.ts")
    test, test_class = read_ts(RAW / "ECG5000_TEST.ts")
    normal_train = train[train_class == 1]
    labels = (test_class != 1).astype(np.int64)
    assert train.shape == (500, 140) and normal_train.shape == (292, 140)
    assert test.shape == (4500, 140) and labels.sum() == 1873
    assert np.isfinite(normal_train).all() and np.isfinite(test).all()
    np.savez_compressed(ROOT / "ecg5000_oneclass.npz", X_train=normal_train,
                        X_test=test, y_test=labels, original_test_class=test_class)
    print("Prepared", normal_train.shape, test.shape)


if __name__ == "__main__":
    main()
