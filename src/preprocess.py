import hashlib
from pathlib import Path
import urllib.request
import zipfile
import pandas as pd

DATA_DIR = Path("data")
CSV_PATH = DATA_DIR / "bank-full.csv"
URL = "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip"
EXPECTED_SHA256 = "d1513ec63b385506f7cfce9f2c5caa9fe99e7ba4e8c3fa264b3aaf0f849ed32d"


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_data() -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not CSV_PATH.exists():
        outer_zip = DATA_DIR / "bank_marketing.zip"
        inner_zip = DATA_DIR / "bank.zip"
        try:
            urllib.request.urlretrieve(URL, outer_zip)
            with zipfile.ZipFile(outer_zip, "r") as zf:
                with zf.open("bank.zip") as inner_bytes:
                    inner_zip.write_bytes(inner_bytes.read())
            with zipfile.ZipFile(inner_zip, "r") as zf:
                zf.extract("bank-full.csv", path=DATA_DIR)
        finally:
            outer_zip.unlink(missing_ok=True)
            inner_zip.unlink(missing_ok=True)

    actual_hash = sha256_of(CSV_PATH)
    if actual_hash != EXPECTED_SHA256:
        raise ValueError(
            f"SHA-256 mismatch for {CSV_PATH}: expected {EXPECTED_SHA256}, got {actual_hash}"
        )

    return CSV_PATH


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";")
    df = df.drop(columns=["duration"])
    return df


if __name__ == "__main__":
    df = load_data(fetch_data())

    print(df.head())
    print("\n--- Data Loaded ---")
    print(df.shape)
    print(df["y"].value_counts(normalize=True))