from pathlib import Path
import urllib.request
import pandas as pd

DATA_URL = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_CSV_PATH = BASE_DIR / "data" / "raw" / "creditcard.csv"
PROCESSED_PARQUET_PATH = BASE_DIR / "data" / "processed" / "creditcard.parquet"


def download_raw_data(url: str = DATA_URL, dest_path: Path = RAW_CSV_PATH) -> Path:
    """Downloads the raw dataset if it does not already exist in data/raw/."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if dest_path.exists() and dest_path.stat().st_size > 0:
        print(f"[OK] Raw file already found: {dest_path} ({dest_path.stat().st_size / (1024**2):.2f} MB)")
        return dest_path

    print(f"[INFO] Downloading data from {url} to {dest_path}...")
    urllib.request.urlretrieve(url, dest_path)
    print(f"[OK] Download completed: {dest_path} ({dest_path.stat().st_size / (1024**2):.2f} MB)")
    return dest_path


def convert_to_parquet(csv_path: Path = RAW_CSV_PATH, parquet_path: Path = PROCESSED_PARQUET_PATH) -> pd.DataFrame:
    """Reads the raw CSV, validates basic integrity, and saves it in Parquet format."""
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Reading raw CSV: {csv_path}...")
    df = pd.read_csv(csv_path)

    # Lightweight dtype optimization on the target variable
    if "Class" in df.columns:
        df["Class"] = df["Class"].astype("int8")

    print(f"[INFO] Saving optimized Parquet file to: {parquet_path}...")
    df.to_parquet(parquet_path, engine="pyarrow", index=False)
    print(
        f"[OK] Parquet successfully generated ({parquet_path.stat().st_size / (1024**2):.2f} MB) | "
        f"Shape: {df.shape[0]:,} rows x {df.shape[1]} columns"
    )
    return df


def main() -> None:
    csv_path = download_raw_data()
    df = convert_to_parquet(csv_path)

    frauds = int(df["Class"].sum())
    total = len(df)
    null_count = int(df.isna().sum().sum())
    print("\n=== Dataset Validation Summary ===")
    print(f"Total transactions : {total:,}")
    print(f"Total frauds       : {frauds:,} ({(frauds / total) * 100:.4f}%)")
    print(f"Missing values     : {null_count}")
    print(f"Columns            : {list(df.columns[:5])} ... {list(df.columns[-3:])}")


if __name__ == "__main__":
    main()
