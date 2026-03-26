"""Download TMDB 5000 dataset and process into movies.jsonl."""

import json
import sys
from pathlib import Path

import kagglehub
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_FILE = PROCESSED_DIR / "movies.jsonl"


def download_dataset() -> Path:
    """Download TMDB 5000 dataset via kagglehub."""
    print("Downloading TMDB 5000 dataset...")
    path = kagglehub.dataset_download("tmdb/tmdb-movie-metadata")
    print(f"Dataset downloaded to: {path}")
    return Path(path)


def parse_json_column(value: str) -> list[str]:
    """Parse JSON string column into list of name strings."""
    if pd.isna(value):
        return []
    try:
        items = json.loads(value)
        return [item["name"] for item in items if "name" in item]
    except (json.JSONDecodeError, TypeError):
        return []


def process_movies(dataset_path: Path) -> pd.DataFrame:
    """Load and process TMDB movies CSV."""
    csv_path = dataset_path / "tmdb_5000_movies.csv"
    if not csv_path.exists():
        candidates = list(dataset_path.rglob("tmdb_5000_movies.csv"))
        if not candidates:
            raise FileNotFoundError(f"tmdb_5000_movies.csv not found in {dataset_path}")
        csv_path = candidates[0]

    print(f"Loading {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} movies")

    df["genres_list"] = df["genres"].apply(parse_json_column)
    df["keywords_list"] = df["keywords"].apply(parse_json_column)

    df["year"] = pd.to_datetime(df["release_date"], errors="coerce").dt.year
    df["year"] = df["year"].fillna(0).astype(int)

    df["duration_min"] = df["runtime"]
    df["rating"] = df["vote_average"]

    before = len(df)
    df = df.dropna(subset=["overview", "runtime"])
    df = df[df["overview"].str.strip().astype(bool)]
    df = df[df["runtime"] > 0]
    print(f"Filtered: {before} → {len(df)} movies (removed {before - len(df)} without overview/runtime)")

    df["genres_str"] = df["genres_list"].apply(lambda g: ", ".join(g))
    df["keywords_str"] = df["keywords_list"].apply(lambda k: ", ".join(k[:15]))

    df["text_for_embedding"] = df.apply(
        lambda r: (
            f"{r['title']} ({int(r['year'])}). "
            f"Genres: {r['genres_str']}. "
            f"{r['overview']}. "
            f"Tags: {r['keywords_str']}"
        ),
        axis=1,
    )

    df["genres_pipe"] = df["genres_list"].apply(lambda g: "|" + "|".join(g) + "|" if g else "")
    df["tags_pipe"] = df["keywords_list"].apply(lambda k: "|" + "|".join(k[:15]) + "|" if k else "")

    return df


def save_jsonl(df: pd.DataFrame, output_path: Path):
    """Save processed movies to JSONL."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    records = []
    for _, row in df.iterrows():
        record = {
            "id": str(int(row["id"])),
            "title": row["title"],
            "year": int(row["year"]),
            "duration_min": int(row["duration_min"]),
            "rating": round(float(row["rating"]), 1),
            "genres": row["genres_str"],
            "genres_pipe": row["genres_pipe"],
            "keywords": row["keywords_str"],
            "tags_pipe": row["tags_pipe"],
            "overview": row["overview"],
            "text_for_embedding": row["text_for_embedding"],
        }
        records.append(record)

    with open(output_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Saved {len(records)} movies to {output_path}")


def main():
    dataset_path = download_dataset()
    df = process_movies(dataset_path)
    save_jsonl(df, OUTPUT_FILE)
    print("Done!")


if __name__ == "__main__":
    main()
