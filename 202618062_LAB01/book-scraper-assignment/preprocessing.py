"""
Task 2: Data Preprocessing
Reads the raw scraped CSV, cleans it, converts types, engineers features,
and writes a cleaned CSV ready for visualization/analysis.
"""

import pandas as pd

RAW_PATH = "data/raw_books.csv"
CLEAN_PATH = "data/cleaned_books.csv"


def main():
    # Read raw data
    df = pd.read_csv(RAW_PATH)

    print(f"Total scraped records: {len(df)}")
    print("Missing values per column:")
    print(df.isna().sum())
    print()

    # ----------------------------
    # Clean text columns
    # ----------------------------
    text_cols = ["title", "category", "description"]

    for col in text_cols:
        df[col] = df[col].fillna("").astype(str).str.strip()
        df[col] = df[col].str.replace(r"\s+", " ", regex=True)

    # ----------------------------
    # Keep all books
    # ----------------------------
    df = df.reset_index(drop=True)

    # ----------------------------
    # Handle missing descriptions
    # ----------------------------
    df["description"] = df["description"].replace(
        {
            "": "No description available",
            "nan": "No description available"
        }
    )

    # ----------------------------
    # Convert price to float
    # ----------------------------
    df["price"] = (
        df["price"]
        .astype(str)
        .str.replace(r"[^\d.]", "", regex=True)
        .astype(float)
    )

    # ----------------------------
    # Rating mapping
    # ----------------------------
    rating_map = {
        "One": 1,
        "Two": 2,
        "Three": 3,
        "Four": 4,
        "Five": 5
    }

    if "rating_numeric" not in df.columns:
        df["rating_numeric"] = df["rating"].map(rating_map)

    # ----------------------------
    # Handle missing availability
    # ----------------------------
    df["availability"] = df["availability"].fillna("")

    df["stock_count"] = (
        df["availability"]
        .astype(str)
        .str.extract(r"\((\d+) available\)")[0]
        .fillna(0)
        .astype(int)
    )

    # ----------------------------
    # Feature Engineering
    # ----------------------------
    df["description_word_count"] = df["description"].apply(
        lambda x: len(str(x).split())
    )

    df["price_band"] = pd.cut(
        df["price"],
        bins=[0, 20, 35, 50, 1000],
        labels=["Low", "Medium", "High", "Premium"]
    )

    df["affordability_score"] = (
        df["rating_numeric"] / df["price"]
    )

    df["value_score"] = (
        (df["rating_numeric"] * (df["stock_count"] + 1))
        / df["price"]
    )

    df["recommended"] = (
        (df["rating_numeric"] >= 4)
        & (df["price"] < df["price"].median())
    )

    # ----------------------------
    # Save cleaned data
    # ----------------------------
    df.to_csv(CLEAN_PATH, index=False)

    print(f"\nCleaned dataset saved to {CLEAN_PATH}")
    print(f"Final records: {len(df)}")


if __name__ == "__main__":
    main()