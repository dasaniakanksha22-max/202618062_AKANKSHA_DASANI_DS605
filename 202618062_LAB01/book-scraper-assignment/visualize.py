"""
Task 3: Visualization and Analysis
Generates the required plots + word cloud from the cleaned dataset.
Run this AFTER preprocessing.py.
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

CLEAN_PATH = "data/cleaned_books.csv"
PLOTS_DIR = "plots"

os.makedirs(PLOTS_DIR, exist_ok=True)
sns.set_theme(style="whitegrid")


def main():
    df = pd.read_csv(CLEAN_PATH)

    # 1. Price distribution
    plt.figure(figsize=(8, 5))
    sns.histplot(df["price"], bins=30, kde=True, color="steelblue")
    plt.title("Price Distribution")
    plt.xlabel("Price (£)")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/price_distribution.png")
    plt.close()

    # 2. Rating distribution
    plt.figure(figsize=(7, 5))
    sns.countplot(x="rating_numeric", data=df, palette="viridis")
    plt.title("Rating Distribution")
    plt.xlabel("Rating (stars)")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/rating_distribution.png")
    plt.close()

    # 3. Average price by category (top 15 for readability)
    plt.figure(figsize=(10, 6))
    avg_price = (
        df.groupby("category")["price"].mean().sort_values(ascending=False).head(15)
    )
    avg_price.plot(kind="bar", color="coral")
    plt.title("Average Price by Category (Top 15)")
    plt.ylabel("Average Price (£)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/avg_price_by_category.png")
    plt.close()

    # 4. Relationship plot: Price vs Rating
    plt.figure(figsize=(8, 5))
    sns.boxplot(x="rating_numeric", y="price", data=df, palette="coolwarm")
    plt.title("Price vs Rating")
    plt.xlabel("Rating (stars)")
    plt.ylabel("Price (£)")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/price_vs_rating.png")
    plt.close()

    # 5. Word cloud from descriptions (required — used as the textual source)
    text = " ".join(df["description"].dropna().astype(str))
    wc = WordCloud(width=1000, height=600, background_color="white").generate(text)
    plt.figure(figsize=(12, 7))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title("Word Cloud of Book Descriptions")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/wordcloud.png")
    plt.close()

    # Summary stats printed for you to quote in your insights
    print("=== Summary Statistics ===")
    print(df[["price", "rating_numeric", "stock_count"]].describe())
    print("\nMost represented categories:")
    print(df["category"].value_counts().head(5))
    print("\nAll plots saved to /plots")


if __name__ == "__main__":
    main()
