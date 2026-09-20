import re
from collections import Counter

import pandas as pd
import matplotlib.pyplot as plt


DATA_PATH = "data/processed/cleaned_amazon_reviews.csv"


def get_top_bigrams(texts, top_n=20):
    bigram_counter = Counter()

    for text in texts.dropna():
        words = text.split()

        if len(words) < 2:
            continue

        bigrams = zip(words, words[1:])
        bigram_counter.update(
            " ".join(bigram) for bigram in bigrams
        )

    return bigram_counter.most_common(top_n)


def main():
    print("=" * 60)
    print("AMAZON REVIEW DATASET - EDA")
    print("=" * 60)

    # ---------------------------------------------------------
    # 1. Load dataset
    # ---------------------------------------------------------
    print("\nLoading dataset...")

    df = pd.read_csv(DATA_PATH)

    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")

    print("\nColumns:")
    print(df.columns.tolist())

    # ---------------------------------------------------------
    # 2. Rating distribution
    # ---------------------------------------------------------
    print("\n" + "=" * 60)
    print("1. RATING DISTRIBUTION")
    print("=" * 60)

    rating_counts = df["rating"].value_counts().sort_index()

    print(rating_counts)

    print("\nRating percentages:")
    print(
        (rating_counts / len(df) * 100)
        .round(2)
    )

    rating_counts.plot(kind="bar")

    plt.title("Rating Distribution")
    plt.xlabel("Rating")
    plt.ylabel("Number of Reviews")
    plt.tight_layout()

    plt.savefig(
        "notebooks/rating_distribution.png",
        dpi=150
    )

    plt.show()

    # ---------------------------------------------------------
    # 3. Duplicate reviews
    # ---------------------------------------------------------
    print("\n" + "=" * 60)
    print("2. DUPLICATE REVIEWS")
    print("=" * 60)

    duplicate_count = df.duplicated(
        subset=["review_title", "review_text"]
    ).sum()

    print(f"Duplicate reviews: {duplicate_count}")

    # ---------------------------------------------------------
    # 4. Empty cleaned text
    # ---------------------------------------------------------
    print("\n" + "=" * 60)
    print("3. EMPTY CLEANED TEXT")
    print("=" * 60)

    empty_cleaned = (
        df["cleaned_text"]
        .fillna("")
        .str.strip()
        .eq("")
        .sum()
    )

    print(f"Empty cleaned_text rows: {empty_cleaned}")

    print(
        f"Percentage: "
        f"{empty_cleaned / len(df) * 100:.2f}%"
    )

    # ---------------------------------------------------------
    # 5. Word-count statistics
    # ---------------------------------------------------------
    print("\n" + "=" * 60)
    print("4. WORD COUNT DISTRIBUTION")
    print("=" * 60)

    print(df["word_count"].describe())

    plt.figure(figsize=(10, 5))

    plt.hist(
        df["word_count"],
        bins=50
    )

    plt.title("Review Word Count Distribution")
    plt.xlabel("Word Count")
    plt.ylabel("Number of Reviews")

    plt.tight_layout()

    plt.savefig(
        "notebooks/word_count_distribution.png",
        dpi=150
    )

    plt.show()

    # ---------------------------------------------------------
    # 6. Positive reviews
    # ---------------------------------------------------------
    print("\n" + "=" * 60)
    print("5. TOP POSITIVE BIGRAMS")
    print("=" * 60)

    positive_reviews = df.loc[
        df["rating"] == 5,
        "cleaned_text"
    ]

    positive_bigrams = get_top_bigrams(
        positive_reviews,
        top_n=20
    )

    for bigram, count in positive_bigrams:
        print(f"{bigram:<30} {count}")

    # ---------------------------------------------------------
    # 7. Negative reviews
    # ---------------------------------------------------------
    print("\n" + "=" * 60)
    print("6. TOP NEGATIVE BIGRAMS")
    print("=" * 60)

    negative_reviews = df.loc[
        df["rating"] == 1,
        "cleaned_text"
    ]

    negative_bigrams = get_top_bigrams(
        negative_reviews,
        top_n=20
    )

    for bigram, count in negative_bigrams:
        print(f"{bigram:<30} {count}")

    # ---------------------------------------------------------
    # 8. Summary
    # ---------------------------------------------------------
    print("\n" + "=" * 60)
    print("EDA SUMMARY")
    print("=" * 60)

    print(f"Total reviews       : {len(df)}")
    print(f"Duplicate reviews   : {duplicate_count}")
    print(f"Empty cleaned text  : {empty_cleaned}")
    print(
        f"Average word count  : "
        f"{df['word_count'].mean():.2f}"
    )

    print("\nEDA completed successfully.")


if __name__ == "__main__":
    main()