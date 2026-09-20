import pandas as pd

from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split


DATA_PATH = "data/processed/cleaned_amazon_reviews.csv"


def main() -> None:
    print("=" * 60)
    print("AMAZON ML - TITLE + REVIEW EXPERIMENT")
    print("=" * 60)

    # ---------------------------------------------------------
    # 1. Load data
    # ---------------------------------------------------------
    df = pd.read_csv(DATA_PATH)

    review_text = df["cleaned_text"].fillna("")
    review_title = df["cleaned_title"].fillna("")
    y = df["rating"]

    print(f"\nTotal samples: {len(df)}")

    # ---------------------------------------------------------
    # 2. Stratified split BEFORE fitting vectorizers
    # ---------------------------------------------------------
    (
        train_review,
        valid_review,
        train_title,
        valid_title,
        y_train,
        y_valid,
    ) = train_test_split(
        review_text,
        review_title,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    print(f"Training samples: {len(train_review)}")
    print(f"Validation samples: {len(valid_review)}")

    # ---------------------------------------------------------
    # 3. Review TF-IDF
    # ---------------------------------------------------------
    review_vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=10000,
        min_df=5,
        max_df=0.8,
        sublinear_tf=True,
    )

    print("\nFitting review TF-IDF...")

    X_train_review = review_vectorizer.fit_transform(train_review)
    X_valid_review = review_vectorizer.transform(valid_review)

    # ---------------------------------------------------------
    # 4. Title TF-IDF
    # ---------------------------------------------------------
    title_vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=2000,
        min_df=3,
        max_df=0.9,
        sublinear_tf=True,
    )

    print("Fitting title TF-IDF...")

    X_train_title = title_vectorizer.fit_transform(train_title)
    X_valid_title = title_vectorizer.transform(valid_title)

    # ---------------------------------------------------------
    # 5. Combine sparse features
    # ---------------------------------------------------------
    X_train = hstack(
        [X_train_review, X_train_title]
    ).tocsr()

    X_valid = hstack(
        [X_valid_review, X_valid_title]
    ).tocsr()

    print(f"\nCombined training shape: {X_train.shape}")
    print(f"Combined validation shape: {X_valid.shape}")

    # ---------------------------------------------------------
    # 6. Logistic Regression
    # ---------------------------------------------------------
    print("\nTraining Logistic Regression...")

    model = LogisticRegression(
        C=1.0,
        max_iter=1000,
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_valid)

    macro_f1 = f1_score(
        y_valid,
        predictions,
        average="macro",
    )

    # ---------------------------------------------------------
    # 7. Compare against baseline
    # ---------------------------------------------------------
    print("\n" + "=" * 60)
    print("RESULT")
    print("=" * 60)

    print(f"Baseline (review only): {0.8424:.4f}")
    print(f"Title + review:        {macro_f1:.4f}")

    improvement = (
        (macro_f1 - 0.8424) / 0.8424
    ) * 100

    print(f"Improvement:            {improvement:+.2f}%")


if __name__ == "__main__":
    main()