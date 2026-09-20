import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from src.preprocessing import ReviewPreprocessor


TRAIN_PATH = "data/processed/cleaned_amazon_reviews.csv"
TEST_RAW_PATH = "data/raw/test.ft.txt.bz2"
OUTPUT_PATH = "submissions/predictions.csv"


def main() -> None:
    print("=" * 60)
    print("FINAL AMAZON ML TRAINING + PREDICTION")
    print("=" * 60)

    # ---------------------------------------------------------
    # 1. Load full labeled training data
    # ---------------------------------------------------------
    train_df = pd.read_csv(TRAIN_PATH)

    print(f"\nTraining samples: {len(train_df)}")

    # ---------------------------------------------------------
    # 2. Fit review TF-IDF on ALL training data
    # ---------------------------------------------------------
    review_vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=10000,
        min_df=5,
        max_df=0.8,
        sublinear_tf=True,
    )

    X_train_review = review_vectorizer.fit_transform(
        train_df["cleaned_text"].fillna("")
    )

    # ---------------------------------------------------------
    # 3. Fit title TF-IDF on ALL training data
    # ---------------------------------------------------------
    title_vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=2000,
        min_df=3,
        max_df=0.9,
        sublinear_tf=True,
    )

    X_train_title = title_vectorizer.fit_transform(
        train_df["cleaned_title"].fillna("")
    )

    X_train = hstack(
        [X_train_review, X_train_title]
    ).tocsr()

    y_train = train_df["rating"]

    print(f"Training feature shape: {X_train.shape}")

    # ---------------------------------------------------------
    # 4. Train final model
    # ---------------------------------------------------------
    model = LogisticRegression(
        C=1.0,
        max_iter=1000,
    )

    print("\nTraining final Logistic Regression...")
    model.fit(X_train, y_train)

    # ---------------------------------------------------------
    # 5. Parse test dataset
    # ---------------------------------------------------------
    print("\nParsing test dataset...")

    preprocessor = ReviewPreprocessor()

    test_df = preprocessor.parse_fasttext_file(
        TEST_RAW_PATH,
        max_samples=None,
    )

    print(f"Test samples: {len(test_df)}")

    # ---------------------------------------------------------
    # 6. Clean test text
    # ---------------------------------------------------------
    print("\nCleaning test reviews...")

    test_df["cleaned_text"] = test_df["review_text"].apply(
        preprocessor.clean_text
    )

    test_df["cleaned_title"] = test_df["review_title"].apply(
        preprocessor.clean_text
    )

    # ---------------------------------------------------------
    # 7. Transform test using TRAINED vectorizers
    # ---------------------------------------------------------
    X_test_review = review_vectorizer.transform(
        test_df["cleaned_text"].fillna("")
    )

    X_test_title = title_vectorizer.transform(
        test_df["cleaned_title"].fillna("")
    )

    X_test = hstack(
        [X_test_review, X_test_title]
    ).tocsr()

    print(f"Test feature shape: {X_test.shape}")

    # ---------------------------------------------------------
    # 8. Predict
    # ---------------------------------------------------------
    print("\nGenerating predictions...")

    predictions = model.predict(X_test)

    # ---------------------------------------------------------
    # 9. Create submission file
    # ---------------------------------------------------------
    submission = pd.DataFrame({
        "product_id": test_df["product_id"],
        "rating": predictions,
    })

    submission.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print(f"\nSubmission saved to: {OUTPUT_PATH}")
    print(f"Prediction rows: {len(submission)}")

    print("\nPrediction distribution:")
    print(submission["rating"].value_counts().sort_index())

    print("\nDone!")


if __name__ == "__main__":
    main()