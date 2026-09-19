import bz2

import pandas as pd
import pytest

from src.preprocessing import ReviewPreprocessor


@pytest.fixture
def preprocessor():
    return ReviewPreprocessor()


def test_clean_text_removes_html_and_stopwords(preprocessor):
    result = preprocessor.clean_text("<p>This is a GREAT product!</p>")

    assert "great" in result
    assert "product" in result
    assert "this" not in result


@pytest.mark.parametrize("value", [None, "", "   ", 123])
def test_clean_text_handles_invalid_or_empty_values(preprocessor, value):
    assert preprocessor.clean_text(value) == ""


def test_clean_text_removes_punctuation_and_lemmatizes(preprocessor):
    result = preprocessor.clean_text("Dogs, cats, and running!")

    assert result == "dog cat running"


def test_extract_metadata_features(preprocessor):
    df = pd.DataFrame({
        "review_text": ["Excellent product", "Very bad"],
        "review_title": ["Great", "Poor"],
    })

    result = preprocessor.extract_metadata_features(df)

    assert "char_count" in result.columns
    assert "word_count" in result.columns
    assert "title_word_count" in result.columns
    assert "avg_word_length" in result.columns
    assert result.loc[0, "word_count"] == 2
    assert result.loc[0, "title_word_count"] == 1
    assert result.loc[0, "char_count"] == len("Excellent product")


def test_parse_fasttext_file(preprocessor, tmp_path):
    input_file = tmp_path / "reviews.txt"
    input_file.write_text(
        "__label__1 Great product: Works perfectly\n"
        "__label__2 Bad product: Stopped working\n",
        encoding="utf-8",
    )

    result = preprocessor.parse_fasttext_file(str(input_file))

    assert len(result) == 2
    assert list(result["rating"]) == [1, 5]
    assert result.loc[0, "product_id"] == "PROD_000000"
    assert result.loc[0, "review_title"] == "Great product"
    assert result.loc[0, "review_text"] == "Works perfectly"


def test_parse_fasttext_file_skips_malformed_lines(preprocessor, tmp_path):
    input_file = tmp_path / "reviews.txt"
    input_file.write_text(
        "invalid line\n"
        "__label__1 Valid title: Valid body\n"
        "\n"
        "__label__2 No title or body separator\n",
        encoding="utf-8",
    )

    result = preprocessor.parse_fasttext_file(str(input_file))

    assert len(result) == 2
    assert result.loc[0, "review_title"] == "Valid title"
    assert result.loc[1, "review_title"] == ""
    assert result.loc[1, "review_text"] == "No title or body separator"


def test_parse_fasttext_file_respects_max_samples(preprocessor, tmp_path):
    input_file = tmp_path / "reviews.txt"
    input_file.write_text(
        "__label__1 First: Review one\n"
        "__label__2 Second: Review two\n"
        "__label__1 Third: Review three\n",
        encoding="utf-8",
    )

    result = preprocessor.parse_fasttext_file(str(input_file), max_samples=2)

    assert len(result) == 2


def test_parse_bz2_file(preprocessor, tmp_path):
    input_file = tmp_path / "reviews.txt.bz2"
    content = "__label__1 Compressed title: Compressed review\n"

    with bz2.open(input_file, "wt", encoding="utf-8") as file:
        file.write(content)

    result = preprocessor.parse_fasttext_file(str(input_file))

    assert len(result) == 1
    assert result.loc[0, "rating"] == 1
    assert result.loc[0, "review_text"] == "Compressed review"


def test_run_pipeline_removes_duplicates_and_writes_csv(
    preprocessor, tmp_path
):
    input_file = tmp_path / "reviews.txt"
    output_file = tmp_path / "cleaned_reviews.csv"

    input_file.write_text(
        "__label__1 Great product: Works well\n"
        "__label__1 Great product: Works well\n"
        "__label__2 Poor product: Does not work\n",
        encoding="utf-8",
    )

    result = preprocessor.run_pipeline(
        str(input_file),
        str(output_file),
        max_samples=10,
    )

    assert output_file.exists()
    assert len(result) == 2
    assert len(pd.read_csv(output_file)) == 2
    assert "cleaned_text" in result.columns
    assert "cleaned_title" in result.columns
    assert "avg_word_length" in result.columns


def test_metadata_features_do_not_create_infinite_values(preprocessor):
    df = pd.DataFrame({
        "review_text": [""],
        "review_title": [""],
    })

    result = preprocessor.extract_metadata_features(df)

    assert result.loc[0, "word_count"] == 0
    assert result.loc[0, "char_count"] == 0
    assert pd.notna(result.loc[0, "avg_word_length"])