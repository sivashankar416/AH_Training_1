import bz2
import re
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm

# Download required NLTK resources
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

class ReviewPreprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()

    def parse_fasttext_file(self, file_path, max_samples=100000):
        """
        Parses raw fastText format lines (__label__1/__label__2 Title: ReviewBody)
        into structured pandas DataFrame columns.
        """
        records = []
        print(f"Parsing fastText file: {file_path}")
        
        # Open bz2 compressed file or standard text file
        open_func = bz2.open if file_path.endswith('.bz2') else open
        
        with open_func(file_path, 'rt', encoding='utf-8', errors='ignore') as f:
            for idx, line in enumerate(f):
                if max_samples and idx >= max_samples:
                    break
                
                # Extract label and text payload
                parts = line.strip().split(' ', 1)
                if len(parts) < 2:
                    continue
                
                raw_label, content = parts[0], parts[1]

                # Reject malformed or unsupported labels
                if raw_label not in {"__label__1", "__label__2"}:
                    continue

                # Map fastText labels
                label_val = 1 if raw_label == "__label__1" else 5
                
                # Split title and body if separated by colon
                if ':' in content:
                    split_content = content.split(':', 1)
                    review_title = split_content[0].strip()
                    review_text = split_content[1].strip()
                else:
                    review_title = ""
                    review_text = content.strip()
                
                records.append({
                    'product_id': f"PROD_{idx:06d}",
                    'review_title': review_title,
                    'review_text': review_text,
                    'rating': label_val
                })
                
        df = pd.DataFrame(records)
        print(f"Successfully loaded {len(df)} records.")
        return df

    def clean_text(self, text):
        """
        Cleans HTML, non-alphabetical characters, converts to lowercase,
        removes stopwords, and applies lemmatization.
        """
        if not isinstance(text, str) or not text.strip():
            return ""
        
        # 1. Remove HTML tags
        text = BeautifulSoup(text, "html.parser").get_text()
        
        # 2. Convert to lowercase & remove non-alphabet characters
        text = re.sub(r'[^a-zA-Z\s]', '', text).lower()
        
        # 3. Tokenize, remove stopwords, apply lemmatization
        words = text.split()
        cleaned_words = [
            self.lemmatizer.lemmatize(word) 
            for word in words 
            if word not in self.stop_words and len(word) > 1
        ]
        
        return " ".join(cleaned_words)

    def extract_metadata_features(self, df):
        """
        Extracts structural features from review title and body.
        """
        print("Extracting metadata features...")
        df['char_count'] = df['review_text'].apply(len)
        df['word_count'] = df['review_text'].apply(lambda x: len(x.split()))
        df['title_word_count'] = df['review_title'].apply(lambda x: len(x.split()))
        df['avg_word_length'] = df['char_count'] / (df['word_count'] + 1e-5)
        return df

    def run_pipeline(self, raw_file_path, output_csv_path, max_samples=50000):
        # 1. Load Data
        df = self.parse_fasttext_file(raw_file_path, max_samples=max_samples)
        
        # 2. Handle Duplicates & Missing Values
        print("Handling missing values and duplicates...")
        df.dropna(subset=['review_text'], inplace=True)
        initial_count = len(df)
        df.drop_duplicates(subset=['review_title', 'review_text'], inplace=True)
        print(f"Removed {initial_count - len(df)} duplicate reviews.")

        # 3. Text Preprocessing
        print("Cleaning review titles and text...")
        tqdm.pandas(desc="Cleaning Text")
        df['cleaned_text'] = df['review_text'].progress_apply(self.clean_text)
        df['cleaned_title'] = df['review_title'].progress_apply(self.clean_text)

        # 4. Extract Structural Features
        df = self.extract_metadata_features(df)

        # 5. Save Processed Dataset
        df.to_csv(output_csv_path, index=False)
        print(f"Preprocessed dataset saved to: {output_csv_path}")
        return df

if __name__ == "__main__":
    preprocessor = ReviewPreprocessor()

    raw_input_path = "data/raw/train.ft.txt.bz2"
    processed_output_path = "data/processed/cleaned_amazon_reviews.csv"

    preprocessor.run_pipeline(
        raw_input_path,
        processed_output_path,
        max_samples=20000
    )