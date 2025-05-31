import os
import logging
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import string
import nltk

nltk.download("stopwords")
nltk.download("punkt")
nltk.download("punkt_tab")

# Ensure logs
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

# Setting up logger
logger = logging.getLogger("data_preprocessing")
logger.setLevel("DEBUG")

console_handler = logging.StreamHandler()
console_handler.setLevel("DEBUG")

log_file_path = os.path.join(log_dir, "data_preprocessing.log")
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel("DEBUG")

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def transform_text(text):
    """
    Transforms the input text by converting it to lowercase, tokenizing, removing stopwords and punctuation, and stemming.
    """
    ps = PorterStemmer()
    # Convert to lowercase
    text = text.lower()
    # Tokenize the text
    text = nltk.word_tokenize(text)
    # Remove non-alphanumeric tokens
    text = [word for word in text if word.isalnum()]
    # Remove stopwords and punctuation
    text = [
        word
        for word in text
        if word not in stopwords.words("english") and word not in string.punctuation
    ]
    # Stem the words
    text = [ps.stem(word) for word in text]
    # Join the tokens back into a single string
    return " ".join(text)


def preprocess_df(df, text_column="text", target_column="target"):
    """Preprocess the df by encoding the target col,removing dups, adntransforming text col"""
    try:
        logger.debug("Starting Preprocessing for DF")

        # encode the target column
        encoder = LabelEncoder()
        df[target_column] = encoder.fit_transform(df[target_column])

        logger.debug("Target column encoded")

        # Remoce Duplicate rows
        df = df.drop_duplicates(keep="first")
        logger.debug("Dropped Duplicate rows from df")

        # Apply text transform to the specified text column
        df.loc[:, text_column] = df[text_column].apply(transform_text)
        logger.debug("Text column transformed")
        return df
    except KeyError as e:
        logger.error("Column not Found: %s", e)
        raise
    except Exception as e:
        logger.error("Unexpected Error Occured")
        raise


def main(text_column="text", target_column="target"):
    """Main Function to load raw data,preprocess it and save the processed data"""
    try:
        train_data = pd.read_csv("./data/raw/train.csv")
        test_data = pd.read_csv("./data/raw/test.csv")
        logger.debug("train and test data loaded")

        # Transform data

        train_processed_data = preprocess_df(train_data, text_column, target_column)
        logger.debug("Train Transformed Successfully")

        test_processed_data = preprocess_df(test_data, text_column, target_column)
        logger.debug("Test Data Transformed Successfully")

        # Store Preprocessed data
        data_path = os.path.join("./data", "interim")
        os.makedirs(data_path, exist_ok=True)

        train_processed_data.to_csv(
            os.path.join(data_path, "train_preprocessed.csv"), index=False
        )
        test_processed_data.to_csv(
            os.path.join(data_path, "test_preprocessed.csv"), index=False
        )

        logger.debug(
            "Preprocesed Train and Test Data Saved Successfully to %s", data_path
        )

    except FileNotFoundError as e:
        logger.error("File Not Found %s", e)
        raise
    except Exception as e:
        logger.error("Unexpected Error %s", e)
        print(f"Erro:{e}")


if __name__ == "__main__":
    main()
