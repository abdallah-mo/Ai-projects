"""IMDB Reviews Prediction
===========================

This module provides a simple pipeline for sentiment analysis on the
IMDB movie reviews dataset. The original notebook loaded a CSV file,
performed basic exploratory analysis, tokenised the review text, built
an LSTM-based neural network using Keras and TensorFlow, trained the
model and evaluated it on a held‑out test set. This module extracts
that logic into reusable functions suitable for a standalone Python
script or for importing into other projects.

The typical workflow is:

* Load the data from a CSV file.
* Preprocess the reviews: tokenise and pad sequences, encode labels.
* Split the data into training and test sets.
* Build the LSTM model.
* Train the model, optionally using early stopping.
* Evaluate the model on the test set and generate a classification report.
* Optionally plot a confusion matrix.

Example usage from the command line:

```
python imdb_reviews_prediction.py --data-path ./IMDB\ Dataset.csv --epochs 5
```

The script will train a model and output accuracy and classification
metrics. If you wish to integrate this into a larger project or unit
test the functions, import the functions defined here instead of
running the module as a script.

Note: This script expects the input CSV to have two columns named
``review`` and ``sentiment``. The ``sentiment`` column should contain
string labels such as ``"positive"`` and ``"negative"`` which will be
encoded to integers.

"""

from __future__ import annotations

import argparse
import logging
from typing import Tuple, List

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dropout, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)



def load_data(file_path: str) -> pd.DataFrame:
    """Load the IMDB dataset from a CSV file.

    Parameters
    ----------
    file_path : str
        Path to the CSV file containing the dataset.

    Returns
    -------
    pd.DataFrame
        A pandas DataFrame with at least ``review`` and ``sentiment`` columns.

    Notes
    -----
    The CSV is read with ``engine='python'`` and ``on_bad_lines='skip'`` to
    bypass problematic rows. Adjust these parameters as needed for your
    dataset.
    """
    logger.debug("Loading data from %s", file_path)
    df = pd.read_csv(file_path, engine="python", on_bad_lines="skip")
    # Ensure required columns exist
    if 'review' not in df.columns or 'sentiment' not in df.columns:
        raise ValueError(
            "Input CSV must contain 'review' and 'sentiment' columns"
        )
    logger.info("Loaded %d records from %s", len(df), file_path)
    return df



def preprocess_data(
    df: pd.DataFrame,
    max_words: int = 10_000,
    max_len: int = 100,
) -> Tuple[np.ndarray, np.ndarray, Tokenizer, LabelEncoder]:
    """Tokenise text, pad sequences and encode labels.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing ``review`` and ``sentiment`` columns.
    max_words : int, optional
        Maximum number of words to keep in the tokenizer vocabulary. Defaults to 10_000.
    max_len : int, optional
        Maximum length of the padded sequences. Defaults to 100.

    Returns
    -------
    X : np.ndarray
        Array of padded integer sequences representing the reviews.
    y : np.ndarray
        Array of encoded sentiment labels (0 or 1 for binary classification).
    tokenizer : Tokenizer
        Fitted Keras Tokenizer instance.
    label_encoder : LabelEncoder
        Fitted scikit‑learn LabelEncoder instance.
    """
    logger.debug("Fitting tokenizer on %d reviews", len(df))
    tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
    tokenizer.fit_on_texts(df['review'].astype(str).tolist())
    sequences = tokenizer.texts_to_sequences(df['review'].astype(str).tolist())
    logger.debug("Converted texts to sequences")
    padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')

    # Encode labels
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(df['sentiment'].astype(str))
    logger.debug("Encoded sentiment labels")
    return padded_sequences, labels, tokenizer, label_encoder



def build_model(
    vocab_size: int,
    embed_dim: int = 128,
    input_length: int = 100,
    lstm_units: List[int] | Tuple[int, ...] = (128, 64),
    dropout_rate: float = 0.5,
    num_classes: int = 1,
) -> Sequential:
    """Construct and compile an LSTM-based Keras model.

    Parameters
    ----------
    vocab_size : int
        Size of the vocabulary (``max_words`` from the preprocessing step).
    embed_dim : int, optional
        Dimension of the embedding layer. Defaults to 128.
    input_length : int, optional
        Length of input sequences (``max_len``). Defaults to 100.
    lstm_units : list of int, optional
        Hidden units for each LSTM layer. Defaults to two layers with 128 and 64 units.
    dropout_rate : float, optional
        Dropout rate applied after each LSTM layer. Defaults to 0.5.
    num_classes : int, optional
        Number of output classes. Use 1 for binary classification. Defaults to 1.

    Returns
    -------
    Sequential
        A compiled Keras model ready for training.
    """
    logger.debug(
        "Building model with vocab_size=%d, embed_dim=%d, input_length=%d", vocab_size, embed_dim, input_length
    )
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=embed_dim, input_length=input_length))
    for units in lstm_units[:-1]:
        # For all but the last LSTM layer, return sequences so the next layer can receive a 3D input
        model.add(LSTM(units, return_sequences=True))
        model.add(Dropout(dropout_rate))
    # Final LSTM layer does not return sequences
    model.add(LSTM(lstm_units[-1]))
    model.add(Dropout(dropout_rate))
    # Dense output layer
    if num_classes == 1:
        activation = 'sigmoid'
        loss = 'binary_crossentropy'
    else:
        activation = 'softmax'
        loss = 'sparse_categorical_crossentropy'
    model.add(Dense(num_classes, activation=activation))
    model.compile(loss=loss, optimizer='adam', metrics=['accuracy'])
    logger.info("Model built and compiled")
    return model


def train_model(
    model: Sequential,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 5,
    batch_size: int = 64,
    early_stopping: bool = True,
) -> object:
    """Train the Keras model.

    Parameters
    ----------
    model : Sequential
        The compiled Keras model to train.
    X_train, y_train : array-like
        Training data and labels.
    X_val, y_val : array-like
        Validation data and labels for early stopping and evaluation.
    epochs : int, optional
        Number of training epochs. Defaults to 5.
    batch_size : int, optional
        Batch size for training. Defaults to 64.
    early_stopping : bool, optional
        Whether to use early stopping callback on validation loss. Defaults to True.

    Returns
    -------
    object
        A Keras History object containing details of the training process.
    """
    callbacks = []
    if early_stopping:
        from tensorflow.keras.callbacks import EarlyStopping

        callbacks.append(EarlyStopping(monitor='val_loss', patience=3, min_delta=1e-4, restore_best_weights=True))
    logger.info(
        "Starting training for %d epochs (early_stopping=%s)", epochs, early_stopping
    )
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )
    logger.info("Training complete")
    return history



def evaluate_model(model: Sequential, X_test: np.ndarray, y_test: np.ndarray) -> float:
    """Evaluate the model on the test set and return accuracy.

    Parameters
    ----------
    model : Sequential
        The trained Keras model.
    X_test, y_test : array-like
        Test data and labels.

    Returns
    -------
    float
        Accuracy of the model on the test set.
    """
    logger.debug("Evaluating model on test data")
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    logger.info("Test accuracy: %.4f", accuracy)
    return accuracy



def predict_and_report(
    model: Sequential,
    X_test: np.ndarray,
    y_test: np.ndarray,
    label_encoder: LabelEncoder,
) -> Tuple[np.ndarray, str]:
    """Generate predictions and classification report.

    Parameters
    ----------
    model : Sequential
        Trained Keras model.
    X_test, y_test : array-like
        Test data and true labels.
    label_encoder : LabelEncoder
        Fitted LabelEncoder used for decoding predictions if needed.

    Returns
    -------
    y_pred : np.ndarray
        Predicted class labels as integers.
    report : str
        Text classification report from scikit‑learn.
    """
    logger.debug("Predicting on test data")
    predictions = model.predict(X_test, verbose=0)
    # Flatten predictions and threshold at 0.5 for binary classification
    if predictions.ndim == 2 and predictions.shape[1] == 1:
        y_pred = np.round(predictions).astype(int).flatten()
    else:
        y_pred = predictions.argmax(axis=1)
    report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)
    logger.info("Generated classification report")
    return y_pred, report



def plot_confusion_matrix(cm: np.ndarray, class_names: List[str]) -> None:
    """Plot a confusion matrix using seaborn.

    Parameters
    ----------
    cm : np.ndarray
        Confusion matrix array (2x2 for binary classification).
    class_names : list of str
        Names of the classes corresponding to the rows and columns of the matrix.

    Returns
    -------
    None
        Displays a heatmap using matplotlib.
    """
    logger.debug("Plotting confusion matrix")
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.show()


def parse_args(args: List[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments for running the script.

    Parameters
    ----------
    args : list of str, optional
        Argument list to parse. If None, defaults to sys.argv[1:].

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Train an LSTM model on the IMDB reviews dataset.")
    parser.add_argument(
        '--data-path',
        type=str,
        required=True,
        help="Path to the CSV file containing the IMDB dataset (requires 'review' and 'sentiment' columns).",
    )
    parser.add_argument(
        '--epochs', type=int, default=5, help="Number of training epochs (default: 5)."
    )
    parser.add_argument(
        '--batch-size', type=int, default=64, help="Batch size for training (default: 64)."
    )
    parser.add_argument(
        '--max-words',
        type=int,
        default=10_000,
        help="Maximum number of words to keep in the vocabulary (default: 10000).",
    )
    parser.add_argument(
        '--max-len',
        type=int,
        default=100,
        help="Maximum length of padded sequences (default: 100).",
    )
    parser.add_argument(
        '--no-early-stopping',
        action='store_true',
        help="Disable early stopping during training.",
    )
    parser.add_argument(
        '--plot-cm',
        action='store_true',
        help="Plot the confusion matrix after evaluation.",
    )
    parser.add_argument(
        '--log-level', type=str, default='INFO', help="Logging level (e.g. DEBUG, INFO, WARNING)"
    )
    parsed_args = parser.parse_args(args)
    return parsed_args



def main(cli_args: argparse.Namespace | None = None) -> None:
    """Main entry point for running from the command line.

    Parameters
    ----------
    cli_args : argparse.Namespace, optional
        Parsed command line arguments. If None, command line arguments are parsed
        from ``sys.argv`` automatically.

    Returns
    -------
    None
    """
    if cli_args is None:
        cli_args = parse_args()
    # Configure logging
    logging.basicConfig(level=getattr(logging, cli_args.log_level.upper(), logging.INFO))

    # Load and preprocess data
    df = load_data(cli_args.data_path)
    X, y, tokenizer, label_encoder = preprocess_data(
        df, max_words=cli_args.max_words, max_len=cli_args.max_len
    )

    # Split data into train/validation and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    # Further split the training set into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    logger.info(
        "Data split into %d training samples, %d validation samples and %d test samples",
        len(X_train),
        len(X_val),
        len(X_test),
    )

    # Build and train the model
    model = build_model(
        vocab_size=min(cli_args.max_words, len(tokenizer.word_index) + 1),
        input_length=cli_args.max_len,
    )
    train_model(
        model,
        X_train,
        y_train,
        X_val,
        y_val,
        epochs=cli_args.epochs,
        batch_size=cli_args.batch_size,
        early_stopping=not cli_args.no_early_stopping,
    )

    # Evaluate and report
    accuracy = evaluate_model(model, X_test, y_test)
    y_pred, report = predict_and_report(model, X_test, y_test, label_encoder)
    print("\nTest accuracy: {:.4f}".format(accuracy))
    print("\nClassification report:\n")
    print(report)

    # Optionally plot confusion matrix
    if cli_args.plot_cm:
        cm = confusion_matrix(y_test, y_pred)
        plot_confusion_matrix(cm, class_names=label_encoder.classes_.tolist())



if __name__ == '__main__':
    main()
