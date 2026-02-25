# IMDB Reviews Sentiment Analysis

This repository contains a Python module and command‑line script for
performing sentiment analysis on movie reviews from the IMDB dataset.
The original project was provided as a Jupyter notebook. It has been
refactored into a more modular and reusable structure suitable for
inclusion in a GitHub repository.

## Overview

The goal of this project is to train a neural network that
classifies movie reviews as positive or negative. The pipeline
implemented in `imdb_reviews_prediction.py` includes data loading,
tokenisation, padding of sequences, label encoding, model construction
with Keras (using an embedding layer followed by LSTM layers),
training with optional early stopping, evaluation and metric reporting,
and optional plotting of a confusion matrix.

## Requirements

The code requires Python 3.8 or newer and depends on several
third‑party libraries:

* **pandas** for data loading and manipulation.
* **numpy** for numerical operations.
* **scikit‑learn** for label encoding, train/test splitting and
  evaluation metrics.
* **TensorFlow/Keras** for building and training the neural network.
* **Matplotlib** and **Seaborn** for plotting (optional).

To install the necessary dependencies, you can use pip:

```bash
pip install pandas numpy scikit-learn tensorflow matplotlib seaborn
```

Depending on your environment, you might prefer installing
`tensorflow-cpu` instead of the full GPU‑enabled `tensorflow` package.

## Dataset

The script expects a CSV file with at least two columns:

* `review` – the text of the movie review.
* `sentiment` – the sentiment label (e.g., `"positive"` or `"negative"`).

You can download the IMDB dataset from Kaggle or another source and
save it locally as a CSV file. The example commands below use
`./IMDB Dataset.csv` as a placeholder; replace it with the actual path
to your CSV.

## Usage

The main functionality is provided by the `imdb_reviews_prediction.py`
script. You can run it from the command line to train a model and
evaluate its performance. The script offers several command‑line
options:

```
usage: imdb_reviews_prediction.py [-h] --data-path DATA_PATH [--epochs EPOCHS]
                                  [--batch-size BATCH_SIZE]
                                  [--max-words MAX_WORDS] [--max-len MAX_LEN]
                                  [--no-early-stopping] [--plot-cm]
                                  [--log-level LOG_LEVEL]

Train an LSTM model on the IMDB reviews dataset.

optional arguments:
  -h, --help            show this help message and exit
  --data-path DATA_PATH
                        Path to the CSV file containing the IMDB dataset
                        (requires 'review' and 'sentiment' columns).
  --epochs EPOCHS       Number of training epochs (default: 5).
  --batch-size BATCH_SIZE
                        Batch size for training (default: 64).
  --max-words MAX_WORDS
                        Maximum number of words to keep in the vocabulary
                        (default: 10000).
  --max-len MAX_LEN     Maximum length of padded sequences (default: 100).
  --no-early-stopping   Disable early stopping during training.
  --plot-cm             Plot the confusion matrix after evaluation.
  --log-level LOG_LEVEL
                        Logging level (e.g. DEBUG, INFO, WARNING)
```

### Training and Evaluation

To train a model using the default parameters on your dataset and
display evaluation metrics, run:

```bash
python imdb_reviews_prediction.py \
  --data-path "./IMDB Dataset.csv" \
  --epochs 5 \
  --batch-size 64 \
  --plot-cm
```

This command will:

1. Load the dataset from the specified CSV file.
2. Tokenise the reviews and encode the labels.
3. Split the data into training, validation and test sets.
4. Build and train an LSTM model using the specified hyperparameters.
5. Evaluate the trained model on the test set, printing the
   classification report and overall accuracy.
6. Optionally display a confusion matrix if `--plot-cm` is provided.

### As a Python Module

If you wish to integrate the functionality into another project, you
can import the module and use the individual functions:

```python
from imdb_reviews_prediction import load_data, preprocess_data, build_model,
    train_model, evaluate_model, predict_and_report, plot_confusion_matrix

# Load and preprocess data
df = load_data('IMDB Dataset.csv')
X, y, tokenizer, label_encoder = preprocess_data(df)

# Split data and build model
...
```

## Notes and Limitations

* The model architecture and parameters used here are relatively
  simple. For better performance, consider experimenting with
  additional layers, different sequence lengths, pre‑trained
  word embeddings (such as GloVe or Word2Vec), or transformers.
* The default early stopping callback restores the best weights based on
  validation loss. Disable early stopping with `--no-early-stopping` if
  you prefer to train for a fixed number of epochs.
* Ensure that your dataset is balanced or apply appropriate
  techniques (such as class weighting or resampling) if the classes are
  imbalanced.

## Repository Structure

```
├── imdb_reviews_prediction.py  # Main script/module
├── README.md                   # Documentation and usage guide
└── IMDB_Reviews_Prediction.ipynb  # Original Jupyter notebook (optional)
```

You may also add a `.gitignore`, `requirements.txt` and other
configuration files when publishing to GitHub.
