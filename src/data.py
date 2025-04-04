import math

import numpy as np
from sklearn.datasets import load_iris


def fetch_and_format_iris():
    # Load the Iris dataset as a pandas DataFrame
    iris = load_iris(as_frame=True)
    df = iris.frame

    # Rename columns for clarity
    df.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'target']
    return df


def test_train_split(dataset, test_percent=0.2, random_state=42):
    # Set seed for reproducible shuffling
    np.random.seed(random_state)

    # Shuffle the dataset randomly
    shuffled_dataset = np.random.permutation(dataset)
    num_examples = shuffled_dataset.shape[0]

    # Calculate number of test samples
    test_size = math.ceil(num_examples * test_percent)

    # Features for test set
    X_test = shuffled_dataset[:test_size, : -1]

    # Labels for test set
    y_test = shuffled_dataset[:test_size, -1:].astype(int).reshape((-1,))

    # Features for training set
    X = shuffled_dataset[test_size:, : -1]

    # Labels for training set
    y = shuffled_dataset[test_size:, -1:].astype(int).reshape((-1,))

    return X, X_test, y, y_test
