from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import typer

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = typer.Typer()

SEED = 42
LABEL = "malfunction"
FEATURES = [
    "feature1",
    "feature2",
    "feature3",
    "feature4",
    "feature5",
    "feature6",
    "feature7",
    "feature9",
]

SKEWED_FEATURES = [
    "feature2",
    "feature3",
    "feature4",
    "feature7",
    "feature9",
]


class Dataset:
    def __init__(self, path: Path):
        df = pd.read_csv(path, sep=";")
        self.X = df[FEATURES]
        self.y = df[LABEL]
        self._sqrt_transform()

    def __len__(self):
        return len(self.X)

    def _sqrt_transform(self):
        for feature in FEATURES:
            self.X[feature] = np.sqrt(self.X[feature])

    def _scale(self, X):
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        return X_scaled, scaler.mean_, scaler.var_

    def _make_one_class(self, X, y):
        """Only consider where LABEL is 0 in y.. Drop the corresponding rows in X."""
        mask = y == 0
        return X[mask], y[mask], X[~mask], y[~mask]

    def get_data(
        self,
        train_size: float,
        make_one_class: bool = False,
    ):
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, train_size=train_size, random_state=SEED
        )
        if make_one_class:
            X_train, y_train, dropped_X, dropped_y = self._make_one_class(
                X_train, y_train
            )
            X_test = pd.concat([X_test, dropped_X])
            y_test = pd.concat([y_test, dropped_y])

        X_train, mean, var = self._scale(X_train)
        X_test = (X_test - mean) / var

        return X_train, X_test, y_train, y_test

    def kfold(self, n_splits: int, make_one_class: bool = False):
        N = len(self.X)

        for i in range(n_splits):
            start = (N // n_splits) * i
            end = (N // n_splits) * (i + 1)
            mask = np.zeros(N, dtype=bool)
            mask[start:end] = True

            X_train = self.X[~mask]
            X_test = self.X[mask]
            y_train = self.y[~mask]
            y_test = self.y[mask]

            if make_one_class:
                X_train, y_train, dropped_X, dropped_y = self._make_one_class(
                    X_train, y_train
                )
                X_test = pd.concat([X_test, dropped_X])
                y_test = pd.concat([y_test, dropped_y])

            X_train, mean, var = self._scale(X_train)
            X_test = (X_test - mean) / var

            yield X_train, X_test, y_train, y_test


@app.command()
def run(
    data_file: Path = typer.Option(..., help="Path to the data file"),
    train_split: float = typer.Option(
        0.8, help="Percentage of data to use for training"
    ),
    make_one_class: bool = typer.Option(
        False, help="Only consider where LABEL is 0 in y"
    ),
):
    path = Path(data_file)
    data = Dataset(path)
    X_train, X_test, y_train, y_test = data.get_data(
        train_split,
        make_one_class=make_one_class,
    )

    logger.info(f"X_train shape: {X_train.shape}")
    logger.info(f"X_test shape: {X_test.shape}")
    logger.info(f"y_train shape: {y_train.shape}")
    logger.info(f"y_test shape: {y_test.shape}")

    # Print number of 0s and 1s in y_train and y_test
    logger.info(f"y_train: {y_train.value_counts()}")
    logger.info(f"y_test: {y_test.value_counts()}")


@app.command()
def run_kfold(
    splits: int = typer.Option(5, help="Number of splits"),
    data_file: Path = typer.Option(..., help="Path to the data file"),
    make_one_class: bool = typer.Option(
        False, help="Only consider where LABEL is 0 in y"
    ),
):
    path = Path(data_file)
    data = Dataset(path)

    for i, (X_train, X_test, y_train, y_test) in enumerate(
        data.kfold(splits, make_one_class=make_one_class)
    ):
        logger.info(f"Split {i + 1}")
        logger.info(f"X_train shape: {X_train.shape}")
        logger.info(f"X_test shape: {X_test.shape}")
        logger.info(f"y_train shape: {y_train.shape}")
        logger.info(f"y_test shape: {y_test.shape}")

        # Print number of 0s and 1s in y_train and y_test
        logger.info(f"y_train: {y_train.value_counts()}")
        logger.info(f"y_test: {y_test.value_counts()}")
        logger.info("")
