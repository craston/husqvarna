from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import typer

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = typer.Typer()

LABEL = "malfunction"
FEATURES = [
    "feature1",
    "feature2",
    "feature3",
    "feature4",
    "feature5",
    "feature6",
    "feature7",
    "feature8",
    "feature9",
]


class Dataset:
    def __init__(self, path: Path):
        df = pd.read_csv(path, sep=";")
        self.input_features = df[FEATURES]
        self.output = df[LABEL]

    def __len__(self):
        return len(self.input_features)

    def _scale(self, X):
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        return X_scaled, scaler.mean_, scaler.var_

    def get_data(self, train_size: float):
        X_train, X_test, y_train, y_test = train_test_split(
            self.input_features, self.output, train_size=train_size
        )
        X_train, mean, var = self._scale(X_train)
        X_test = (X_test - mean) / var

        return X_train, X_test, y_train, y_test


@app.command()
def test_dataset(
    data_file: Path = typer.Option(..., help="Path to the data file"),
    train_split: float = typer.Option(
        0.8, help="Percentage of data to use for training"
    ),
):
    path = Path(data_file)
    data = Dataset(path)
    X_train, X_test, y_train, y_test = data.get_data(train_split)

    logger.info(f"X_train shape: {X_train.shape}")
    logger.info(f"X_test shape: {X_test.shape}")
    logger.info(f"y_train shape: {y_train.shape}")
    logger.info(f"y_test shape: {y_test.shape}")
