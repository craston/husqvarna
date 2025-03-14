import logging
from pathlib import Path
from typing import Literal

import numpy as np
import typer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import OneClassSVM

from preprocess.dataset import Dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = typer.Typer()


class SVMTrainer:
    def __init__(
        self,
        X_train: np.ndarray,
        nu: float = 0.0001,
        kernel: Literal["linear", "poly", "rbf", "sigmoid", "precomputed"] = "rbf",
        gamma: float | Literal["scale", "auto"] = "scale",
    ):
        self.X_train = X_train
        self.model = OneClassSVM(nu=nu, kernel=kernel, gamma=gamma)

    def train(self):
        self.model.fit(self.X_train)

    def evaluate(self, X_test):
        y_pred = self.model.predict(X_test)
        return y_pred


@app.command()
def run(
    data_file: str = typer.Option(..., help="Path to the data file"),
    train_split: float = typer.Option(
        0.8, help="Percentage of data to use for training"
    ),
    nu: float = typer.Option(
        0.0001,
        help="An upper bound on the fraction of training errors and a lower bound of the fraction of support vectors",
    ),
    kernel: str = typer.Option(
        "rbf",
        help="Specifies the kernel type to be used in the algorithm [linear, poly, rbf, sigmoid, precomputed]",
    ),
    gamma: str = typer.Option(
        "scale",
        help="Kernel coefficient for 'rbf', 'poly' and 'sigmoid'. Options: scale, auto",
    ),
):
    path = Path(data_file)
    data = Dataset(path)
    X_train, X_test, y_train, y_test = data.get_data(train_split, make_one_class=True)

    logger.info(f"Training data shape: {y_train.value_counts()}")
    logger.info(f"Testing data shape: {y_test.value_counts()}")

    trainer = SVMTrainer(X_train, nu=nu, kernel=kernel, gamma=gamma)

    logger.info("Training SVM model ....")
    trainer.train()

    logger.info("Evaluating SVM model ....")
    y_pred = trainer.evaluate(X_test)

    # y_pred is +1 for inliers and -1 for outliers
    # Convert to 0 for inliers and 1 for outliers
    y_pred = (y_pred == -1).astype(int)

    metrics = classification_report(y_test, y_pred)
    logger.info(f"Classification report:\n{metrics}")

    cm = confusion_matrix(y_test, y_pred)
    logger.info(f"Confusion matrix:\n{cm}")


@app.command()
def run_kfold(
    data_file: str = typer.Option(..., help="Path to the data file"),
    n_splits: int = typer.Option(5, help="Number of splits"),
    nu: float = typer.Option(
        0.0001,
        help="An upper bound on the fraction of training errors and a lower bound of the fraction of support vectors",
    ),
    kernel: str = typer.Option(
        "rbf",
        help="Specifies the kernel type to be used in the algorithm [linear, poly, rbf, sigmoid, precomputed]",
    ),
    gamma: str = typer.Option(
        "scale",
        help="Kernel coefficient for 'rbf', 'poly' and 'sigmoid'. Options: scale, auto",
    ),
):
    path = Path(data_file)
    data = Dataset(path)

    for i, (X_train, X_test, y_train, y_test) in enumerate(
        data.kfold(n_splits, make_one_class=True)
    ):
        logger.info(f"Split {i + 1}")
        logger.info(f"Training data shape: {y_train.value_counts()}")
        logger.info(f"Testing data shape: {y_test.value_counts()}")

        trainer = SVMTrainer(X_train, nu=nu, kernel=kernel, gamma=gamma)

        logger.info("Training SVM model ....")
        trainer.train()

        logger.info("Evaluating SVM model ....")
        y_pred = trainer.evaluate(X_test)

        # y_pred is +1 for inliers and -1 for outliers
        # Convert to 0 for inliers and 1 for outliers
        y_pred = (y_pred == -1).astype(int)

        metrics = classification_report(y_test, y_pred)
        logger.info(f"Classification report:\n{metrics}")

        cm = confusion_matrix(y_test, y_pred)
        logger.info(f"Confusion matrix:\n{cm}")
        # write to file
        with open(f"classification_report_split{i + 1}.txt", "w") as f:
            f.write(f"Classification report:\n{metrics}")
            f.write("\n\n")
            f.write(f"Confusion matrix:\n{cm}")
