import logging
from pathlib import Path

import typer
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix

from preprocess.dataset import Dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = typer.Typer()


class IFTrainer:
    def __init__(self, X_train):
        self.X_train = X_train
        self.model = IsolationForest(
            contamination="auto",
            random_state=42,
            n_estimators=100,
            n_jobs=-1,
            max_features=3,
        )

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
):
    path = Path(data_file)
    data = Dataset(path)
    X_train, X_test, y_train, y_test = data.get_data(train_split, make_one_class=True)

    logger.info(f"Training data shape: {y_train.value_counts()}")
    logger.info(f"Testing data shape: {y_test.value_counts()}")

    trainer = IFTrainer(X_train)

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
