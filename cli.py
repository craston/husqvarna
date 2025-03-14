from typer import Typer

from preprocess import dataset
from trainer import train_svm
from trainer import train_nn
from trainer import train_if
from utils import seed_everything

app = Typer()

app.add_typer(dataset.app, name="ds", help="Split data")
app.add_typer(train_svm.app, name="svm", help="Train SVM model")
app.add_typer(train_nn.app, name="nn", help="Train neural network model")
app.add_typer(train_if.app, name="if", help="Train Isolation Forest model")

if __name__ == "__main__":
    seed_everything(42)
    app()
