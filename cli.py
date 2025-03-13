from typer import Typer

from preprocess import dataset

app = Typer()

app.add_typer(dataset.app, name="ds", help="Split data")

if __name__ == "__main__":
    app()
