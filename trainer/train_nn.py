import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import typer
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, TensorDataset

from preprocess.dataset import Dataset

app = typer.Typer()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Define the neural network
class ThreeLayerNN(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super(ThreeLayerNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        return out


# Example usage
class Trainer:
    def __init__(
        self,
        train_loader: DataLoader,
        test_loader: DataLoader,
        input_size: int,
        hidden_size: int,
        output_size: int,
        weight: torch.Tensor = torch.tensor([1.0, 1.0]),
        learning_rate: float = 0.001,
    ):
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.model = ThreeLayerNN(input_size, hidden_size, output_size)
        self.criterion = nn.CrossEntropyLoss(weight=weight)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

    def train(self, num_epochs: int):
        for epoch in range(num_epochs):
            loss_avg = 0
            for inputs, labels in self.train_loader:
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss_avg += loss.item()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            print(
                f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss_avg / len(self.train_loader)}"
            )

        # Save the model
        torch.save(self.model.state_dict(), "model.pth")

    def evaluate(self):
        with torch.no_grad():
            y_gt = []
            y_pred = []
            for inputs, labels in self.test_loader:
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)

                y_gt.extend(labels.numpy())
                y_pred.extend(predicted.numpy())

        metrics = classification_report(y_gt, y_pred)
        logger.info(f"Classification report:\n{metrics}")

        cm = confusion_matrix(y_gt, y_pred)
        logger.info(f"Confusion matrix:\n{cm}")


@app.command()
def run(
    data_file: str = typer.Option(..., help="Path to the data file"),
    hidden_size: int = typer.Option(20, help="Size of the first hidden layer"),
    batch_size: int = typer.Option(128, help="Batch size for training"),
    train_split: float = typer.Option(
        0.8, help="Percentage of data to use for training"
    ),
    num_epochs: int = typer.Option(10, help="Number of epochs to train"),
    weight: list[float] = typer.Option([1.0, 1000.0], help="Weight for each class"),
):
    path = Path(data_file)
    data = Dataset(path)
    X_train, X_test, y_train, y_test = data.get_data(train_split, make_one_class=False)

    logger.info(f"Training data labels: {y_train.value_counts()}")
    logger.info(f"Testing data labels: {y_test.value_counts()}")

    X_test, y_train, y_test = X_test.to_numpy(), y_train.to_numpy(), y_test.to_numpy()

    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)

    train_loader = DataLoader(
        TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True
    )
    test_loader = DataLoader(
        TensorDataset(X_test, y_test), batch_size=batch_size, shuffle=False
    )

    trainer = Trainer(
        train_loader,
        test_loader,
        X_train.shape[1],
        hidden_size,
        2,
        weight=torch.tensor(weight),
    )

    logger.info("Training neural network ....")
    trainer.train(num_epochs=num_epochs)
    trainer.evaluate()
