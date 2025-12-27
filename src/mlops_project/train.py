import matplotlib.pyplot as plt
import torch
import typer
import wandb

from mlops_project.data import corrupt_mnist
from mlops_project.model import MyAwesomeModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def train(lr: float = 1e-3, batch_size: int = 32, epochs: int = 10) -> None:
    """Train a model on MNIST."""
    print("Training day and night")
    config = {"lr": lr, "batch_size": batch_size, "epochs": epochs}
    with wandb.init(project="corrupt_mnist", config=config) as run:

        print(f"{lr=}, {batch_size=}, {epochs=}")

        model = MyAwesomeModel().to(DEVICE)
        train_set, _ = corrupt_mnist()

        train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size)

        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        statistics = {"train_loss": [], "train_accuracy": []}
        for epoch in range(epochs):
            model.train()
            for i, (img, target) in enumerate(train_dataloader):
                img, target = img.to(DEVICE), target.to(DEVICE)
                optimizer.zero_grad()
                y_pred = model(img)
                loss = loss_fn(y_pred, target)
                loss.backward()
                optimizer.step()
                statistics["train_loss"].append(loss.item())

                accuracy = (y_pred.argmax(dim=1) == target).float().mean().item()
                statistics["train_accuracy"].append(accuracy)
                wandb.log({"train_loss": loss.item(), "train_accuracy": accuracy})
                if i % 100 == 0:
                    print(f"Epoch {epoch}, iter {i}, loss: {loss.item()}")
                

        torch.save(model.state_dict(), "models/model.pth")
        artifact = wandb.Artifact(
        name="corrupt_mnist_model",
        type="model",
        description="A model trained to classify corrupt MNIST images",
        metadata={"accuracy": accuracy, "final_loss": loss.item()},
        )
        artifact.add_file("models/model.pth")
        run.log_artifact(artifact)

    print("Training complete")
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    axs[0].plot(statistics["train_loss"])
    axs[0].set_title("Train loss")
    axs[1].plot(statistics["train_accuracy"])
    axs[1].set_title("Train accuracy")
    fig.savefig("reports/figures/training_statistics.png")

if __name__ == "__main__":
    typer.run(train)
