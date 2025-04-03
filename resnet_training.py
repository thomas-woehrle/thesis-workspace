import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader, Subset


def get_cifar100_dataloader(is_train: bool, batch_size: int, use_every_nth: int | None = None):
    dataset = torchvision.datasets.CIFAR100(
        "./artifacts", train=is_train, download=True, transform=torchvision.transforms.ToTensor())
    if use_every_nth is not None:
        dataset = Subset(dataset, indices=range(
            0, len(dataset), use_every_nth))

    dataloader = DataLoader(dataset, batch_size, shuffle=is_train)
    return dataloader


class ResNetTraining:
    def __init__(self, dry_run: bool = False):
        torch.manual_seed(42)
        self.model = torchvision.models.resnet18()
        self.num_epochs = 10
        self.curr_ep = 0
        self.train_dataloader = get_cifar100_dataloader(
            is_train=True, batch_size=16, use_every_nth=1000 if dry_run else None)
        self.val_dataloader = get_cifar100_dataloader(
            is_train=False, batch_size=16, use_every_nth=1000 if dry_run else None)
        self.optimizer = optim.Adam(self.model.parameters())
        self.criterion = nn.CrossEntropyLoss()

    def train(self):
        """Trains for one epoch"""
        self.model.train()
        for x, y in self.train_dataloader:
            self.optimizer.zero_grad()

            y_hat = self.model(x)
            loss = self.criterion(y_hat, y)

            loss.backward()
            self.optimizer.step()

    @torch.no_grad()
    def validate(self):
        """Validate the current model"""
        self.model.eval()

        losses = torch.zeros(len(self.val_dataloader))

        for idx, (x, y) in enumerate(self.val_dataloader):
            y_hat = self.model(x)
            loss = self.criterion(y_hat, y)
            losses[idx] = loss

    def run_training(self):
        for ep in range(self.num_epochs):
            self.curr_ep = ep
            self.train()
            self.validate()


if __name__ == "__main__":
    training = ResNetTraining(dry_run=True)

    training.run_training()
