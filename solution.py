import csv
import time
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from IPython.display import display, clear_output


# =========================
# Exercise 1 – id_to_fruit
# =========================

def id_to_fruit(fruit_id: int, fruits: Iterable[str]) -> str:
    """
    Return the fruit name at index `fruit_id`.

    Original bug:
        The function iterated over a set. Sets are unordered, so the mapping
        from index to element is not stable.

    Fix:
        Convert the iterable to a list with deterministic order, then index.
    """
    fruits_list = list(fruits)  # or sorted(fruits) for alphabetical order
    try:
        return fruits_list[fruit_id]
    except IndexError:
        raise RuntimeError(f"Fruit with id {fruit_id} does not exist")


# quick self‑test for exercise 1
if __name__ == "__main__":
    fruits = ["apple", "orange", "melon", "kiwi", "strawberry"]
    name1 = id_to_fruit(1, fruits)
    name3 = id_to_fruit(3, fruits)
    name4 = id_to_fruit(4, fruits)
    print("EX1:", name1, name3, name4)  # expected: orange kiwi strawberry


# ======================
# Exercise 2 – swap()
# ======================

def swap(coords: np.ndarray) -> np.ndarray:
    """
    Flip x and y coordinates in `coords`.

    Input per row: [x1, y1, x2, y2, class_id]
    Output per row: [y1, x1, y2, x2, class_id]

    Original bugs:
        1) Both x1 and x2 were assigned from coords[:, 1].
        2) In‑place chained assignment on the same array was error‑prone.

    Fix:
        Work on a copy and explicitly swap (x1, y1) and (x2, y2).
    """
    original = coords
    swapped = original.copy()
    # swap (x1, y1)
    swapped[:, 0], swapped[:, 1] = original[:, 1], original[:, 0]
    # swap (x2, y2)
    swapped[:, 2], swapped[:, 3] = original[:, 3], original[:, 2]
    return swapped


if __name__ == "__main__":
    coords_example = np.array(
        [
            [10, 5, 15, 6, 0],
            [11, 3, 13, 6, 0],
            [5, 3, 13, 6, 1],
        ]
    )
    print("EX2 input:\n", coords_example)
    print("EX2 swapped:\n", swap(coords_example))


# =========================
# Exercise 3 – plot_data()
# =========================

def plot_data(csv_file_path: str) -> None:
    """
    Plot a precision–recall curve from a CSV file.

    The CSV must have a header: "precision,recall" and rows of floats.

    Original bug:
        Precision and recall were swapped on the axes: recall was on x‑axis
        and precision on y‑axis, which contradicts the description.

    Fix:
        Use precision as x and recall as y.
    """
    results = []
    with open(csv_file_path) as result_csv:
        csv_reader = csv.reader(result_csv, delimiter=",")
        next(csv_reader)  # skip header
        for row in csv_reader:
            results.append(row)

    results = np.stack(results).astype(float)
    precision = results[:, 0]
    recall = results[:, 1]

    plt.figure()
    plt.plot(precision, recall, marker="o")
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel("Precision")
    plt.ylabel("Recall")
    plt.title("Precision–Recall curve")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    # self‑test for exercise 3 (uncomment the last line to see the plot)
    f = open("data_file.csv", "w", newline="")
    w = csv.writer(f)
    w.writerow(["precision", "recall"])
    w.writerows(
        [
            [0.013, 0.951],
            [0.376, 0.851],
            [0.441, 0.839],
            [0.570, 0.758],
            [0.635, 0.674],
            [0.721, 0.604],
            [0.837, 0.531],
            [0.860, 0.453],
            [0.962, 0.348],
            [0.982, 0.273],
            [1.0, 0.0],
        ]
    )
    f.close()
    # plot_data("data_file.csv")


# ====================================
# Exercise 4 – GAN: Generator, Discriminator, train_gan
# ====================================

class Generator(nn.Module):
    """Generator network for the GAN."""

    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(100, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 784),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.model(x)
        output = output.view(x.size(0), 1, 28, 28)
        return output


class Discriminator(nn.Module):
    """Discriminator network for the GAN."""

    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(784, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), 784)
        output = self.model(x)
        return output


def train_gan(
    batch_size: int = 32,
    num_epochs: int = 5,
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu",
) -> None:
    """
    Train a simple GAN on MNIST (debugged version).

    Structural bug (original):
        Labels for real and fake samples used a fixed `batch_size` even when
        the last batch contained fewer samples. This caused size mismatches
        when `batch_size` did not divide the dataset size.

    Fix:
        Use the actual batch size `current_bs = real_samples.size(0)` whenever
        labels or latent vectors are created.

    Cosmetic bug (original):
        The logging/visualisation condition used `if n == batch_size - 1`,
        which depends on the number of batches and often never triggered.
        Here, generated images are shown once per epoch using `if n == 0`.
    """
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )

    try:
        train_set = torchvision.datasets.MNIST(
            root=".", train=True, download=True, transform=transform
        )
    except Exception:
        print("Failed to download MNIST, retrying with different URL")
        torchvision.datasets.MNIST.resources = [
            (
                "https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz",
                "f68b3c2dcbeaaa9fbdd348bbdeb94873",
            ),
            (
                "https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz",
                "d53e105ee54ea40749a09fcbcd1e9432",
            ),
            (
                "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz",
                "9fb629c4189551a2d022fa330f9573f3",
            ),
            (
                "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz",
                "ec29112dd5afa0611ce80d1b7f02629c",
            ),
        ]
        train_set = torchvision.datasets.MNIST(
            root=".", train=True, download=True, transform=transform
        )

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True
    )

    # show some real images once
    real_samples, _ = next(iter(train_loader))
    fig = plt.figure()
    for i in range(16):
        sub = fig.add_subplot(4, 4, 1 + i)
        sub.imshow(real_samples[i].reshape(28, 28), cmap="gray_r")
        sub.axis("off")
    fig.tight_layout()
    fig.suptitle("Real images")
    display(fig)
    time.sleep(2)

    discriminator = Discriminator().to(device)
    generator = Generator().to(device)
    lr = 0.0001
    loss_function = nn.BCELoss()
    optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=lr)
    optimizer_generator = torch.optim.Adam(generator.parameters(), lr=lr)

    for epoch in range(num_epochs):
        for n, (real_samples, _) in enumerate(train_loader):
            real_samples = real_samples.to(device=device)
            current_bs = real_samples.size(0)

            # Train discriminator
            real_labels = torch.ones((current_bs, 1), device=device)
            latent_space_samples = torch.randn((current_bs, 100), device=device)
            fake_samples = generator(latent_space_samples)
            fake_labels = torch.zeros((current_bs, 1), device=device)

            all_samples = torch.cat((real_samples, fake_samples))
            all_labels = torch.cat((real_labels, fake_labels))

            discriminator.zero_grad()
            output_discriminator = discriminator(all_samples)
            loss_discriminator = loss_function(output_discriminator, all_labels)
            loss_discriminator.backward()
            optimizer_discriminator.step()

            # Train generator
            latent_space_samples = torch.randn((current_bs, 100), device=device)
            generator.zero_grad()
            generated_samples = generator(latent_space_samples)
            output_discriminator_generated = discriminator(generated_samples)
            generator_labels = torch.ones((current_bs, 1), device=device)
            loss_generator = loss_function(
                output_discriminator_generated, generator_labels
            )
            loss_generator.backward()
            optimizer_generator.step()

            # show generated images once per epoch
            if n == 0:
                title = (
                    f"Generated images\nEpoch: {epoch} "
                    f"Loss D.: {loss_discriminator:.2f} "
                    f"Loss G.: {loss_generator:.2f}"
                )
                imgs = generated_samples.detach().cpu().numpy()
                fig = plt.figure()
                for i in range(16):
                    sub = fig.add_subplot(4, 4, 1 + i)
                    sub.imshow(imgs[i].reshape(28, 28), cmap="gray_r")
                    sub.axis("off")
                fig.suptitle(title)
                fig.tight_layout()
                clear_output(wait=False)
                display(fig)


if __name__ == "__main__":
    # Uncomment for a quick GAN test (may take time and GPU if available)
    # train_gan(batch_size=64, num_epochs=1)
    pass
