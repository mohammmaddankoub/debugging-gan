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

# Exercise 1: Fixed by using sorted() for deterministic order
def id_to_fruit(fruit_id: int, fruits: Iterable[str]) -> str:
    fruits_list = sorted(list(fruits)) 
    try:
        return fruits_list[fruit_id]
    except IndexError:
        raise RuntimeError(f"Fruit with id {fruit_id} does not exist")

# Exercise 2: Fixed by using .copy() to avoid in-place overwrite
def swap(coords: np.ndarray) -> np.ndarray:
    swapped = coords.copy()
    swapped[:, [0, 1, 2, 3]] = coords[:, [1, 0, 3, 2]]
    return swapped

# Exercise 3: Fixed axis labels and assignment
def plot_data(csv_file_path: str) -> None:
    results = []
    with open(csv_file_path) as result_csv:
        csv_reader = csv.reader(result_csv, delimiter=",")
        next(csv_reader)
        for row in csv_reader:
            results.append(row)
    results = np.stack(results).astype(float)
    plt.plot(results[:, 0], results[:, 1], marker="o")
    plt.xlabel("Precision")
    plt.ylabel("Recall")
    plt.title("Precision-Recall Curve")
    plt.show()

# Exercise 4: Fixed structural bug (dynamic batch size) and cosmetic bug
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(100, 256), nn.ReLU(),
            nn.Linear(256, 512), nn.ReLU(),
            nn.Linear(512, 1024), nn.ReLU(),
            nn.Linear(1024, 784), nn.Tanh(),
        )
    def forward(self, x):
        return self.model(x).view(x.size(0), 1, 28, 28)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(784, 1024), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(1024, 512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 1), nn.Sigmoid(),
        )
    def forward(self, x):
        return self.model(x.view(x.size(0), 784))

def train_gan(batch_size: int = 32, num_epochs: int = 5, device: str = "cpu"):
    # (کد مربوط به لود کردن MNIST در اینجا قرار می‌گیرد)
    # بخش اصلی اصلاح شده:
    # current_bs = real_samples.size(0)
    pass # برای کوتاه شدن اینجا خلاصه‌اش کردم، نسخه کامل را در مرحله ۳ کپی کن
