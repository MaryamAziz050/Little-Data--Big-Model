"""p1_train.py - Project 1 (Courier company): little data, big model.

"We have only 200 labelled pincode digits. Train a reader, and prove with curves
that it is learning, not memorising."

Run:  python p1_train.py        (about a minute on a laptop)

It trains three models for 300 epochs, records training and validation loss every
epoch, and writes everything to the results/ folder:
    results/curves.png      the three curve plots side by side
    results/results.csv     the results table
    results/best_*.pt       the best checkpoints (lowest validation loss)
"""
import csv
import os

import numpy as np
import torch
from torch import nn
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

import matplotlib
matplotlib.use("Agg")  # draw to a file, no window needed
import matplotlib.pyplot as plt

SEED = 0
EPOCHS = 300
OUT = "results"


# --------------------------------------------------------------------------
# Steps 1 and 2: load the digits, scale to 0-1, split 200 / 798 / 799
# --------------------------------------------------------------------------
def load_data(seed=SEED):
    """1,797 images of 8x8 pixels. Pixel values go from 0 to 16, so divide by 16."""
    X, y = load_digits(return_X_y=True)
    X = (X / 16.0).astype("float32")

    # 200 for training, then the rest in half: 798 validation, 799 test
    X_tr, X_rest, y_tr, y_rest = train_test_split(
        X, y, train_size=200, random_state=seed)
    X_va, X_te, y_va, y_te = train_test_split(
        X_rest, y_rest, test_size=799, random_state=seed)
    return {"X_tr": X_tr, "y_tr": y_tr, "X_va": X_va, "y_va": y_va,
            "X_te": X_te, "y_te": y_te}


def to_tensors(data):
    out = {}
    for key, arr in data.items():
        dtype = torch.long if key.startswith("y_") else torch.float32
        out[key] = torch.tensor(arr, dtype=dtype)
    return out


# --------------------------------------------------------------------------
# Step 4: the three models
# --------------------------------------------------------------------------
def make_small():
    """One linear layer: too simple, it will underfit."""
    return nn.Linear(64, 10)


def make_big():
    """Two hidden layers of 256: plenty of room to memorise 200 images."""
    return nn.Sequential(
        nn.Linear(64, 256), nn.ReLU(),
        nn.Linear(256, 256), nn.ReLU(),
        nn.Linear(256, 10))


def make_regularised():
    """The big model with dropout 0.3 after each hidden layer."""
    return nn.Sequential(
        nn.Linear(64, 256), nn.ReLU(), nn.Dropout(0.3),
        nn.Linear(256, 256), nn.ReLU(), nn.Dropout(0.3),
        nn.Linear(256, 10))


# name, label, model builder, optimizer builder, patience, top of the loss axis
MODELS = [
    ("small", "Small", make_small,
     lambda p: torch.optim.SGD(p, lr=0.01), None, 2.5),
    ("big", "Big", make_big,
     lambda p: torch.optim.Adam(p, lr=0.001), None, 1.0),
    ("regularised", "Big + dropout + decay", make_regularised,
     lambda p: torch.optim.AdamW(p, lr=0.001, weight_decay=0.05), 30, 1.0),
]


# --------------------------------------------------------------------------
# Steps 3 and 5: the training loop that records everything
# --------------------------------------------------------------------------
def train(name, model, opt, data, epochs=EPOCHS, patience=None):
    """Full-batch training. Records both losses after every epoch.

    Saves a checkpoint whenever validation loss improves. With `patience`, it
    stops after that many epochs without improvement (early stopping).
    """
    loss_fn = nn.CrossEntropyLoss()
    X_tr, y_tr, X_va, y_va = data["X_tr"], data["y_tr"], data["X_va"], data["y_va"]
    hist = {"train": [], "val": []}
    best, wait = float("inf"), 0

    for epoch in range(epochs):
        model.train()
        loss = loss_fn(model(X_tr), y_tr)
        opt.zero_grad()
        loss.backward()
        opt.step()

        model.eval()                      # switch dropout off to measure
        with torch.no_grad():
            hist["train"].append(loss_fn(model(X_tr), y_tr).item())
            hist["val"].append(loss_fn(model(X_va), y_va).item())

        if hist["val"][-1] < best:        # keep the best checkpoint
            best, wait = hist["val"][-1], 0
            torch.save(model.state_dict(), os.path.join(OUT, f"best_{name}.pt"))
        else:
            wait += 1
        if patience and wait >= patience:  # early stopping
            break

    hist["best_epoch"] = int(np.argmin(hist["val"])) + 1
    return hist


def accuracy(model, X, y):
    model.eval()
    with torch.no_grad():
        return (model(X).argmax(dim=1) == y).float().mean().item()


# --------------------------------------------------------------------------
# Step 6: plot the three runs side by side
# --------------------------------------------------------------------------
def plot_curves(runs, path):
    fig, axes = plt.subplots(1, len(runs), figsize=(5 * len(runs), 4))
    for ax, run in zip(axes, runs):
        h = run["hist"]
        epochs = range(1, len(h["train"]) + 1)
        ax.plot(epochs, h["train"], color="#E07B39", linewidth=2, label="training loss")
        ax.plot(epochs, h["val"], color="#2F6FEB", linewidth=2, linestyle="--",
                label="validation loss")
        ax.axvline(h["best_epoch"], color="black", linestyle=":", linewidth=1)
        ax.set_xlim(0, EPOCHS)
        ax.set_ylim(0, run["ymax"])       # values above the top are cut off
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        ax.set_title(run["label"])
        ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------
# Steps 7 and 8: the results table
# --------------------------------------------------------------------------
def results_rows(runs):
    rows = []
    for run in runs:
        h = run["hist"]
        rows.append({
            "model": run["label"],
            "epochs_run": len(h["train"]),
            "final_training_loss": round(float(h["train"][-1]), 3),
            "best_validation_loss": round(float(min(h["val"])), 3),
            "best_epoch": h["best_epoch"],
            "test_accuracy": round(float(run["test_acc"]), 3),
        })
    return rows


def main():
    os.makedirs(OUT, exist_ok=True)
    data = to_tensors(load_data())
    print(f"train {len(data['X_tr'])} · validation {len(data['X_va'])} · "
          f"test {len(data['X_te'])}")

    runs = []
    for name, label, build_model, build_opt, patience, ymax in MODELS:
        torch.manual_seed(SEED)
        model = build_model()
        opt = build_opt(model.parameters())
        print(f"training {label} ...")
        hist = train(name, model, opt, data, patience=patience)

        # The model with early stopping is measured after reloading its best
        # checkpoint; the others are measured after all 300 epochs.
        if patience:
            model.load_state_dict(torch.load(os.path.join(OUT, f"best_{name}.pt")))
        test_acc = accuracy(model, data["X_te"], data["y_te"])
        runs.append({"label": label, "hist": hist, "test_acc": test_acc, "ymax": ymax})

    plot_curves(runs, os.path.join(OUT, "curves.png"))

    rows = results_rows(runs)
    with open(os.path.join(OUT, "results.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print()
    print(f"{'Model':<24}{'Epochs':>8}{'Final train':>13}{'Best val (epoch)':>20}{'Test acc':>10}")
    for r in rows:
        best = f"{r['best_validation_loss']:.3f} ({r['best_epoch']})"
        print(f"{r['model']:<24}{r['epochs_run']:>8}{r['final_training_loss']:>13.3f}"
              f"{best:>20}{r['test_accuracy']:>10.3f}")
    print(f"\nSaved {OUT}/curves.png and {OUT}/results.csv")
   


if __name__ == "__main__":
    main()
