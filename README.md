# Project 1 · Little data, big model (Courier company)

**Task:** We have only 200 labelled pincode digits. Train a reader and prove with curves that it is learning, not memorising.

## What this project does

It trains three models on 200 digit images and records the training loss and the validation loss after every epoch. The curves show which model underfits, which one overfits, and which one is healthy.

**Data:** the scikit-learn digits dataset, 1,797 images of 8x8 pixels (digits 0 to 9). The pixels are scaled to 0-1. Split: 200 training, 798 validation, 799 test.

**Models** (all trained for 300 epochs):
- **Small:** one linear layer
- **Big:** two hidden layers of 256
- **Big + dropout + decay:** the big model with dropout 0.3 and weight decay 0.05, stopped by early stopping (patience 30)

## Files

- `p1_train.py`: loads the data, trains the three models, plots the curves and makes the results table
- `REPORT.md`: results table and my decision
- `requirements.txt`: libraries

## Setup and run

```
pip install -r requirements.txt
python p1_train.py
```

It takes about a minute and creates a `results/` folder with `curves.png`, `results.csv` and the best checkpoint of each model.

## Results

| Model | Final training loss | Best validation loss (epoch) | Test accuracy |
|---|---|---|---|
| Small | 1.749 | 1.818 (300) | 0.660 |
| Big | 0.001 | 0.216 (97) | 0.957 |
| Big + dropout + decay | 0.009 | 0.202 (138) | 0.965 |

- **Small:** underfitting. Both losses stay high.
- **Big:** overfitting. The validation loss is lowest at epoch 97 and then goes up.
- **Big + dropout + decay:** healthy. It has the lowest validation loss and stopped at epoch 168.

## Decision

We ship the Big + dropout + decay model. The thing that would help most next is more labelled digits, because the regularised model is only 0.8 points better than the big one and 200 images is too little data.

## Limitations

- **Very little training data.** Only 200 images were used. The training loss (0.009) is still much lower than the validation loss (about 0.2), so the model is limited by the amount of data.
- **One run and one seed per model.** The big model (0.957) and the regularised model (0.965) differ by about 6 of 799 test images. With one seed, this difference could be luck.
- **Not perfect.** A test accuracy of 0.965 means about 28 of 799 test images are still read wrongly.
- **Only 8x8 digits from this dataset.** The model was not tested on photos of handwriting, other image sizes or another scanner, and it is likely to fail on them.
- **The validation set helped choose the model.** The best checkpoint and the early stopping were chosen with the validation loss, so the validation numbers are a little optimistic. The test images were used only once, for the final accuracy.