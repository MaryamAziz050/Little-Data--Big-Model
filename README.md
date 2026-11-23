# Project 1 · Little data, big model (Courier company)

**Task:** We have only 200 labelled pincode digits. Train a reader and prove with curves that it is learning, not memorising.

## Files

- `p1_train.py`: trains three models, records training and validation loss every epoch, plots the curves and makes the results table
- `REPORT.md`: results table and my decision
- `requirements.txt`: libraries

## How to run

```
pip install -r requirements.txt
python p1_train.py
```

It creates a `results/` folder with `curves.png` and `results.csv`.

## What it does

1. Loads the digits data (1,797 images, 8x8 pixels) and scales the pixels to 0-1.
2. Splits it: 200 training, 798 validation, 799 test.
3. Trains three models for 300 epochs and records training loss and validation loss after every epoch:
   - **Small:** one linear layer
   - **Big:** two hidden layers of 256
   - **Big + dropout + decay:** the big model with dropout 0.3 and weight decay 0.05, stopped by early stopping 
4. Plots the three runs side by side in `curves.png`.

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