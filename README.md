# project11_deeplearning_neuralnetwork

# Online Food Delivery Prediction (PyTorch)

A neural network that predicts whether a customer will order food online.

## What the Network Is Doing

This is a **binary classifier**. It takes customer data — age, income, occupation, location, feedback — and predicts one of two outcomes: will they order (1) or not (0).

The network learns by looking at 388 past customers. For each one, it makes a guess, compares it to the real answer, and adjusts its weights to get closer. After 500 rounds, it has learned the pattern well enough to predict new customers.

## packages Used

- **PyTorch** — built the neural network, handled backpropagation and weight updates automatically.
- **Pandas** — loaded the CSV, dropped nulls, encoded text columns into numbers.
- **Scikit-learn** — split data into train/dev/test, scaled features with StandardScaler.
- **Matplotlib** — plotted histograms to inspect data distribution.

## Step by Step

1. **Loaded the data** — 388 rows, 14 columns. Target was `Output` (Yes/No).
2. **Cleaned it** — dropped the trailing empty column, mapped Yes/No to 1/0, one-hot encoded Occupation, Marital Status, Monthly Income, Education, Customer Type.
3. **Split it** — 60% train, 20% dev, 20% test.
4. **Scaled it** — StandardScaler fit on train only, then applied to dev and test.
5. **Built the network** — 3 layers: input → 60 ReLU → 30 ReLU → 1 linear.
6. **Trained it** — 500 epochs of forward pass, loss, backward pass, and weight update using SGD.
7. **Evaluated it** — checked accuracy and loss on dev and test sets.
8. **Saved it** — exported weights to `neural_network.pth`.

## Architecture

- Input → 60 (ReLU) → 30 (ReLU) → 1 (linear logits)
- Loss: `BCEWithLogitsLoss` (sigmoid + cross-entropy combined)
- Optimizer: SGD (lr = 0.01)
- Seed: `torch.manual_seed(42)`

