"""Digits splits and MLP math preserved from the historical training implementation."""

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

WORKERS = 4
VECTOR_SIZE = 2400
Q_SCALE = 1_000_000
LEARNING_RATE = 1.0


def get_data():

    X, y = load_digits(return_X_y=True)

    X = X.astype(np.float64) / 16.0

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=7,
        stratify=y,
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.20,
        random_state=8,
        stratify=y_train,
    )

    # Одинаковое число примеров
    # у каждого участника.
    n = (len(X_train) // WORKERS) * WORKERS

    X_train = X_train[:n]
    y_train = y_train[:n]

    workers = []

    for i in range(WORKERS):
        start = i * n // WORKERS
        end = (i + 1) * n // WORKERS

        workers.append(
            (
                X_train[start:end],
                y_train[start:end],
            )
        )

    return (
        workers,
        (X_val, y_val),
        (X_test, y_test),
    )


class MLP:
    def __init__(self, seed):

        rng = np.random.default_rng(seed)

        self.params = [
            rng.normal(
                0,
                np.sqrt(1 / 64),
                (64, 30),
            ),
            np.zeros(30),
            rng.normal(
                0,
                np.sqrt(1 / 30),
                (30, 10),
            ),
            np.zeros(10),
        ]

        self.nparam = sum(p.size for p in self.params)

        assert self.nparam <= VECTOR_SIZE

    def gradient(self, X, y):

        w1, b1, w2, b2 = self.params

        hidden = np.tanh(X @ w1 + b1)

        logits = hidden @ w2 + b2

        logits -= logits.max(
            axis=1,
            keepdims=True,
        )

        probabilities = np.exp(logits)

        probabilities /= probabilities.sum(
            axis=1,
            keepdims=True,
        )

        probabilities[np.arange(len(y)), y] -= 1

        probabilities /= len(y)

        grad_w2 = hidden.T @ probabilities

        grad_b2 = probabilities.sum(axis=0)

        grad_hidden = (probabilities @ w2.T) * (1 - hidden * hidden)

        grad_w1 = X.T @ grad_hidden

        grad_b1 = grad_hidden.sum(axis=0)

        return np.concatenate(
            [
                grad_w1.ravel(),
                grad_b1,
                grad_w2.ravel(),
                grad_b2,
            ]
        )

    def update(self, integer_sum):

        learning_rate = getattr(self, "learning_rate", LEARNING_RATE)
        if not np.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("Learning rate must be positive and finite")

        gradient = integer_sum[: self.nparam] / Q_SCALE

        position = 0

        for parameter in self.params:
            size = parameter.size

            part = gradient[position : position + size].reshape(parameter.shape)

            parameter -= learning_rate * part

            position += size

    def score(self, X, y):

        w1, b1, w2, b2 = self.params

        hidden = np.tanh(X @ w1 + b1)

        logits = hidden @ w2 + b2

        logits -= logits.max(
            axis=1,
            keepdims=True,
        )

        probabilities = np.exp(logits)

        probabilities /= probabilities.sum(
            axis=1,
            keepdims=True,
        )

        loss = -np.log(probabilities[np.arange(len(y)), y] + 1e-12).mean()

        accuracy = np.mean(probabilities.argmax(axis=1) == y)

        return float(loss), float(accuracy)


def make_workers(count):

    chunks, validation, test = get_data()

    X = np.concatenate([x for x, _ in chunks])

    y = np.concatenate([labels for _, labels in chunks])

    indices = np.array_split(
        np.arange(len(y)),
        count,
    )

    workers = [(X[ii], y[ii]) for ii in indices]

    return workers, validation, test
