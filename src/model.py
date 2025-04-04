import numpy as np


class SoftmaxRegression:
    def __init__(self):
        self.output = None
        self.W = None
        self.b = None

    def predict(self, X):
        # 1. Compute raw scores (logits) by performing a linear transformation:
        #   A. scores = X @ W + b
        #   B. This results in a shape of (num_examples, num_classes).
        scores = np.dot(X, self.W) + self.b

        # 2. Apply the softmax function to convert raw scores into probabilities
        exponent = np.exp(scores - np.max(scores, axis=1, keepdims=True))
        self.output = exponent / np.sum(exponent, axis=1, keepdims=True)
        return np.argmax(self.output, axis=1)

    def train(self, X, y, learning_rate=1e-3, epochs=1000):
        for epoch in range(epochs):
            self.forward(X, y)
            self.backward(X, y, learning_rate)

    def forward(self, X, y):
        num_examples, num_features = X.shape[0], X.shape[1]
        num_classes = len(np.unique(y))

        # 1. Initialize weights (W) with shape (num_features, num_classes) and biases (b) with shape (num_classes,).
        if self.W is None or self.b is None:
            self.b = np.zeros(shape=(1, num_classes))
            self.W = np.random.normal(loc=0.0, scale=1e-1, size=(num_features, num_classes))

        # 2. Predict
        self.predict(X)

        # 3. Compute the loss (e.g., cross-entropy) between predicted probabilities and true labels.
        log_loss = -np.log(self.output[np.arange(num_examples), y])
        mean_log_loss = np.mean(log_loss)
        return mean_log_loss, log_loss

    def backward(self, X, y, alpha):
        num_examples = X.shape[0]

        # The derivative is as follows dL/dW = (S_i - 1)X
        scores = self.output.copy()
        scores[np.arange(num_examples), y] -= 1
        scores = scores / num_examples

        dW = np.dot(X.T, scores)  # X.T: shape (num_classes, num_example) (num_examples, num_classes)
        db = np.sum(scores, axis=0, keepdims=True)

        self.W -= alpha * dW
        self.b -= alpha * db

    def evaluate(self, X, y):
        return self.forward(X, y)


if __name__ == '__main__':
    test = np.array([[0.2, 0.6, 0.2], [0.8, 0.1, 0.1], [0.9, 0.05, 0.05]])
    true = np.array([1, 2, 1]).T
    print(true)
    print(test[np.arange(len(true)), true])
