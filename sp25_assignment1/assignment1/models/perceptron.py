"""Perceptron model."""

import numpy as np
np.random.seed(444)


class Perceptron:
    def __init__(self, n_class: int, lr: float, epochs: int, batch_size: int = 128):
        """Initialize a new classifier.

        Parameters:
            n_class: the number of classes
            lr: the learning rate
            epochs: the number of epochs to train for
        """
        self.w = np.zeros(n_class) 
        self.lr = lr
        self.epochs = epochs
        self.n_class = n_class
        self.reg_const = 10
        self.batch_size = batch_size

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the classifier.

        - Use the perceptron update rule as introduced in the Lecture.
        - Initialize self.w as a matrix with random values sampled uniformly from [-1, 1)
        and scaled by 0.01. This scaling prevents overly large initial weights,
        which can adversely affect training.

        Parameters:
            X_train: a number array of shape (N, D) containing training data;
                N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labels
        """

        samples, features = X_train.shape

        # Initialize weights with random values sampled uniformly from [-1, 1) and scaled by 0.01.
        self.w = np.random.uniform(
            low=-1, high=1, size=(self.n_class, features)) * 0.01

        for epoch in range(self.epochs):
            indices = np.arange(samples)
            np.random.shuffle(indices)

            # Compute accuracy during training
            y_hat = self.predict(X_train)
            print(
                f"Epoch {epoch} accuracy {np.sum(y_train == y_hat) / len(y_train) * 100} %")

            for start in range(0, samples, self.batch_size):

                # Batch start and end indices
                end = min(start + self.batch_size, samples)
                batch_indices = indices[start:end]

                X_batch = X_train[batch_indices]
                y_batch = y_train[batch_indices]

                # Initialize gradient vector with zeroes
                gradient = np.zeros_like(self.w)
                
                # Iterate through batch
                for i in range(len(batch_indices)):
                    xi = X_batch[i]
                    yi = y_batch[i]

                    # Iterate through classes
                    for c in range(self.n_class):
                        if c != yi:
                            # If class is not the correct class, update weights, if w_c * xi > w_yi * xi
                            w_ctx_xi = np.dot(self.w[c], xi)
                            w_yit_xi = np.dot(self.w[yi], xi)
                            
                            if w_ctx_xi > w_yit_xi:
                                gradient[yi] += self.lr * xi
                                gradient[c] -= self.lr * xi

                # Update weights
                self.w += self.lr * gradient / self.batch_size
        return

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Use the trained weights to predict labels for test data points.

        Parameters:
            X_test: a numpy array of shape (N, D) containing testing data;
                N examples with D dimensions

        Returns:
            predicted labels for the data in X_test; a 1-dimensional array of
                length N, where each element is an integer giving the predicted
                class.
        """
        # Inference of labels using computed weights
        y_hat = X_test @ self.w.T

        return [np.argmax(i) for i in y_hat]
