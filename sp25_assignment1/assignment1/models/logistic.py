"""Logistic regression model."""

import numpy as np
np.random.seed(444)


class Logistic:
    def __init__(self, lr: float, epochs: int, threshold: float):
        """Initialize a new classifier.

        Parameters:
            lr: the learning rate
            epochs: the number of epochs to train for
        """
        self.w = np.random.uniform(
            low=-1, high=1, size=11)*0.01  # TODO: change this
        self.lr = lr
        self.epochs = epochs
        self.threshold = threshold

    def sigmoid(self, z: np.ndarray) -> np.ndarray:
        """Sigmoid function.

        Parameters:
            z: the input

        Returns:
            the sigmoid of the input
        """
        # Hint: To prevent numerical overflow, try computing the sigmoid for positive numbers and negative numbers separately.
        #       - For negative numbers, try an alternative formulation of the sigmoid function.

        # Calculates sigmoid function for positive and negative elements of the numpy array
        return np.where(z >= 0, 1/(1 + np.exp(-z)), np.exp(z) / (1+np.exp(z)))

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the classifier.

        - Use the logistic regression update rule as introduced in lecture.
        - Initialize self.w as a matrix with random values sampled uniformly from [-1, 1)
        and scaled by 0.01. 
        - This initialization prevents the weights from starting too large,
        which can cause saturation of the sigmoid function 

        Parameters:
            X_train: a numpy array of shape (N, D) containing training data;
                N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labels
        """

        samples, features = X_train.shape

        # Initialize weights with random values sampled uniformly from [-1, 1) and scaled by 0.01.

        self.w = np.random.uniform(low=-1, high=1, size=features)*0.01

        print(self.w)

        for epoch in range(self.epochs):

            # Shuffle indices for SGD
            indices = np.arange(samples)
            np.random.shuffle(indices)
            X_train = X_train[indices]
            y_train = y_train[indices]

            # Set batch size to 1 as it is not required for Logistic regression
            batch_size = 1
            for i in range(0, samples, batch_size):
                X_batch = X_train[i:i + batch_size]
                y_batch = y_train[i:i + batch_size]

                # Calculate linear output and the sigmoid of the linear output to be able to print training accuracies
                linear_output = np.dot(X_batch, self.w)
                sig = self.sigmoid(linear_output)

                # Calculate gradient
                grad = np.dot(X_batch.T, (sig - y_batch)) / len(X_batch)

                # Update weights
                self.w -= self.lr * grad

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Use the trained weights to predict labels for test data points.

        Parameters:
            X_test: a numpy array of shape (N, D) containing testing data;
                N examples with D dimensions

        Returns:exce
            predicted labels for the data in X_test; a 1-dimensional array of
                length N, where each element is an integer giving the predicted
                class.
        """

        # Inference of labels using computed weights
        inference = self.sigmoid(np.dot(X_test, self.w))

        return np.where(inference > self.threshold, 1, 0)
