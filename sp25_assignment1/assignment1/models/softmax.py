"""Softmax model."""

import numpy as np

np.random.seed(444)


class Softmax:
    def __init__(self, n_class: int, lr: float, epochs: int, reg_const: float, batch_size: int = 128):
        """Initialize a new classifier.

        Parameters:
            n_class: the number of classes
            lr: the learning rate
            epochs: the number of epochs to train for
            reg_const: the regularization constant
        """
        self.w = np.zeros(5)  
        self.lr = lr
        self.epochs = epochs
        self.reg_const = reg_const
        self.n_class = n_class
        self.batch_size = batch_size

    def softmax(self, x: np.ndarray) -> np.ndarray:
        """Compute the softmax of vector x.

        Parameters:
            x: a numpy array of shape (D,) containing the input vector

        Returns:
            the softmax of x
        """

        # Compute softmax of an input vector x
        max = np.max(x, axis=1, keepdims=True)
        return np.exp(x-max) / np.sum(np.exp(x - max), axis=1, keepdims=True)

    def calc_gradient(self, X_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
        """Calculate gradient of the softmax loss.

        Inputs have dimension D, there are C classes, and we operate on
        mini-batches of N examples.

        Parameters:
            X_train: a numpy array of shape (N, D) containing a mini-batch
                of data
            y_train: a numpy array of shape (N,) containing training labels;
                y[i] = c means that X[i] has label c, where 0 <= c < C

        Returns:
            gradient with respect to weights w; an array of same shape as w
        """
        
        minibatch_size, _ = X_train.shape

        # Caclulate linear outputs and softmax of linear outputs
        linear_output = np.dot(X_train, self.w)
        p_w_y_i_x_i = self.softmax(linear_output)


        # Subtract 1 from the correct class
        p_w_y_i_x_i[range(minibatch_size), y_train] -= 1
        p_w_y_i_x_i = p_w_y_i_x_i / minibatch_size

        # Calculate gradient
        gradient = np.dot(X_train.T, p_w_y_i_x_i)
        gradient += self.reg_const * self.w

        return gradient

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the classifier.

        Hint: operate on mini-batches of data for SGD.
        - Initialize self.w as a matrix with random values sampled uniformly from [-1, 1)
        and scaled by 0.01. This scaling prevents overly large initial weights,
        which can adversely affect training.

        Parameters:
            X_train: a numpy array of shape (N, D) containing training data;
                N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labels
        """
        samples, features = X_train.shape

        # Initialize weights with random values sampled uniformly from [-1, 1) and scaled by 0.01.
        self.w = np.random.uniform(
            low=-1, high=1, size=(features, self.n_class))*0.01

        indices = np.arange(samples)

        for epoch in range(self.epochs):

            # Compute accuracy during training
            y_hat = self.predict(X_train)
            print(
                f"Epoch {epoch} accuracy {np.sum(y_train == y_hat) / len(y_train) * 100} %")

            np.random.shuffle(indices)
            for batch in range(0, samples, self.batch_size):

                # Batch start and end indices
                start = batch
                end = min(samples, batch + self.batch_size)

                X_batch, y_batch = X_train[indices[start: end]
                                           ], y_train[indices[start: end]]
                
                # Calculate gradient and update weights
                batch_w = self.calc_gradient(X_batch, y_batch)
                self.w -= self.lr * batch_w

            # Decrease learning rate with a factor of 0.95
            self.lr *= 0.95

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
        linear_outputs = self.softmax(X_test @ self.w)

        return [np.argmax(i) for i in linear_outputs]

