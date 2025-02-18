"""Support Vector Machine (SVM) model."""

import numpy as np

np.random.seed(444)


class SVM:
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

    def calc_gradient(self, X_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
        """Calculate gradient of the svm hinge loss.

        Inputs have dimension D, there are C classes, and we operate on
        mini-batches of N examples.

        Parameters:

            X_train: a numpy array of shape (N, D) containing a mini-batch
                of data
            y_train: a numpy array of shape (N,) containing training labels;
                y[i] = c means that X[i] has label c, where 0 <= c < C

        Returns:
            the gradient with respect to weights w; an array of the same shape
                as w
        """
        minibatch_size, _ = X_train.shape
        batch_grad = np.zeros_like(self.w)
        
        # For each minibatch datapoint
        for i in range(minibatch_size):

            # Calculate dot product of w_yi and x_i

            w_yit_xi = np.dot(self.w[:, y_train[i]], X_train[i])

            for c in range(self.n_class):
                if c != y_train[i]:

                    # Calculate dot product of w_c and x_i
                    w_ctx_xi = np.dot(self.w[:, c], X_train[i])
                    svm_margin = w_yit_xi - w_ctx_xi

                    # If the margin is less than 1, update gradients for points
                    if svm_margin < 1:
                        # Update gradients for w_yi and w_c
                        batch_grad[:, y_train[i]] = batch_grad[:,
                                                               y_train[i]] + self.lr * X_train[i]
                        batch_grad[:, c] = batch_grad[:, c] - \
                            self.lr * X_train[i]
                # Update gradients with regularization term
                batch_grad[:, c] = batch_grad[:, c] - \
                    (self.lr * self.reg_const / minibatch_size) * self.w[:, c]
                
        return batch_grad

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

        # Initialize weights
        self.w = np.random.uniform(
            low=-1, high=1, size=(features, self.n_class))*0.01
        

        indices = np.arange(samples)

        for epoch in range(self.epochs):
            
            # Print accuracies for each training epoch
            y_hat = self.predict(X_train)
            print(
                f"Epoch {epoch} accuracy {np.sum(y_train == y_hat) / len(y_train) * 100} %")

            # Shuffle indices for SGD
            np.random.shuffle(indices)
            for batch in range(0, samples, self.batch_size):

                # Batch star and end indices
                start = batch
                end = min(samples, batch + self.batch_size)

                X_batch, y_batch = X_train[indices[start: end]
                                           ], y_train[indices[start: end]]
                
                # Calculate gradient
                batch_w = self.calc_gradient(X_batch, y_batch)
                # Update weights
                self.w += self.lr * batch_w

            # Decrease learning rate
            self.lr *= 0.85

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

        linear_outputs = X_test @ self.w

        return [np.argmax(i) for i in linear_outputs]
