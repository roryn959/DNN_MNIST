import numpy as np

class Accuracy:
    
    def calculate(self, y_pred, y_true):
        comparisons = self.compare(y_pred, y_true)
        accuracy = np.mean(comparisons)

        self.accumulated_sum += np.sum(comparisons)
        self.accumulated_count += len(comparisons)

        return accuracy

    def calculate_accumulated(self):
        accuracy = self.accumulated_sum / self.accumulated_count
        return accuracy
    
    def reset_accumulated(self):
        self.accumulated_sum = 0
        self.accumulated_count = 0

class Accuracy_Regression(Accuracy):
    def __init__(self):
        self.precision = None

    def init(self, y, reinit=False):
        if self.precision is None or reinit:
            self.precision = np.std(y) / 250
    
    def compare(self, y_true, y_pred):
        return np.abs(y_pred - y_true) < self.precision

class Accuracy_Categorical(Accuracy):
    def __init__(self, *, binary=False):
        self.binary = binary

    def init(self, y):
        pass

    def compare(self, y_pred, y_true):
        if not self.binary and len(y_true.shape)==2:
            y_true = np.argmax(y_true, axis=1)
        return y_pred == y_true