import numpy as np

class Loss:
    def calculate(self, output, y_true, *, include_regularisation=False):
        sample_losses = self.forward(output, y_true)
        data_loss = np.mean(sample_losses)

        self.accumulated_sum += np.sum(sample_losses)
        self.accumulated_count += len(sample_losses)

        if not include_regularisation:
            return data_loss

        return data_loss, self.regularisation_loss()

    def calculate_accumulated(self, *, include_regularisation=False):
        data_loss = self.accumulated_sum / self.accumulated_count

        if not include_regularisation:
            return data_loss

        return data_loss, self.regularisation_loss()

    def reset_accumulated(self):
        self.accumulated_sum = 0
        self.accumulated_count = 0

    def regularisation_loss(self):
        regularisation_loss = 0

        for layer in self.trainable_layers:
            if layer.weight_regulariser_l1>0:
                regularisation_loss += layer.weight_regulariser_l1 * np.sum(np.abs(layer.weights))

            if layer.weight_regulariser_l2>0:
                regularisation_loss += layer.weight_regulariser_l2 * np.sum(layer.weights*layer.weights)

            if layer.bias_regulariser_l1>0:
                regularisation_loss += layer.bias_regulariser_l1 * np.sum(np.abs(layer.biases))

            if layer.bias_regulariser_l2>0:
                regularisation_loss += layer.bias_regulariser_l2 * np.sum(layer.biases*layer.biases)

        return regularisation_loss

    def remember_trainable_layers(self, trainable_layers):
        self.trainable_layers = trainable_layers

class Loss_CategoricalCrossEntropy(Loss):
    def forward(self, predictions, y_true):
        samples = len(predictions)
        predictions_clipped = np.clip(predictions, 1e-7, 1 - 1e-7)

        # Categorical Labels
        if len(y_true.shape) == 1:
            correct_confidences = predictions_clipped[range(samples), y_true]
        # One-hot Labels
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(predictions_clipped*y_true, axis=1)

        # Losses
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods

    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        labels = len(dvalues[0])

        # If sparse, convert into one-hot
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]
        
        self.dinputs = -y_true / dvalues
        self.dinputs = self.dinputs / samples

class Loss_BinaryCrossEntropy(Loss):
    def forward(self, y_pred, y_true):
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        sample_losses = -(y_true * np.log(y_pred_clipped) + (1 - y_true) * np.log(1 - y_pred_clipped))
        sample_losses = np.mean(sample_losses, axis=-1)
        return sample_losses
    
    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        outputs = len(dvalues[0])

        clipped_dvalues = np.clip(dvalues, 1e-7, 1 - 1e-7)

        self.dinputs = -(y_true / clipped_dvalues - (1 - y_true) / (1 - clipped_dvalues)) / outputs

        self.dinputs = self.dinputs / samples

class Loss_MeanSquaredError(Loss):
    def forward(self, y_pred, y_true):
        sample_losses = np.mean((y_true - y_pred)**2, axis=-1)
        return sample_losses

    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        outputs = len(dvalues[0])

        self.dinputs = -2 * (y_true - dvalues) / outputs
        self.dinputs / samples

class Loss_MeanAbsoluteError(Loss):
    def forward(self, y_pred, y_true):
        sample_losses = np.mean(np.abs(y_true - y_pred), axis=-1)
        return sample_losses
    
    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        outputs = len(dvalues[0])

        self.dinputs = np.sign(y_true - dvalues) / outputs
        self.dinputs = self.dinputs / samples