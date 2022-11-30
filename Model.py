import pickle, copy, numpy as np

from Activations import Activation_Softmax, Activation_Softmax_Loss_CategoricalCrossEntropy
from Layers import *
from Losses import Loss_CategoricalCrossEntropy

class Model:
    def __init__(self):
        self.layers = []
        self.softmax_classifier_output = None

    def getParameters(self):
        parameters = []
        
        for layer in self.trainable_layers:
            parameters.append(layer.getParameters())
        
        return parameters

    def setParameters(self, parameters):
        for parameter_set, layer in zip(parameters, self.trainable_layers):
            layer.setParameters(*parameter_set)

    def saveParameters(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.getParameters(), f)

    def saveModel(self, path):
        model = copy.deepcopy(self)
        
        model.loss.reset_accumulated()
        model.accuracy.reset_accumulated()

        model.input_layer.__dict__.pop('output', None)
        model.loss.__dict__.pop('dinputs', None)

        for layer in model.layers:
            for property in ['inputs', 'output', 'dinouts', 'dweights', 'dbiases']:
                layer.__dict__.pop(property, None)

        with open(path, 'wb') as f:
            pickle.dump(model, f)
    
    def loadParameters(self, path):
        with open(path, 'rb') as f:
            self.setParameters(pickle.load(f))

    @staticmethod
    def loadModel(path):
        with open(path, 'rb') as f:
            model = pickle.load(f)
        return model

    def add(self, layer):
        self.layers.append(layer)
    
    def set(self, *, loss=None, optimiser=None, accuracy=None):
        if loss is not None:
            self.loss = loss
        if optimiser is not None:
            self.optimiser = optimiser
        if accuracy is not None:
            self.accuracy = accuracy

    def finalise(self):
        self.input_layer = Layer_Input()

        self.trainable_layers = []

        layer_count = len(self.layers)
        for i in range(layer_count):
            if i==0:
                self.layers[i].prev = self.input_layer
                self.layers[i].next = self.layers[i+1]
            elif i < layer_count - 1:
                self.layers[i].prev = self.layers[i-1]
                self.layers[i].next = self.layers[i+1]
            else:
                self.layers[i].prev = self.layers[i-1]
                self.layers[i].next = self.loss
                self.output_layer_activation = self.layers[i]
            
            if hasattr(self.layers[i], 'weights'):
                self.trainable_layers.append(self.layers[i])
        
        if self.loss is not None:
            self.loss.remember_trainable_layers(self.trainable_layers)

        if isinstance(self.layers[-1], Activation_Softmax) and isinstance(self.loss, Loss_CategoricalCrossEntropy):
            self.softmax_classifier_output = Activation_Softmax_Loss_CategoricalCrossEntropy()

    def forward(self, X, training=False):
        self.input_layer.forward(X, training)
        for layer in self.layers:
            layer.forward(layer.prev.output, training)
        return layer.output
    
    def backward(self, output, y):
        if self.softmax_classifier_output is not None:
            self.softmax_classifier_output.backward(output, y)
            self.layers[-1].dinputs = self.softmax_classifier_output.dinputs

            for layer in reversed(self.layers[:-1]):
                layer.backward(layer.next.dinputs)
            return

        self.loss.backward(output, y)

        for layer in reversed(self.layers):
            layer.backward(layer.next.dinputs)

    def train(self, X, y, *, epochs=1, display_interval=1, validation_data=None, batch_size=None):
        self.accuracy.init(y)

        training_steps = 1

        if validation_data is not None:
            validation_steps = 1
            X_val, y_val = validation_data
        
        if batch_size is not None:
            training_steps = len(X) // batch_size
            if training_steps * batch_size < len(X):
                training_steps += 1

            if validation_data is not None:
                validation_steps = len(X_val) // batch_size
                if validation_steps * batch_size < len(X_val):
                    validation_steps += 1
        
        for epoch in range(1, epochs+1):
            print(f'epoch: {epoch}')

            self.loss.reset_accumulated()
            self.accuracy.reset_accumulated()

            for step in range(training_steps):
                if batch_size is None:
                    batch_X = X
                    batch_y = y

                else:
                    batch_X = X[step*batch_size:(step+1)*batch_size]
                    batch_y = y[step*batch_size:(step+1)*batch_size]

                output = self.forward(batch_X, training=True)

                data_loss, regularisation_loss = self.loss.calculate(output, batch_y, include_regularisation=True)
                loss = data_loss + regularisation_loss

                predictions = self.output_layer_activation.predictions(output)
                accuracy = self.accuracy.calculate(predictions, batch_y)

                self.backward(output, batch_y)

                self.optimiser.pre_update_params()
                for layer in self.trainable_layers:
                    self.optimiser.update_params(layer)
                self.optimiser.post_update_params()

                if not step%display_interval or step == training_steps-1:
                    print(
                        f'step: {step}, ' +
                        f'acc: {accuracy:.3f}, ' +
                        f'loss: {loss:.3f}, ' +
                        f'data_loss: {data_loss:.3f}, ' +
                        f'reg_loss: {regularisation_loss:.3f}, ' +
                        f'lr: {self.optimiser.current_learning_rate}'
                    )
            
            epoch_data_loss, epoch_regularisation_loss = self.loss.calculate_accumulated(include_regularisation=True)
            epoch_loss = epoch_data_loss + epoch_regularisation_loss
            epoch_accuracy = self.accuracy.calculate_accumulated()

            print(
                f'training, ' +
                f'acc: {epoch_accuracy:.3f}, ' +
                f'loss: {epoch_loss:.3f}, ' +
                f'data_loss: {epoch_data_loss:.3f}, ' +
                f'reg_loss: {epoch_regularisation_loss:.3f}, ' +
                f'lr: {self.optimiser.current_learning_rate}'
            )
        
        if validation_data is not None:
            self.evaluate(*validation_data, batch_size=batch_size)

    def evaluate(self, X_val, y_val, *, batch_size=None):
        validation_steps = 1

        if batch_size is not None:
            validation_steps = len(X_val) // batch_size
            if validation_steps * batch_size < len(X_val):
                validation_steps += 1
        
        self.loss.reset_accumulated()
        self.accuracy.reset_accumulated()

        for step in range(validation_steps):
            if batch_size is None:
                batch_X = X_val
                batch_y = y_val
            else:
                batch_X = X_val[step*batch_size:(step+1)*batch_size]
                batch_y = y_val[step*batch_size:(step+1)*batch_size]
            
            output = self.forward(batch_X, training=False)
            self.loss.calculate(output, batch_y)
            predictions = self.output_layer_activation.predictions(output)
            self.accuracy.calculate(predictions, batch_y)
        
        validation_loss = self.loss.calculate_accumulated()
        validation_accuracy = self.accuracy.calculate_accumulated()

        print(f'validation, ' +
        f'acc: {validation_accuracy:.3f}, ' +
        f'loss: {validation_loss:.3f}'
        )

    def predict(self, X, *, batch_size=None):
        prediction_steps = 1

        if batch_size is not None:
            prediction_steps = len(X) // batch_size

            if prediction_steps * batch_size < len(X):
                prediction_steps += 1
        
        output = []

        for step in range(prediction_steps):
            if batch_size is None:
                batch_X = X
            else:
                batch_X = X[step*batch_size:(step+1)*batch_size]

            batch_output = self.forward(batch_X)

            output.append(batch_output)

        return np.vstack(output)