import numpy as np

class NeuralNetwork:
    def __init__(self):
        np.random.seed(1)
        self.weights = 2 * np.random.random((3, 1)) - 1

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def train(self, inputs, outputs, iterations):
        for _ in range(iterations):
            predictions = self.predict(inputs)
            error = outputs - predictions
            adjustments = np.dot(inputs.T, error * self.sigmoid_derivative(predictions))
            self.weights += adjustments

    def predict(self, inputs):
        return self.sigmoid(np.dot(inputs, self.weights))


if __name__ == "__main__":
    nn = NeuralNetwork()
    print("Initial Weights:\n", nn.weights)

    # Training dataset
    X = np.array([[0, 0, 1],
                  [1, 1, 1],
                  [1, 0, 1],
                  [0, 1, 1]])
    y = np.array([[0], [1], [1], [0]])

    # Train the model
    nn.train(X, y, 15000)

    print("Trained Weights:\n", nn.weights)

    # Take user input for prediction
    inputs = [int(input(f"Input {i+1}: ")) for i in range(3)]
    prediction = nn.predict(np.array([inputs]))
    print("New Prediction:\n", prediction)
