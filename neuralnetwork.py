import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

class layer:
    def __init__(self):
        self.input = None
        self.output = None
    
    def forwardPropagation(self):
        pass

    def backwardPropagation(self):
        pass

class dense(layer):
    def __init__(self, inputSize, outputSize):
        self.weights = np.random.randn(outputSize, inputSize)
        self.bias = np.random.randn(outputSize, 1)
    
    def forwardPropagation(self, input):
        self.input = input
        return np.dot(self.weights, input) + self.bias

    def backwardPropagation(self, outputGradient, learningRate):
        weightGradient = np.dot(outputGradient, self.input.T)
        self.weights -= learningRate * weightGradient
        self.bias -= learningRate * np.sum(outputGradient, axis=1, keepdims=True)
        return np.dot(self.weights.T, outputGradient)

class activation(layer):
    def __init__(self, activation, activationPrime):
        self.activation = activation
        self.activationPrime = activationPrime

    def forwardPropagation(self, input):
        self.input = input
        return self.activation(self.input)
    
    def backwardPropagation(self, outputGradient):
        return np.multiply(outputGradient, self.activationPrime(self.input))

class relu(activation):
    def __init__(self):
        relu = lambda x: np.maximum(0, x)
        reluPrime = lambda x: (x > 0).astype(float)
        super().__init__(relu, reluPrime)

class softmax(activation):
    def __init__(self):
        softmax = lambda x: np.exp(x - np.max(x)) / np.sum(np.exp(x - np.max(x)), axis=0, keepdims=True)
        softmaxPrime = lambda x: softmax(x) * (1 - softmax(x))
        super().__init__(softmax, softmaxPrime)

def crossEntropyLoss(trueY, predY):
    return -np.sum(trueY * np.log(predY + 1e-9)) / trueY.shape[1]

def crossEntropyLossPrime(trueY, predY):
    return -(trueY / (predY + 1e-9))

class neuralNetwork:
    def __init__(self, inputSize, hiddenSize, outputSize):
        self.firstLayer = dense(inputSize, hiddenSize)
        self.secondLayer = dense(hiddenSize, outputSize)
        self.inputActivation = relu()
        self.outputActivation = softmax()
    
    def forwardPropagation(self, input):
        self.hiddenOutput = self.firstLayer.forwardPropagation(input)
        self.activatedHidden = self.inputActivation.forwardPropagation(self.hiddenOutput)
        self.finalOutput = self.secondLayer.forwardPropagation(self.activatedHidden)
        self.activatedOutput = self.outputActivation.forwardPropagation(self.finalOutput)
        return self.activatedOutput

    def backwardPropagation(self, trueY, learningRate):
        self.lossGradient = crossEntropyLossPrime(trueY, self.activatedOutput)
        self.outputGradient = self.outputActivation.backwardPropagation(self.lossGradient)
        self.hiddenGradient = self.secondLayer.backwardPropagation(self.outputGradient, learningRate)
        self.activatedGradient = self.inputActivation.backwardPropagation(self.hiddenGradient)
        self.firstLayer.backwardPropagation(self.activatedGradient, learningRate)

    def train(self, X, Y, epochs, learningRate, batch_size=128):
        num_samples = X.shape[1]
        for epoch in range(epochs):
            total_loss = 0
            for start in range(0, num_samples, batch_size):
                end = min(start + batch_size, num_samples)
                X_batch = X[:, start:end]
                Y_batch = Y[:, start:end]

                self.activatedOutput = self.forwardPropagation(X_batch)
                total_loss += crossEntropyLoss(Y_batch, self.activatedOutput)
                self.backwardPropagation(Y_batch, learningRate)
            
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / num_samples}")

    def predict(self, X):
        output = self.forwardPropagation(X)
        return np.argmax(output, axis=0)

def load_mnist():
    mnist = fetch_openml('mnist_784', version=1)
    X = mnist.data / 255.0
    y = mnist.target.astype(int)
    y_one_hot = np.eye(10)[y]
    X = X.values.T
    return X, y_one_hot.T

X, y = load_mnist()

X_train, X_test, y_train, y_test = train_test_split(X.T, y.T, test_size=0.2, random_state=42)

nn = neuralNetwork(784, 128, 10)

nn.train(X_train.T, y_train.T, epochs=10, learningRate=0.01, batch_size=128)

predictions = nn.predict(X_test.T)

y_test_indices = np.argmax(y_test, axis=0)

accuracy = np.mean(predictions == y_test_indices)
print(f"Test accuracy: {accuracy * 100:.2f}%")
