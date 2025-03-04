import numpy as np

class layer:
    def __init__(self):
        self.input = None
        self.output = None

class dense(layer):
    def __init__(self, inputSize, outputSize):
        self.weights = np.random.randn(outputSize, inputSize)
        self.bias = np.random.randn(outputSize, 1)
    
    def forwardPropagation(self, input):
        self.input = input
        return np.dot(self.weights, input) + self.bias

    def backwardPropagation(self, outputGradient, learningRate):
        weightGradient = np.dot(outputGradient, self.input.T)
        self.weights = self.weights - (learningRate * weightGradient)
        self.bias = self.bias - (learningRate * outputGradient)
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

class tanh(activation):
    def __init__(self):
        tanh = lambda x: np.tanh(x)
        tanhPrime = lambda x: 1 - ((np.tanh(x))**2)
        super().__init__(tanh, tanhPrime)

class relu(activation):
    def __init__(self):
        relu = lambda x: np.maximum(0, x)
        reluPrime = lambda x: (x > 0).astype(float)
        super().__init__(relu, reluPrime)

def meanSquaredError(trueY, predY):
    return np.mean((trueY - predY)**2)

def meanSquaredErrorPrime(trueY, predY):
    return (2 * (predY - trueY)) / np.size(trueY)

class neuralNetwork:
    def __init__(self, inputSize, hiddenSize, outputSize):
        self.firstLayer = dense(inputSize, hiddenSize)
        self.secondLayer = dense(hiddenSize, outputSize)
        self.activation = relu()
    
    def forwardPropagation(self, input):
        self.hiddenOutput = self.firstLayer.forwardPropagation(input)
        self.activatedOutput = self.activation.forwardPropagation(self.hiddenOutput)
        self.finalOutput = self.secondLayer.forwardPropagation(self.activatedOutput)
        return self.finalOutput

    def backwardPropagation(self, trueY, learningRate):
        lossGradient = meanSquaredErrorPrime(trueY, self.finalOutput)
        activationGradient = self.secondLayer.backwardPropagation(lossGradient, learningRate)
        hiddenGradient = self.activation.backwardPropagation(activationGradient)
        self.firstLayer.backwardPropagation(hiddenGradient, learningRate)

