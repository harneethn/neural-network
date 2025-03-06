import numpy as np

# layers
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

# activation functions
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

class softmax(activation):
    def __init__(self):
        softmax = lambda x: np.exp(x - np.max(x)) / np.sum(np.exp(x - np.max(x)), axis=0, keepdims=True)
        softmaxPrime = lambda x: softmax(x) * (1 - softmax(x))
        super().__init__(softmax, softmaxPrime)

# loss functions
def meanSquaredError(trueY, predY):
    return np.mean((trueY - predY)**2)

def meanSquaredErrorPrime(trueY, predY):
    return (2 * (predY - trueY)) / np.size(trueY)

def crossEntropyLoss(trueY, predY):
    return -np.sum(trueY * np.log(predY + 1e-9)) / trueY.shape[1]

def crossEntropyLossPrime(trueY, predY):
    return -(trueY / (predY + 1e-9))

# neural networks
class neuralNetwork:
    def __init__(self):
        self.firstLayer = None
        self.secondLayer = None
        self.inputActivation = None
        self.outputActivation = None
    
    def forwardPropagation(self):
        pass

    def backwardPropagation(self):
        pass

class softmaxNetwork(neuralNetwork):
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

class reluNetwork(neuralNetwork):
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
        self.lossGradient = meanSquaredErrorPrime(trueY, self.finalOutput)
        self.activationGradient = self.secondLayer.backwardPropagation(self.lossGradient, learningRate)
        self.hiddenGradient = self.activation.backwardPropagation(self.activationGradient)
        self.firstLayer.backwardPropagation(self.hiddenGradient, learningRate)

class tanhNetwork(neuralNetwork):
    def __init__(self, inputSize, hiddenSize, outputSize):
        self.firstLayer = dense(inputSize, hiddenSize)
        self.secondLayer = dense(hiddenSize, outputSize)
        self.activation = tanh()
    
    def forwardPropagation(self, input):
        self.hiddenOutput = self.firstLayer.forwardPropagation(input)
        self.activatedOutput = self.activation.forwardPropagation(self.hiddenOutput)
        self.finalOutput = self.secondLayer.forwardPropagation(self.activatedOutput)
        return self.finalOutput

    def backwardPropagation(self, trueY, learningRate):
        self.lossGradient = meanSquaredErrorPrime(trueY, self.finalOutput)
        self.activationGradient = self.secondLayer.backwardPropagation(self.lossGradient, learningRate)
        self.hiddenGradient = self.activation.backwardPropagation(self.activationGradient)
        self.firstLayer.backwardPropagation(self.hiddenGradient, learningRate)
