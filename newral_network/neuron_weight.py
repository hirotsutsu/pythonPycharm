# coding: UTF-8
import math

# シグモイド関数
def sigmoid(x):
    return 1.0/(1.0+math.exp(-x))

# ニューロン
class Neuron:
    input_sum = 0.0
    output = 0.0

    def setInput(self, inp):
        self.input_sum += inp

    def getOutput(self):
        self.output = sigmoid(self.input_sum)
        return self.output

# ニューラルネットワーク
class NewralNetwork:
    neuron = Neuron()

    weight = [0.5, 0.5, 0.5]
    def commit(self, input_data):
        self.neuron.setInput(input_data[0] * self.weight[0])
        self.neuron.setInput(input_data[1] * self.weight[1])
        self.neuron.setInput(input_data[2] * self.weight[2])
        return self.neuron.getOutput()

neural_network = NewralNetwork()

trial_data = [1.0, 2.0, 3.0]
print neural_network.commit(trial_data)