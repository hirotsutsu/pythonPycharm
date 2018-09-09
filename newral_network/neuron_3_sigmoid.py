# coding: UTF-8
import math

# シグモイド関数（入力の合計値を0~1に正規化する）
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

    def commit(self, input_data):
        for data in input_data:
            self.neuron.setInput(data)
        return self.neuron.getOutput()

neural_network = NewralNetwork()

trial_data = [1.0, 2.0, 3.0]
print neural_network.commit(trial_data)