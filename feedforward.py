import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

# データの読み込み（MNIST手書き文字）
def read_mnist():
    train_df = pd.read_csv('data/mnist-in-csv/mnist_train.csv', sep=',')    # パス
    test_df = pd.read_csv('data/mnist-in-csv/mnist_test.csv', sep=',')      # パス

    train_data = train_df.iloc[:,1:].to_numpy(dtype='float')
    train_target = train_df.iloc[:,0].to_numpy(dtype='int')
    train_target = OneHotEncoder(sparse=False).fit_transform(train_target.reshape(-1, 1))

    test_data = test_df.iloc[:,1:].to_numpy(dtype='float')
    test_target = test_df.iloc[:,0].to_numpy(dtype='int')
    test_target = OneHotEncoder(sparse=False).fit_transform(test_target.reshape(-1, 1))

    return train_df, train_data, train_target, test_df, test_data, test_target


##################################
# 結合層や活性化関数をクラスとして定義 #
##################################

class ReLU:   # Rectified Linear Unit
    def forward(self, u):
        self.u = u
        return np.maximum(0, u)

    def backward(self, dout):
        return dout * (self.u > 0).astype(float)   # uが正である場合、微分は1.0

    def update_weights(self, lr=0.1):
        pass


class LinearLayer:   # 線形の全結合層
    def __init__(self, I, O):
        self.I = I
        self.O = O
        self.W = np.random.randn(I,O) / np.sqrt(I)   # 重みの初期化。分散 1/I の正規分布
        self.b = np.zeros(O)
        self.grad_W = np.zeros((I,O))
        self.grad_b = np.zeros(O)

    def forward(self, x):
        self.x = x
        u = x @ self.W + self.b
        return u

    def backward(self, dout):   # dout shape: (O)
        din = self.W @ dout   # shape: (I)
        self.grad_W = self.x.reshape(self.I,1) @ dout.reshape(1, self.O)   # shape: (I, O)
        self.grad_b = dout    # shape: (O)
        return din

    def update_weights(self, lr=0.1):
        self.W = self.W - lr * self.grad_W
        self.b = self.b - lr * self.grad_b


class Softmax_CrossEntropy:   # 多クラス分類の出力層（softmax関数＋交差エントロピー誤差）
    def forward(self, u):
        self.y = np.exp(u)
        self.y = self.y / np.sum(self.y)
        return self.y

    def calculate_error(self, t):   # u（すなわちy）とtの次元数は一致する前提
        self.t = t
        error = -np.sum(t * np.log(self.y))
        return error

    def backward(self, dout=1.0):
        return dout*(self.y - self.t)

    def update_weights(self, lr=0.1):
        pass


#############################
# 作ったネットワークを訓練・検証 #
#############################

class Network:
    def __init__(self, layers, train_size, test_size, epochs=100, lr=0.1):
        self.layers = layers
        self.train_size = train_size
        self.test_size = test_size
        self.epochs = epochs
        self.lr = lr

    def train(self, train_data, train_target):      # 訓練
        for epoch in range(self.epochs):
            error = 0

            for i in range(self.train_size):
                x = train_data[i]
                t = train_target[i]
                dout = 1.0

                for layer in self.layers:
                    x = layer.forward(x)

                error += self.layers[-1].calculate_error(t)

                for layer in reversed(self.layers):
                    dout = layer.backward(dout)

                for layer in self.layers:
                    layer.update_weights(self.lr)

            if epoch % 1 == 0:
                print("Epoch no. {}, error is {}".format(epoch, error))


    def test(self, test_data, test_target):     # 検証
        correct_number = 0

        for i in range(self.test_size):
            x = test_data[i]
            t = test_target[i]

            for layer in self.layers:
                x = layer.forward(x)

            predict = np.argmax(x)
            correct_value = np.argmax(t)

            if predict == correct_value:
                correct_number += 1

        print("Accuracy: {}".format(correct_number*1.0/TEST_SIZE))
