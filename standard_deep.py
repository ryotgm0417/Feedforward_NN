from feedforward import *
import matplotlib.pyplot as plt

TRAIN_SIZE = 2000        # 訓練データ数
TEST_SIZE = 2000         # テストデータ数
EPOCHS = 10             # エポック数
LEARNING_RATE = 0.01     # 学習率

train_df, train_data, train_target, test_df, test_data, test_target = read_mnist()

train_data = train_data[:TRAIN_SIZE]
test_data = test_data[:TEST_SIZE]
train_target = train_target[:TRAIN_SIZE]
test_target = test_target[:TEST_SIZE]

# データの正規化

# def normalize(data):
#     mean = np.mean(data, axis=1).reshape(-1,1)
#     var = np.var(data, axis=1).reshape(-1,1)
#     return (data - mean) / (np.sqrt(var) + 1e-6)

# train_data = normalize(train_data)
# test_data = normalize(test_data)

train_data = train_data / 255.0
test_data = test_data / 255.0

l1 = LinearLayer(784, 100)
f1 = ReLU()
l2 = LinearLayer(100, 50)
f2 = ReLU()
l3 = LinearLayer(50, 10)
out = Softmax_CrossEntropy()

layers = [l1, f1, l2, f2, l3, out]

network = Network(layers, TRAIN_SIZE, TEST_SIZE, EPOCHS, LEARNING_RATE)
network.train(train_data, train_target)
network.test(test_data, test_target)

plt.plot(network.train_curve)
plt.show()
