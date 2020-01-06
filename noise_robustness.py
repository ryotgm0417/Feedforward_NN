from feedforward import *
import copy
import pickle
import datetime

TRAIN_SIZE = 10000        # 訓練データ数
TEST_SIZE = 2000         # テストデータ数
EPOCHS = 100             # エポック数
LEARNING_RATE = 0.01     # 学習率

train_df, train_data, train_target, test_df, test_data, test_target = read_mnist()

train_data = train_data[:TRAIN_SIZE]
test_data = test_data[:TEST_SIZE]
train_target = train_target[:TRAIN_SIZE]
test_target = test_target[:TEST_SIZE]

train_data = train_data / 255.0
test_data = test_data / 255.0

networks = []

for layer_size in [100,300,500,700,900,1100]:
    l1 = LinearLayer(784, layer_size)
    f1 = ReLU()
    l2 = LinearLayer(layer_size, 10)
    out = Softmax_CrossEntropy()
    layers = [l1, f1, l2, out]

    net = Network(layers, TRAIN_SIZE, TEST_SIZE, EPOCHS, LEARNING_RATE)
    net.train(train_data,train_target)

    for j in range(6):
        p = 0.05*j
        te = add_noise(test_data, p)
        net_j = copy.deepcopy(net)
        net_j.test(te,test_target)

        networks.append(net_j)

    del net


filename = "noise_robustness_{}.pickle".format(datetime.datetime.now())

with open(filename, "wb") as f:
    for i in range(len(networks)):
        pickle.dump(networks[i], f)
