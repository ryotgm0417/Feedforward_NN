from feedforward import *
import copy
import datetime
import pickle

TRAIN_SIZE = 10000        # 訓練データ数
TEST_SIZE = 2000         # テストデータ数
EPOCHS = 50             # エポック数
LEARNING_RATE = 0.01     # 学習率

train_df, train_data, train_target, test_df, test_data, test_target = read_mnist()

train_data = train_data[:TRAIN_SIZE]
test_data = test_data[:TEST_SIZE]
train_target = train_target[:TRAIN_SIZE]
test_target = test_target[:TEST_SIZE]

train_data = train_data / 255.0
test_data = test_data / 255.0

networks = []
for i in range(6):
    tr_p = 0.05*i      # 訓練用データのノイズ率： 0%, 12.5%, 25%
    tr = add_noise(train_data, tr_p)

    l1 = LinearLayer(784, 150)
    f1 = ReLU()
    l2 = LinearLayer(150, 100)
    f2 = ReLU()
    l3 = LinearLayer(100, 50)
    f3 = ReLU()
    l4 = LinearLayer(50, 10)
    out = Softmax_CrossEntropy()
    layers = [l1, f1, l2, f2, l3, f3, l4, out]

    net_i = Network(layers, TRAIN_SIZE, TEST_SIZE, EPOCHS, LEARNING_RATE)
    net_i.train(tr,train_target)

    for j in range(6):
        te_p = 0.05*j      # 検証用データのノイズ率
        te = add_noise(test_data, te_p)

        net_ij = copy.deepcopy(net_i)
        net_ij.test(te,test_target)

        networks.append(net_ij)

    del net_i


filename = "deep_learning_{}.pickle".format(datetime.datetime.now())

with open(filename, "wb") as f:
    for i in range(len(networks)):
        pickle.dump(networks[i], f)
