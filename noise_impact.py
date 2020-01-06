from feedforward import *
import copy
import pickle
import datetime

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
    tr_p = 0.05*i      # 訓練用データのノイズ率： 0%, 5%, ... , 25%
    tr = add_noise(train_data, tr_p)

    l1 = LinearLayer(784, 100)
    f1 = ReLU()
    l2 = LinearLayer(100, 10)
    out = Softmax_CrossEntropy()
    layers = [l1, f1, l2, out]

    net_i = Network(layers, TRAIN_SIZE, TEST_SIZE, EPOCHS, LEARNING_RATE)
    net_i.train(tr,train_target)

    for j in range(6):
        te_p = 0.05*j      # 検証用データのノイズ率
        te = add_noise(test_data, te_p)

        net_ij = copy.deepcopy(net_i)
        net_ij.test(te,test_target)

        networks.append(net_ij)

    del net_i



filename = "noise_impact_{}.pickle".format(datetime.datetime.now())

with open(filename, "wb") as f:
    for i in range(len(networks)):
        pickle.dump(networks[i], f)
