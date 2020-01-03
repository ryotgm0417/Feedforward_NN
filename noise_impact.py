from feedforward import *
import pickle
import datetime

TRAIN_SIZE = 10000        # 訓練データ数
TEST_SIZE = 2000         # テストデータ数
# EPOCHS = 100             # エポック数
LEARNING_RATE = 0.01     # 学習率

train_df, train_data, train_target, test_df, test_data, test_target = read_mnist()

train_data = train_data[:TRAIN_SIZE]
test_data = test_data[:TEST_SIZE]
train_target = train_target[:TRAIN_SIZE]
test_target = test_target[:TEST_SIZE]

train_data = train_data / 255.0
test_data = test_data / 255.0


#### Network A: 3層（パラメータ数少なめ） ####
network_A = [-1,-1,-1,-1,-1,-1]
for i in range(6):
    p = 0.05*i      # ノイズ率： 0%, 5%, ... , 25%

    l1 = LinearLayer(784, 50)
    f1 = ReLU()
    l2 = LinearLayer(50, 10)
    out = Softmax_CrossEntropy()
    layers = [l1, f1, l2, out]

    tr = add_noise(train_data, p)
    te = add_noise(test_data, p)

    network_A[i] = Network(layers, TRAIN_SIZE, TEST_SIZE, 50, LEARNING_RATE)
    network_A[i].train(tr,train_target)
    network_A[i].test(te,test_target)

#    del network_A[i].layers     # メモリ解放のために重みなどを破棄


#### Network B: 3層（パラメータ数多め） ####
network_B = [-1,-1,-1,-1,-1,-1]
for i in range(6):
    p = 0.05*i      # ノイズ率： 0%, 5%, ... , 25%

    l1 = LinearLayer(784, 400)
    f1 = ReLU()
    l2 = LinearLayer(400, 10)
    out = Softmax_CrossEntropy()
    layers = [l1, f1, l2, out]

    tr = add_noise(train_data, p)
    te = add_noise(test_data, p)

    network_B[i] = Network(layers, TRAIN_SIZE, TEST_SIZE, 100, LEARNING_RATE)
    network_B[i].train(tr,train_target)
    network_B[i].test(te,test_target)

#    del network_B[i].layers     # メモリ解放のために重みなどを破棄


#### Network C: 4層 ####
network_C = [-1,-1,-1,-1,-1,-1]
for i in range(6):
    p = 0.05*i      # ノイズ率： 0%, 5%, ... , 25%

    l1 = LinearLayer(784, 100)
    f1 = ReLU()
    l2 = LinearLayer(100, 50)
    f2 = ReLU()
    l3 = LinearLayer(50, 10)
    out = Softmax_CrossEntropy()
    layers = [l1, f1, l2, f2, l3, out]

    tr = add_noise(train_data, p)
    te = add_noise(test_data, p)

    network_C[i] = Network(layers, TRAIN_SIZE, TEST_SIZE, 150, LEARNING_RATE)
    network_C[i].train(tr,train_target)
    network_C[i].test(te,test_target)

#    del network_C[i].layers     # メモリ解放のために重みなどを破棄

filename = "noise_impact_{}.pickle".format(datetime.datetime.now())

with open(filename, "wb") as f:
    for i in range(6):
        pickle.dump(network_A[i], f)
        pickle.dump(network_B[i], f)
        pickle.dump(network_C[i], f)
