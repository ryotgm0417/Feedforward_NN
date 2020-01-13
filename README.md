# Feedforward_NN

実行方法：<br>
環境はpython 3.6.8<br>
必要なライブラリは numpy, pandas, matplotlib, copy, time, datetime, pickle

feedforward.py内の以下の行（第8,9行）のパスを書き換える：

>      train_df = pd.read_csv('data/mnist-in-csv/mnist_train.csv', sep=',')    # パス
>      test_df = pd.read_csv('data/mnist-in-csv/mnist_test.csv', sep=',')      # パス

その後、ターミナルでpython3 <ファイル名> として noise_impact.py, noise_robustness.py, deep_learning.py のいずれかを実行

ファイル説明：<br>
feedforward.py ------ ニューラルネットの実装の主要部（クラス・関数の定義）<br>
noise_impact.py, noise_robustness.py, deep_learning.py ------ 実行ファイル<Br>

使っているデータセット：
出典：https://www.kaggle.com/oddrationale/mnist-in-csv

mnist_train.csv ------ 訓練用データ。CSVファイル
mnist_test.csv ------ 検証用データ。CSVファイル
