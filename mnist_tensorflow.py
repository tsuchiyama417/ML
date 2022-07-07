import tensorflow as tf

# KerasからMNISTを読み出す
mnist = tf.keras.datasets.mnist

# （訓練画像、訓練ラベル）、（テスト画像、テストラベル）
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# print(x_train.shape)
# (60000, 28, 28)
# print(x_train)
# [[[0 0 0 ... 0 0 0]
#   [0 0 0 ... 0 0 0]
#   [0 0 0 ... 0 0 0]
# print(y_train.shape)
# (60000,)
# print(y_train)
# [5 0 4 ... 5 6 8]

# print(x_test.shape)
# (10000, 28, 28)
# print(y_test.shape)
# (10000,)

# データの値を正規化(0 ~ 1)、結果を少数で格納
x_train, x_test = x_train / 255.0, x_test / 255.0

# tf.keras.Sequentialモデルを構築
# レイヤーを決定
model = tf.keras.models.Sequential([
  # 28×28の二次元データを784の一次元データに平滑化してinput
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  # Hidden Layerを定義
  # ノード数:128、活性化関数:ReLU
  tf.keras.layers.Dense(128, activation='relu'),
  # Dropout率を設定(0 ~ 1)
  tf.keras.layers.Dropout(0.2),
  # 出力層の設定、活性化関数:softmax
  tf.keras.layers.Dense(10, activation='softmax')

])


# 訓練プロセスの定義
# optimizer(最適化手法):Adam
# loss(損失関数):交差エントロピー誤差
# Metrics(評価関数)

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 学習
# 訓練データ、エポック数を指定
model.fit(x_train, y_train, epochs=5)
