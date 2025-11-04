import tensorflow as tf
import matplotlib.pyplot as plt

(train_x, train_y), (test_x, test_y) = tf.keras.datasets.fashion_mnist.load_data()

# plt.imshow(train_x[0], cmap='gray')

class_name = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
              'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# train_x = (60000, 28, 28) 이라서 input shape 가 28, 28인거 

model = tf.keras.Sequential([
  tf.keras.layers.Dense(128, input_shape=(28, 28), activation='relu'),
  tf.keras.layers.Dense(64, activation='relu'),
  tf.keras.layers.Flatten(), # 행렬을 1차원압축 
  tf.keras.layers.Dense(10, activation='softmax'),
])

# 모델 잘만들었는지 확인
model.summary()

model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])
model.fit(train_x, train_y, epochs=5)


# metrices -> 평가 지표

# 확률예측 문제라면 일단 마지막 레이터 노드수를 카테고리 갯수만큼
# cross entropy 라는 loss(손실) 함수 사용

# activevation='relu' -> 음수가 나오면 안됨 
# softmax 결과를 0~1 카테고리 예측문제에 사용
# sigmoid 결과를 0~1 사이로 바꿔줌 binary예측문제에 사용 마지막 노드는 1개 


# 모델 만들기 -> compile -> fit(학습) -> evaluate(평가) -> predict(예측)