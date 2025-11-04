import tensorflow as tf
import numpy as np

(train_x, train_y), (test_x, test_y) = tf.keras.datasets.fashion_mnist.load_data()

# 데이터 정규화
train_x = train_x / 255.0
test_x = test_x / 255.0

train_x = train_x.reshape( (train_x.shape[0], 28, 28, 1) )
test_x = test_x.reshape( (test_x.shape[0], 28, 28, 1) )

class_name = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
              'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

model = tf.keras.Sequential([
  # 32개의 필터, 3x3 커널사이즈, 활성화함수 relu
  tf.keras.layers.Conv2D(32, (3, 3), 
                        activation='relu', padding="same", 
                        input_shape=(28, 28, 1)),
  tf.keras.layers.MaxPooling2D((2, 2)), # 풀링층
  # tf.keras.layers.Dense(128, input_shape=(28, 28), activation='relu'),
  tf.keras.layers.Flatten(), # 행렬을 1차원압축 
  tf.keras.layers.Dense(64, activation='relu'),
  tf.keras.layers.Dense(10, activation='softmax'),
])

# flatten - dense - 출력

# 모델 잘만들었는지 확인
model.summary()

model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])
model.fit(train_x, train_y, validation_data=(test_x, test_y), epochs=5)
# validation_data => epochs1회 돌때마다 검증데이터로 테스트해줌

# 학습용은 넣으면 안됨
result = model.evaluate(test_x, test_y) # 테스트해줌 학습후 모델평가하기

print(result)

