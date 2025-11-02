import pandas as pd

data = pd.read_csv("gpascore.csv")

# print(data.isnull().sum())
data = data.dropna()
# data.fillna(100) # 빈칸을 채워줌
# print(data['gpa'].min())  # gpa 칼럼의 최소값 출력
# print(data['gpa'].max())  # gpa 칼럼의 최대값 출력 

y_train = data['admit'].values  # 합격여부
x_train = []
# x_train = data.drop('admit', axis=1).values  # 합격여부 칼럼을 제외한 나머지 칼럼들

for i, rows in data.iterrows():
  x_train.append([ rows['gre'], rows['gpa'], rows['rank'] ])

# exit() # 결측치 없음  , 여기서 break

import numpy as np
import tensorflow as tf

model = tf.keras.models.Sequential([
  tf.keras.layers.Dense(64, activation='tanh', input_shape=(3,)),
  tf.keras.layers.Dense(128, activation='tanh'),
  tf.keras.layers.Dense(1, activation='sigmoid') # Output layer
])

# 정수예측은 힘듬 꼭 active(활성함수가 필요함, activation 파라미터 ) sigmoid = 0~1 사이로 예측

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(np.array(x_train), np.array(y_train), epochs=10, batch_size=32)

# 예측
predict = model.predict( np.array([[750, 3.70, 3], [400, 2.2, 1]]) )
print(predict)
print("합격확률 : ", predict[0][0] * 100)