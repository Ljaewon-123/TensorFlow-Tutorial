# LSTM

text = open("rnn/pianoabc.txt").read()
print(text)

text = list(set(text))
text_size = len(text)
text.sort()

text_to_num = {}
num_to_text = {}
for i, c in enumerate(text):
    text_to_num[c] = i
    num_to_text[i] = c

numberic_text = []

for i in text:
    numberic_text.append(text_to_num[i])

x = []
y = []

for i in range(0, len(numberic_text) - 25):
    x.append(numberic_text[i:i+25])
    y.append(numberic_text[i+25])

print(x[0 : 5])
print(y[0: 5])

# 문자 숫자로 바꾸는법 
# 1. back of words(단어주머니) 만들기

import numpy as np
import tensorflow as tf

x = tf.one_hot(x, text_size)
y = tf.one_hot(y, text_size)


model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(128, input_shape=(25, text_size)),
    tf.keras.layers.Dense(text_size, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# batch_size: batch_size만큼 업데이트 후에 W값 업데이트
# vervose=2 로그 출력을 좀 덜해줌
model.fit(x, y, batch_size=64, epochs=50, vervose=2)
model.save("rnn/piano_model.h5")
