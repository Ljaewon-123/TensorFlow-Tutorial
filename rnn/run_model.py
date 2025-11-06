import tensorflow as tf
import numpy as np

Pmodel = tf.keras.models.load_model("rnn/piano_model.h5")

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

first_input = numberic_text[117 : 117+25]
first_input = tf.one_hot([first_input], text_size)
first_input = tf.expand_dims(first_input, axis=0)

music = []

for i in range(200):
    predict = Pmodel.predict(first_input)
    predict = np.argmax(predict[0], axis=1)  # 최댓값

    music.append(predict)

    next_input = first_input.numpy().tolist()[0][0][1:]  # 첫입력값 앞에 짜르기
    one_hot_num = tf.one_hot([predict], text_size).numpy().tolist()[0]  # 예측한 다음문자를 원핫인코딩하기
    next_input.append(one_hot_num)  # 예측한 다음문자를 뒤에 넣기

    first_input = np.vstack([next_input, one_hot_num.numpy()])  # expand dims
    first_input = tf.expand_dims(first_input, axis=0)

print(music)
"""
0. 첫 입력값 만들기
1. predict로 다음문제 예측
2. 예측한 다음문제 리스트에 저장하기
3. 첫입력값 앞에 짜르기
4. 예측한 다음문자를 뒤에 넣기
5. 원핫인코딩하기, expand dims
"""