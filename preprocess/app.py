import pandas as pd
import tensorflow as tf
import numpy as np

data = pd.read_csv('train.csv')
print(data, data.isnull().sum())

data['Age'] = data['Age'].fillna(value=30)
data['Embarkder'] = data['Embarkder'].fillna(value='S')

y = data.pop('Survived')

# 전처리들 방법 숫자 즉 압축 0~1사이로 변환 -> normalize이라고함 
# 숫자들 normalize 레이어
fare_preprocess_layer = tf.keras.layers.Normalization(axis=None)
fare_preprocess_layer.adapt(np.array(data['Fare']))

sib_preprocess_layer = tf.keras.layers.Normalization(axis=None)
sib_preprocess_layer.adapt(np.array(data['SibSp']))

Parch_preprocess_layer = tf.keras.layers.Normalization(axis=None)
Parch_preprocess_layer.adapt(np.array(data['Parch']))

Pclass_preprocess_layer = tf.keras.layers.Normalization(axis=None)
Pclass_preprocess_layer.adapt(np.array(data['Pclass']))

# print(fare_preprocess_layer(np.array(data['Fare'])))  노말리제이션 확인 

# Discretization 나이구간같이 나이별 차이가 없을경우 10대, 20대로 분리해서 처리함 
Age_preprocess_layer = tf.keras.layers.Discretization(bin_boundaries=[10, 20, 30, 40, 50, 60])

# String Lookup 문자형 범주형 데이터 one hot 인코딩
Sex_preprocess_layer = tf.keras.layers.StringLookup(output_mode='one_hot')
Sex_preprocess_layer.adapt(np.array(data['Sex']))

Embarkder_preprocess_layer = tf.keras.layers.StringLookup(output_mode='one_hot')
Embarkder_preprocess_layer.adapt(np.array(data['Embarkder']))

# 임베딩 
# 사용법은 Ticket 데이터 -> 정수로 전처리해주는 레이어 -> 임베딩해주는 레이어
Ticket_preprocess_layer = tf.keras.layers.StringLookup() # -> 문자들을 각각 유니크한 정수로 치환해줌
Ticket_preprocess_layer.adapt(np.array(data['Ticket']))

data['Ticket'].unique()
Ticket_embedding_layer = tf.keras.layers.Embedding(data['Ticket'].unique() + 1, 9)  # input_dim: 고유값의 개수, output_dim: 임베딩 차원 수

input_fare = tf.keras.Input(shape=(1,), name='Fare')
input_parch = tf.keras.Input(shape=(1,), name='Parch')
input_sibsp = tf.keras.Input(shape=(1,), name='SibSp')
input_pclass = tf.keras.Input(shape=(1,), name='Pclass')
input_age = tf.keras.Input(shape=(1,), name='Age')
input_sex = tf.keras.Input(shape=(1,), name='Sex', dtype=tf.string)
input_embarked= tf.keras.Input(shape=(1,), name='Embarked', dtype=tf.string)
input_ticket = tf.keras.Input(shape=(1,), name='Ticket', dtype=tf.string)

# 1. 맨 위는 Input레이어
input_fare = tf.keras.Input(shape=(1,), name='Fare')
input_parch = tf.keras.Input(shape=(1,), name='Parch')
input_sibsp = tf.keras.Input(shape=(1,), name='SibSp')
input_pclass = tf.keras.Input(shape=(1,), name='Pclass')
input_age = tf.keras.Input(shape=(1,), name='Age')
input_sex = tf.keras.Input(shape=(1,), name='Sex', dtype=tf.string)
input_embarked= tf.keras.Input(shape=(1,), name='Embarked', dtype=tf.string)
input_ticket = tf.keras.Input(shape=(1,), name='Ticket', dtype=tf.string)

# 2. Input을 전처리레이어에 넣기
x_fare = fare_preprocess_layer(input_fare)
x_parch = Parch_preprocess_layer(input_parch)
x_sibsp = sib_preprocess_layer(input_sibsp)
x_pclass = Pclass_preprocess_layer(input_pclass)
x_age = Age_preprocess_layer(input_age)
x_sex = Sex_preprocess_layer(input_sex)
x_embarked = Embarkder_preprocess_layer(input_embarked)
x_ticket_onehot = Ticket_preprocess_layer(input_ticket)
x_ticket_embed = Ticket_embedding_layer(x_ticket_onehot)
x_ticket = tf.keras.layers.Flatten()(x_ticket_embed)

# 3. 레이어들 하나로 합치기
concat1 = tf.keras.layers.concatenate([
  x_fare, x_parch, x_sibsp, x_pclass, x_age, x_sex, x_embarked, x_ticket
])

x = tf.keras.layers.Dense(128, activation='relu')(concat1)
x = tf.keras.layers.Dense(64, activation='relu')(x)
마지막레이어 = tf.keras.layers.Dense(1, activation='sigmoid')(x)


# 4, 모델의 시작과 끝 레이어 알려주기 
model = tf.keras.Model(
    inputs=[input_fare, input_parch, input_sibsp, input_pclass, input_age, input_sex, input_embarked, input_ticket],
    outputs=마지막레이어
)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

x_data = {
    'Fare': np.array(data['Fare']),
    'Parch': np.array(data['Parch']),
    'SibSp': np.array(data['SibSp']),
    'Pclass': np.array(data['Pclass']),
    'Age': np.array(data['Age']),
    'Sex': np.array(data['Sex']),
    'Embarked': np.array(data['Embarked']),
    'Ticket': np.array(data['Ticket'])
}
model.fit(x_data, np.array(y), epochs=15, validation_split=0.1) 