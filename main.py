import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
from tensorflow.python.keras import Sequential, utils
from tensorflow.python.keras.layers import Flatten, Dense, Conv1D, MaxPool1D, Dropout


data = pd.read_csv(r'archive\mitbih_train.csv', header=None)
df = pd.DataFrame(data)  #187欄表ECG種類0~4
#print(df.head())

#plt.plot(data.iloc[0])
#plt.show()

#print(df[187].value_counts())  #class0數量與其他差距極大

#平均樣本數
class_0 = df[df[187] == 0.0].sample(8000)
class_1 = df[df[187] == 1.0]
class_2 = df[df[187] == 2.0]
class_3 = df[df[187] == 3.0]
class_4 = df[df[187] == 4.0]

new_df = pd.concat([class_0, class_1, class_2, class_3, class_4])
#print(new_df[187].value_counts())

x = df.drop(187, axis=1).values
y = df[187].values
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

x_train = np.array(x_train).reshape(x_train.shape[0], x_train.shape[1], 1)
x_test = np.array(x_test).reshape(x_test.shape[0], x_test.shape[1], 1)

model = Sequential()

model.add(Conv1D(filters=32, kernel_size=(3,), padding='same', activation='relu', input_shape = (x_train.shape[1],1)))
model.add(Conv1D(filters=64, kernel_size=(3,), padding='same', activation='relu'))
model.add(Conv1D(filters=128, kernel_size=(5,), padding='same', activation='relu'))

model.add(MaxPool1D(pool_size=(3,), strides=2, padding='same'))
model.add(Dropout(0.5))

model.add(Flatten())

model.add(Dense(units = 512, activation='relu'))
model.add(Dense(units = 1024, activation='relu'))

model.add(Dense(units=5, activation='softmax'))


model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10, batch_size=32)

y_pred = model.predict(x_test)
