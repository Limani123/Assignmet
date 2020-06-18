from tensorflow.keras.models import load_model
import csv
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Conv1D, Flatten, GlobalMaxPooling1D
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from keras.utils import to_categorical
import numpy as np

#reading the input dataset
data = pd.read_csv('LabelledData.csv',delimiter='\t',header=None)
data[1] = data[1].str.strip()

#identifying the unique labels
aa = data[1].to_list()
unique_label = set(aa)
label1 = list(unique_label)

#encoding the text labels into integer labels
label_encoder = {}
count = 0
for i in unique_label:
    label_encoder[i] = count
    count +=1

#replacing all the text labels into integer labels
y_encoded = []
for i in aa:
    y_encoded.append(label_encoder[i])
label_encoded_array = np.array(y_encoded) 
aa = to_categorical(y_encoded, num_classes=len(label_encoder))

X_train, X_test, y_train, y_test = train_test_split(data[0], aa, test_size=0.6, stratify=aa, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, stratify=y_train, random_state=42) 
# print(X_train.shape, y_train.shape, X_val.shape, y_val.shape, X_test.shape, y_test.shape)

#Universal-sentence-encoder
loaded_obj = 'https://tfhub.dev/google/universal-sentence-encoder/4'

# Our pre-trained model which is being trained on our dataset
model = Sequential()
model.add(hub.KerasLayer(loaded_obj, input_shape=[], dtype=tf.string, trainable=True))
#model.add(Dense(64, activation='relu'))
#model.add(Dropout(0.2))
model.add(Dense(5, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
checkpoint = tf.keras.callbacks.ModelCheckpoint('model.h5', monitor='val_loss', save_best_only=True, verbose=1, save_weights_only=False)
history = model.fit(X_train, y_train, batch_size=16, epochs=6, shuffle=True, validation_data=(X_val,y_val), callbacks=[checkpoint])

model.save('model.h5')
model = load_model('model.h5', custom_objects={'KerasLayer':hub.KerasLayer})
model.summary()

# print test accuracy
model.evaluate(X_test, y_test)


##### Enter user input

x = [input()]
pred = model.predict(x)
tt = np.argmax(pred)
print(tt)
print(label1[tt])
