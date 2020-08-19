import numpy as np
import pandas as pd
import tensorflow as tf
import sys

dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:-1].values  # все кроме последней колонки
y = dataset.iloc[:, -1].values  # только последняя колонка

#encodeing categorical data
#ЗАКОДИРУЕМ ПОЛ
# закодируем наши игрики
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
X[:,2]=le.fit_transform(X[:,2])

#ЗАКОДИРУЕМ СТРАНУ
# зашивруем категории в виде вектора
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
# 1 элемент - вид трансвормации, 2 - класс странсформера, 3 - колонка для кодирования
# remainder - инструкция для трансформации, говорит расширить
ct=ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[1])],remainder='passthrough')
X=ct.fit_transform(X)

#spliting
# разодьем данные на тестовые и проверочные
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2, random_state=1)#random_state=1 убирает рандом что он всегжа одинаков


# feature scaling
from sklearn.preprocessing import StandardScaler
ss=StandardScaler() #сколько среднеквадратичных отклонений содержит наша величина
X_train=ss.fit_transform(X_train)#применяем к тестовой выборке
# когда мы вызываем fit_transform мы (1) готовим модель кторая конвертирует, а потом на основе ее изменяем наши данные
X_test=ss.transform(X_test) # тут только transform потому что мы ТОЛЬКО ЧТО создали модель странсформации, и среднее и отклонение УЖЕ расчитаны, поэтому только меняем









#Building ANN

#initialize the ANN
ann=tf.keras.models.Sequential()
#ading the input layer and the first hidden layer
ann.add(tf.keras.layers.Dense(units=6,activation='relu'))
#ading the input layer and the seconf hidden layer
ann.add(tf.keras.layers.Dense(units=6,activation='relu'))
#adding output layer
ann.add(tf.keras.layers.Dense(units=1,activation='sigmoid'))
#если мы предстказываем категории то активация должна быть softmax

#Train ANN
#compiling the ANN
ann.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
#Loss такой потому что мы предсказываем бинарную переменную иначе categorical_crossentropy
#если классификация не бинарная categorical_crossentropy

#Training the Ann on the training set
ann.fit(X_train,y_train,batch_size=32,epochs=100)

#making predictions
single_predict_vector=np.array([[600,'France','Male',40,3,60000,2,1,1,50000]])

single_predict_vector[:,2]=le.transform(single_predict_vector[:,2])
single_predict_vector=ct.transform(single_predict_vector)
single_predict_vector=ss.transform(single_predict_vector)


result=ann.predict(single_predict_vector)
print(result>0.5)

predicted_results=ann.predict(X_test)
predicted_results=(predicted_results>0.5)
print(np.concatenate(
    (y_test.reshape(len(y_test), 1),
     predicted_results.reshape(len(predicted_results), 1)
     ),
    1))

# making confusion matrix
# количество правильных и не правильных предсказаний
from sklearn.metrics import confusion_matrix, accuracy_score
cm=confusion_matrix(y_test,predicted_results)
print(cm)
print(accuracy_score(y_test,predicted_results)) # вернет от 0 до 1