# Deep-Learning

Deep learning is a subfield of machine learning that is focused on training artificial neural networks to learn and make predictions from data. These neural networks are modeled after the human brain and are composed of layers of interconnected nodes that are trained using large amounts of data. Deep learning has been successful in a wide range of applications, including image and speech recognition, natural language processing, and autonomous vehicles, among others. It has revolutionized many industries and continues to be an area of active research and development.



### Import the necessary libraries and modules:
from keras.models import Sequential
from keras.layers import Dense

### Create a sequential model:
model = Sequential()


### Add layers to the model using the add() method:
model.add(Dense(units=64, activation='relu', input_dim=100))
model.add(Dense(units=10, activation='softmax'))


### Compile the model:
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])


### Fit the model to the training data:
model.fit(X_train, Y_train, epochs=5, batch_size=32)


### Evaluate the model on the test data:
loss_and_metrics = model.evaluate(X_test, Y_test, batch_size=128)


### Make predictions using the model:
classes = model.predict(X_test, batch_size=128)
