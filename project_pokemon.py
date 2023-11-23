from numpy import loadtxt
from keras.model import Sequential
from keras.layers import Dense

dataset = loadtxt('pokemon.csv', delimiter=',')

x = dataset[:,1:7]
y = dataset[:,8]

model = sequential()
model = add(dense(12, input_dime=6, activation='relu'))
model = add(dense(8, activation='relu'))
model = add(dense(1, activation='sigmod'))

model.compile(loss='binary_crossentropy', metrics=['accuracy'])
model.fit(x, y, epochs=250, batch_size=100)
predictions = model.predict_classes(x)
for i in range(785,800):
    print(f'{x[i].tolist()} => {predictions} expected {y[i]}')
