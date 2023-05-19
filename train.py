import json
import pickle
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers
from keras.callbacks import History
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os

def trainMLP(metric, epochs, hidden_layer_neuron_number, learning_rate, batch_size, test_set_percentage):
    # definindo o modelo
    mlp = tf.keras.models.Sequential()
    mlp.add(layers.Dense(784, activation='sigmoid', input_shape=(784,)))
    mlp.add(layers.Dense(hidden_layer_neuron_number, activation='sigmoid'))
    mlp.add(layers.Dense(10, activation='sigmoid'))

    # compilando o modelo
    optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    mlp.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=[metric])

    # lendo os dados do arquivo de texto
    data = np.loadtxt('data_tp1', delimiter=',')
    X = data[:, 1:]
    y = data[:, 0]

    history = History()
    mlp.fit(X, y, validation_split=test_set_percentage, epochs=epochs, batch_size=batch_size, verbose=1, shuffle=True, workers=8, callbacks=[history])
    return (mlp, history)

def saveMLP(base_folder, mlp, learning_rate, history):
    if not os.path.exists('models'):
        os.makedirs('models')

    with open(f"{base_folder}/history-lr{learning_rate}.pkl", 'wb') as file:
        pickle.dump(history.history, file)
        
    mlp.save(f"{base_folder}/mlp-lr{learning_rate}")

params = json.load(open('params.json'))

test_set_percentage = params['test_set_percentage']
metric = params['metric']
epochs = params['epochs']

# Ordenado de forma que as primeiras execuções sejam mais rápidas
batch_sizes = [round(5000 * round(1 - params['test_set_percentage'], 2))]
for batch_size in params.get('batch_sizes'):
    batch_sizes.append(batch_size)
# batch_sizes = [3750, 50, 10, 1] Originalmente

hidden_layer_neuron_numbers = params['hidden_layer_neuron_numbers']
learning_rates = params['learning_rates']

for batch_size in batch_sizes:
    batch_folder = f"models/batch-size-{batch_size}"
    if not os.path.exists(batch_folder):
        os.makedirs(batch_folder)

    for hidden_layer_neuron_number in hidden_layer_neuron_numbers:
        hidden_layer_folder = f"{batch_folder}/hidden-layer-neuron-number-{hidden_layer_neuron_number}"
        if not os.path.exists(hidden_layer_folder):
            os.makedirs(hidden_layer_folder)

        for learning_rate in learning_rates:
            mlp, history = trainMLP(metric, epochs, hidden_layer_neuron_number, learning_rate, batch_size, test_set_percentage)
            saveMLP(hidden_layer_folder, mlp, learning_rate, history)

# mlp = trainMLP(metric, epochs, hidden_layer_neuron_numbers[2], learning_rates[2], batch_sizes[0])