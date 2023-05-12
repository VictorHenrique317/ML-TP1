from sklearn.model_selection import train_test_split
from tensorflow.keras import layers
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

    mlp.fit(X, y, validation_split=test_set_percentage, epochs=epochs, batch_size=batch_size, verbose=1, shuffle=True, workers=8)
    return mlp

def saveMLP(base_folder, mlp, learning_rate):
    if not os.path.exists('models'):
        os.makedirs('models')

    mlp.save(f"{base_folder}/mlp-lr{learning_rate}")

metric = 'accuracy'
test_set_percentage = 0.25
epochs = 200 # 200 parece ser o suficiente

# Ordenado de forma que as primerias execuções sejam mais rápidas
GD_batch_size = 5000 * round(1 - test_set_percentage, 2)
GD_batch_size = round(GD_batch_size)
batch_sizes = [GD_batch_size, 50, 10, 1]
hidden_layer_neuron_numbers = [25, 50, 100]
learning_rates = [10, 1, 0.5]

# for batch_size in batch_sizes:
for batch_size in [GD_batch_size]:
    batch_folder = f"models/batch-size-{batch_size}"
    if not os.path.exists(batch_folder):
        os.makedirs(batch_folder)

    for hidden_layer_neuron_number in hidden_layer_neuron_numbers:
        hidden_layer_folder = f"{batch_folder}/hidden-layer-neuron-number-{hidden_layer_neuron_number}"
        if not os.path.exists(hidden_layer_folder):
            os.makedirs(hidden_layer_folder)

        for learning_rate in learning_rates:
            mlp = trainMLP(metric, epochs, hidden_layer_neuron_number, learning_rate, batch_size, test_set_percentage)
            saveMLP(hidden_layer_folder, mlp, learning_rate)

# mlp = trainMLP(metric, epochs, hidden_layer_neuron_numbers[2], learning_rates[2], batch_sizes[0])

# plotando o erro de teste e o erro empirico
plt.plot(mlp.history.history['loss'])
plt.plot(mlp.history.history['val_loss'])
plt.title('Erro empírico e erro de teste')
plt.ylabel('Erro')
plt.xlabel('Época')
plt.legend(['Empírico', 'Teste'], loc='upper left')
plt.show()