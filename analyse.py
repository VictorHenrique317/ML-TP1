from sklearn.model_selection import train_test_split
from keras.models import load_model
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os

metric = 'accuracy'
test_set_percentage = 0.25
epochs = 200 # 200 parece ser o suficiente

# Ordenado de forma que as primerias execuções sejam mais rápidas
GD_batch_size = 5000 * round(1 - test_set_percentage, 2)
GD_batch_size = round(GD_batch_size)
batch_sizes = [GD_batch_size, 50, 10, 1]
hidden_layer_neuron_numbers = [25, 50, 100]
learning_rates = [10, 1, 0.5]

models = {}
for batch_size in [GD_batch_size]:
    batch_folder = f"models/batch-size-{batch_size}"
    if not os.path.exists(batch_folder):
        os.makedirs(batch_folder)

    for hidden_layer_neuron_number in hidden_layer_neuron_numbers:
        hidden_layer_folder = f"{batch_folder}/hidden-layer-neuron-number-{hidden_layer_neuron_number}"
        
        for learning_rate in learning_rates:
            mlp = load_model(f"{hidden_layer_folder}/mlp-lr{learning_rate}")

            if models.get(batch_size) is None:
                models[batch_size] = {}

            if models[batch_size].get(hidden_layer_neuron_number) is None:
                models[batch_size][hidden_layer_neuron_number] = {}
            
            models[batch_size][hidden_layer_neuron_number][learning_rate] = mlp
        
print(models)
# for learning_rate, model_group in models.items():
#     pass
#     for mlp in model_group:
#         pass
#         plt.plot(mlp.history.history['loss'])
#         plt.plot(mlp.history.history['val_loss'])
#         plt.title('Erro empírico e erro de teste')
#         plt.ylabel('Erro')
#         plt.xlabel('Época')
#         plt.legend(['Empírico', 'Teste'], loc='upper left')

#     plt.show()