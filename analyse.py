import pickle
from sklearn.model_selection import train_test_split
from keras.models import load_model
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os
import json

params = json.load(open('params.json'))

test_set_percentage = params['test_set_percentage']
metric = params['metric']
epochs = params['epochs']

# Ordenado de forma que as primeiras execuções sejam mais rápidas
batch_sizes = [round(5000 * round(1 - params['test_set_percentage'], 2))]
for batch_size in params.get('batch_sizes'):
    batch_sizes.append(batch_size)

hidden_layer_neuron_numbers = params['hidden_layer_neuron_numbers']
learning_rates = params['learning_rates']

models = {}
# loads the models
for batch_size in batch_sizes:
    batch_folder = f"models/batch-size-{batch_size}"
    if not os.path.exists(batch_folder):
        os.makedirs(batch_folder)

    for hidden_layer_neuron_number in hidden_layer_neuron_numbers:
        hidden_layer_folder = f"{batch_folder}/hidden-layer-neuron-number-{hidden_layer_neuron_number}"
        
        for learning_rate in learning_rates:
            mlp = load_model(f"{hidden_layer_folder}/mlp-lr{learning_rate}")
            history = pickle.load(open(f"{hidden_layer_folder}/history-lr{learning_rate}.pkl", 'rb'))

            if models.get(batch_size) is None:
                models[batch_size] = {}

            if models[batch_size].get(hidden_layer_neuron_number) is None:
                models[batch_size][hidden_layer_neuron_number] = {}
            
            models[batch_size][hidden_layer_neuron_number][learning_rate] = (mlp, history)

# plots the models
for batch_size, model_group in models.items():
    for hidden_layer_neuron_number, model_group in model_group.items():
        colors = ['blue', 'green', 'red']
        labels = []
        color_index = -1

        for learning_rate, model in model_group.items():
            color_index += 1
            color = colors[color_index]
            mlp, history = model

            plt.plot(history['val_loss'], color=color, linestyle='dashed')
            plt.plot(history['loss'], color=color, linestyle='solid')
            
            labels.append(f"Teste (LR={learning_rate})")
            labels.append(f"Empírico (LR={learning_rate})")

        plt.title('Erro de teste e erro empírico')
        plt.ylabel('Erro')
        plt.xlabel('Época')
        plt.legend(labels, loc='upper right')

        save_folder = "plots"
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        save_folder = f"{save_folder}/batch-size-{batch_size}"
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        save_path = f"{save_folder}/hidden-layer-neuron-number-{hidden_layer_neuron_number}"
        plt.savefig(f"{save_path}.png")
        plt.clf()