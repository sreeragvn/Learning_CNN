import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import tensorflow_datasets as tfds

from tensorflow import keras

epochs = 100
batch_size = 32
results = pd.DataFrame(columns=['start_learning_rate', 'width', 'depth', 'l2_weight', 'train_loss', 'val_loss', 'train_acc', 'val_acc'])

start_learning_rates = [1e-5, 1e-4, 1e-3]
widths = [256, 512, 1024]
depths = [1, 2, 3]
l2_weights = [0, 1e-5, 1e-4]

for start_learning_rate in start_learning_rates:
    for width in widths:
        for depth in depths:
            for l2_weight in l2_weights:
                model = keras.models.Sequential()
                # your code (add input layer)
                for _ in range(depth):
                    model.add(keras.layers.Dense(units=width, activation="relu", kernel_regularizer=keras.regularizers.l2(l2_weight)))
                # your code (add output layer)
                scheduler = tf.keras.optimizers.schedules.PolynomialDecay(start_learning_rate, epochs * sum(train_occurences) // batch_size, 1e-8, power=1.0)
                model.compile(loss=keras.losses.sparse_categorical_crossentropy, optimizer=keras.optimizers.Adam(learning_rate=scheduler), metrics=[keras.metrics.sparse_categorical_accuracy])
                # your code (call fit function with verbose=0)
                train_loss, train_acc = model.evaluate(train_ds, steps=np.sum(train_occurences) // batch_size)
                val_loss, val_acc = model.evaluate(val_ds)
                results_tmp = np.array([start_learning_rate, width, depth, l2_weight, train_loss, val_loss, train_acc, val_acc]).reshape(1, -1)
                results = results.append(pd.DataFrame(data=results_tmp, columns=results.columns), ignore_index=True)
results.to_csv('results.csv')