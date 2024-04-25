
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import tensorflow_datasets as tfds

from tensorflow import keras


ds, info= tfds.load("eurosat/rgb", with_info= True, split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'], as_supervised=True)
train_ds, val_ds, test_ds = ds
print(info)

print("Image shape: {}".format(info.features['image'].shape))
print("Label shape: {}".format(info.features['label'].shape))
print("Number of {}: {}".format(info.description.split(' ')[15], info.features['label'].num_classes))
print("Class names: {}".format(info.features['label'].names))
print("Number of training examples: {}".format(info.splits['train'].num_examples))

# your code
def count_occurences(ds, info):
  out = []
  for index, label_name in enumerate(info.features['label'].names):
    out.append(len(list(ds.filter(lambda img, label: label == index))))
  return out

train_occurences = count_occurences(train_ds, info)
validation_occurences = count_occurences(val_ds, info)
test_occurences = count_occurences(test_ds, info)
# your code
print("train_ds:\t {}".format(train_occurences))
print("validation_ds:\t {}".format(validation_occurences))
print("test_ds:\t {}".format(test_occurences))
print("data format:\t {}".format((train_ds.take(1).element_spec[0].dtype)))

item = train_ds.take(1)

for image, label in item:
  print(np.min(image))
  print(np.max(image))

def preprocess(image,label):
    return (tf.cast(image, tf.float32) / 255, label)

train_ds= train_ds.map(preprocess)
val_ds= val_ds.map(preprocess)
test_ds= test_ds.map(preprocess)


# your code
batch_size = 32

train_ds = train_ds.repeat()
train_ds = train_ds.shuffle(buffer_size=1024, seed=0)
train_ds = train_ds.batch(batch_size=batch_size)
train_ds = train_ds.prefetch(buffer_size=1)

for image, label in train_ds.take(1):
    print(image.shape)
    plt.imshow(image[0, ...], cmap='gray')
    plt.xticks([])
    plt.yticks([])
    plt.xlabel(f'Label: {label}')
    plt.show()

test_batchsize = 1
test_ds = test_ds.batch(batch_size=test_batchsize)
test_ds = test_ds.prefetch(buffer_size=1)

val_ds = val_ds.batch(batch_size=batch_size)
val_ds = val_ds.prefetch(buffer_size=1)

model = tf.keras.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=info.features["image"].shape))
model.add(tf.keras.layers.Dense(256, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

model.summary()

epochs =5 # hint: you need this here

steps_per_epoch = sum(train_occurences) / batch_size

starter_learning_rate = 1e-4
end_learning_rate = 1e-8
decay_steps = 10000
scheduler = tf.keras.optimizers.schedules.PolynomialDecay(initial_learning_rate= starter_learning_rate,
    decay_steps= decay_steps,
    end_learning_rate= end_learning_rate,
    power=1)
early_stopping = tf.keras.callbacks.EarlyStopping(patience=7, min_delta=0.001, restore_best_weights=True)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=scheduler),
              loss='sparse_categorical_crossentropy',
              metrics=['sparse_categorical_accuracy'])

# your code
history = model.fit(train_ds, epochs=epochs, steps_per_epoch = steps_per_epoch, validation_data=val_ds, validation_steps = sum(validation_occurences)/batch_size,
              callbacks=[early_stopping])


sns.set_style('darkgrid', {'axes.facecolor': '.9'})
sns.set_context('notebook')

# your code
### Learning curves
history_frame = pd.DataFrame(history.history)
history_frame.plot(figsize=(8, 5))
plt.show()

# your code
evaluate_train = model.evaluate(train_ds, batch_size=batch_size, steps=steps_per_epoch)

steps_per_epoch_val = sum(validation_occurences) / batch_size
evaluate_validation = model.evaluate(val_ds, batch_size=batch_size, steps=steps_per_epoch_val)

# get one batch from train_ds and select first item
for item in train_ds.take(1):
    demo_image = item[0][:1, ...]
    demo_label = item[1][0]

# compute hidden activations, model output and predicted_label
flattened_input = model.layers[0](demo_image) #None # your code
hidden_activations = model.layers[1](flattened_input)
output = model.layers[2](hidden_activations)
predicted_label = np.argmax(output.numpy())
# # print(flattened_input)
# # print(hidden_activations)
print('Output: ', output)
print('\ndemo_label: ',demo_label)
print('predicted_label: ',predicted_label)

    
# plot demo_image, print demo_label
plt.imshow(demo_image[0, ...], cmap='gray')
plt.xticks([])
plt.yticks([])
plt.xlabel('Input demo_image')
plt.show()
print(f'True label: {demo_label} ({info.features["label"].names[demo_label]})\n')

# print my predicted class probabilities and demo_label
print('\nMy Predicted Class Probabilities:\n')
for i in range(output.shape[1]):
    print(f'{i}: {output[0][i]:.5f}')
print(f'\nMy Predicted label: {predicted_label} ({info.features["label"].names[predicted_label]})')

# print model predicted class probabilities
model_output = model.predict(demo_image)
print('\nModel Predicted Class Probabilities:\n')
for i in range(model_output.shape[1]):
    print(f'{i}: {model_output[0][i]:.5f}')

results = pd.DataFrame(columns=['start_learning_rate', 'width', 'depth', 'l2_weight', 'train_loss', 'val_loss', 'train_acc', 'val_acc'])

start_learning_rates = [1e-5]
widths = [256]
depths = [1
l2_weights = [0]

for start_learning_rate in start_learning_rates:
    for width in widths:
        for depth in depths:
            for l2_weight in l2_weights:
                model = keras.models.Sequential()
                model.add(tf.keras.layers.Flatten(input_shape=info.features["image"].shape))
                for _ in range(depth):
                    model.add(keras.layers.Dense(units=width, activation="relu", kernel_regularizer=keras.regularizers.l2(l2_weight)))
                model.add(tf.keras.layers.Dense(10, activation='softmax'))
                scheduler = tf.keras.optimizers.schedules.PolynomialDecay(start_learning_rate, epochs * sum(train_occurences) // batch_size, 1e-8, power=1.0)
                model.compile(loss=keras.losses.sparse_categorical_crossentropy, optimizer=keras.optimizers.Adam(learning_rate=scheduler), metrics=[keras.metrics.sparse_categorical_accuracy])
                history = model.fit(train_ds, epochs=epochs, steps_per_epoch = steps_per_epoch, validation_data=val_ds, validation_steps = sum(validation_occurences)/batch_size,callbacks=[early_stopping],verbose=0)
                train_loss, train_acc = model.evaluate(train_ds, steps=np.sum(train_occurences) // batch_size)
                val_loss, val_acc = model.evaluate(val_ds)
                results_tmp = np.array([start_learning_rate, width, depth, l2_weight, train_loss, val_loss, train_acc, val_acc]).reshape(1, -1)
                results = results.append(pd.DataFrame(data=results_tmp, columns=results.columns), ignore_index=True)
results.to_csv('resultdds.csv')

