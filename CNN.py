import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout, Reshape, BatchNormalization
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import ImageDataGenerator

dataGen_training = ImageDataGenerator(
  rotation_range=10,
  horizontal_flip=True,
  vertical_flip = False,
  width_shift_range=0.1,
  height_shift_range=0.1,
  rescale=1. / 255,
  shear_range=0.05,
  zoom_range=0.05,
)

dataGen_testing = ImageDataGenerator(
  rescale=1. / 255,
)


def scheduler(lr, epochs):
    if epochs < 30:
        return lr
    else:
        lr *= tf.math.exp(-0.04)


def plot_graphs(all_logs):
    for logs in all_logs:
        losses = logs.history['loss']
        name = logs.history['name']
        plt.plot(list(range(len(losses))), losses, label=name)
    plt.xlabel("number of epochs")
    plt.ylabel("error")
    plt.title("error on training data")
    plt.legend()
    plt.show()

    for logs in all_logs:
        losses = logs.history['val_loss']
        name = logs.history['name']
        plt.plot(list(range(len(losses))), losses, label=name)
    plt.xlabel("number of epochs")
    plt.ylabel("error")
    plt.title("error on testing data")
    plt.legend()
    plt.show()

    for logs in all_logs:
        metric = logs.history['categorical_accuracy']
        name = logs.history['name']
        plt.plot(list(range(len(metric))), metric, label=name)
    plt.xlabel("number of epochs")
    plt.ylabel("accuracy")
    plt.legend()
    plt.title("prediction accuracy on training test")
    plt.show()

    for logs in all_logs:
        metric = logs.history['val_categorical_accuracy']
        name = logs.history['name']
        plt.plot(list(range(len(metric))), metric, label=name)
    plt.xlabel("number of epochs")
    plt.ylabel("accuracy")
    plt.title("prediction accuracy on testing test")
    plt.legend()
    plt.show()


def linear_model(x, y, val_x, val_y, opt, loss_func, epochs, batch_size):
    model = keras.Sequential([
        Flatten(),
        Dense(10, activation=keras.activations.softmax),
    ])

    model.compile(optimizer=opt, loss=loss_func, metrics=keras.metrics.categorical_accuracy)

    logs = model.fit(x, y, validation_data=(val_x, val_y), epochs=epochs, batch_size=batch_size,
                     callbacks=[keras.callbacks.EarlyStopping(patience=20),
                                keras.callbacks.LearningRateScheduler(scheduler)])

    return logs


def multi_layer_perceptron(x, y, val_x, val_y, opt, loss_func, epochs, batch_size, activation, dropout):

    model = keras.Sequential([
        # convert a two dimensional matrix into a vector
        Flatten(),
        Dense(1000, activation=activation),
        Dropout(dropout),
        Dense(820, activation=activation),
        Dropout(dropout),
        Dense(580, activation=activation),
        Dropout(dropout),
        Dense(360, activation=activation),
        Dropout(dropout),
        Dense(240, activation=activation),
        Dropout(dropout),
        Dense(160, activation=activation),
        Dropout(dropout),
        Dense(80, activation=activation),
        Dropout(dropout),
        Dense(10, activation=keras.activations.softmax),
    ])

    model.compile(optimizer=opt, loss=loss_func, metrics=keras.metrics.categorical_accuracy)

    logs = model.fit(x, y, validation_data=(val_x, val_y), epochs=epochs, batch_size=batch_size,
                     callbacks=[keras.callbacks.LearningRateScheduler(scheduler),
                                keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)])
    model.summary()

    return logs


def convolutional_neural_network(activation):
    model = keras.Sequential([
        Reshape((32, 32, 3)),
        BatchNormalization(),

        Conv2D(196, (3, 3), padding="same", activation=activation),
        BatchNormalization(),
        Conv2D(196, (3, 3), padding="same", activation=activation),
        BatchNormalization(),

        MaxPool2D(),

        Conv2D(92, (3, 3), padding="same", activation=activation),
        BatchNormalization(),
        Conv2D(92, (3, 3), padding="same", activation=activation),
        BatchNormalization(),
        MaxPool2D(),

        Conv2D(48, (3, 3), padding="same", activation=activation),
        BatchNormalization(),
        Conv2D(48, (3, 3), padding="same", activation=activation),
        BatchNormalization(),

        Flatten(),

        Dense(10, activation=keras.activations.softmax)
    ])

    model.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.categorical_crossentropy,
                  metrics=keras.metrics.categorical_accuracy)

    logs = model.fit_generator(
        train_generator,
        steps_per_epoch=len(x_train) // batch_size,
        epochs=180,
        callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=30)],
        validation_data=validation_generator,
        validation_freq=1,
        validation_steps=valid_steps,
        verbose=2,
    )

    model.summary()

    return logs


if __name__ == "__main__":
    epochs = 500
    batch_size = 256

    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

    x_train = x_train / 255.0
    x_test = x_test / 255.0

    '''
    x_train = tf.cast(x_train, tf.float32)
    x_test = tf.cast(x_test, tf.float32)
    y_train = tf.cast(y_train, tf.float32)
    y_test = tf.cast(y_test, tf.float32)'''

    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    train_generator = dataGen_training.flow(x_train, y_train, batch_size=batch_size)

    x_valid = x_train[:150 * batch_size]
    y_valid = y_train[:150 * batch_size]

    valid_steps = x_valid.shape[0] // batch_size
    validation_generator = dataGen_testing.flow(x_valid, y_valid, batch_size=batch_size)

    all_logs = []

    '''
    log = linear_model(x_train, y_train, x_test, y_test, keras.optimizers.Adam(),
                       keras.losses.categorical_crossentropy, epochs, batch_size)
    log.history['name'] = "linear model"
    all_logs.append(log)
    '''
    '''
    log = multi_layer_perceptron(x_train, y_train, x_test, y_test, keras.optimizers.Adam(learning_rate=0.002),
                                     keras.losses.categorical_crossentropy, epochs, batch_size, "elu", 0.28)
    log.history['name'] = "elu"
    all_logs.append(log)
    '''

    log = convolutional_neural_network("relu")
    log.history['name'] = "relu"
    all_logs.append(log)

    plot_graphs(all_logs)
