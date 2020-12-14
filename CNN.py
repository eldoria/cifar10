import tensorflow as tf
import tensorflow.keras as keras
import matplotlib.pyplot as plt


def scheduler(epoch, lr):
    if epoch < 30:
        return lr
    else:
        return lr * tf.math.exp(-0.05)


data_augmentation = tf.keras.Sequential([
  keras.layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
  keras.layers.experimental.preprocessing.RandomRotation((-0.2, 0.2)),
  keras.layers.experimental.preprocessing.RandomZoom((0, 0.3)),
  # keras.layers.experimental.preprocessing.RandomContrast(0.5),
  # keras.layers.experimental.preprocessing.RandomCrop(2, 2),
])


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
        keras.layers.Flatten(),
        keras.layers.Dense(10, activation=keras.activations.softmax),
    ])

    model.compile(optimizer=opt, loss=loss_func, metrics=keras.metrics.categorical_accuracy)

    logs = model.fit(x, y, validation_data=(val_x, val_y), epochs=epochs, batch_size=batch_size,
                     callbacks=[keras.callbacks.EarlyStopping(patience=20),
                                keras.callbacks.LearningRateScheduler(scheduler)])

    return logs


def multi_layer_perceptron(x, y, val_x, val_y, opt, loss_func, epochs, batch_size, activation, dropout):

    model = keras.Sequential([
        # convert a two dimensional matrix into a vector
        keras.layers.Flatten(),
        keras.layers.Dense(1000, activation=activation),
        keras.layers.Dropout(dropout),
        keras.layers.Dense(820, activation=activation),
        keras.layers.Dropout(dropout),
        keras.layers.Dense(580, activation=activation),
        keras.layers.Dropout(dropout),
        keras.layers.Dense(360, activation=activation),
        keras.layers.Dropout(dropout),
        keras.layers.Dense(240, activation=activation),
        keras.layers.Dropout(dropout),
        keras.layers.Dense(160, activation=activation),
        keras.layers.Dropout(dropout),
        keras.layers.Dense(80, activation=activation),
        keras.layers.Dropout(dropout),
        keras.layers.Dense(10, activation=keras.activations.softmax),
    ])

    model.compile(optimizer=opt, loss=loss_func, metrics=keras.metrics.categorical_accuracy)

    logs = model.fit(x, y, validation_data=(val_x, val_y), epochs=epochs, batch_size=batch_size,
                     callbacks=[keras.callbacks.LearningRateScheduler(scheduler),
                                keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)])
    model.summary()

    return logs


def convolutional_neural_network(x, y, val_x, val_y, opt, loss_func, epochs, batch_size, activation, dropout):
    model = keras.Sequential([
        # data_augmentation,
        keras.layers.Reshape((32, 32, 3)),

        keras.layers.Conv2D(64, (3, 3), padding="same", activation=activation),
        keras.layers.MaxPool2D(),
        keras.layers.Dropout(dropout),

        keras.layers.Conv2D(32, (3, 3), padding="same", activation=activation),
        keras.layers.MaxPool2D(),
        keras.layers.Dropout(dropout),

        keras.layers.Conv2D(16, (3, 3), padding="same", activation=activation),
        keras.layers.MaxPool2D(),
        keras.layers.Dropout(dropout),

        keras.layers.Flatten(),

        keras.layers.Dense(64, activation=activation),
        keras.layers.Dropout(dropout),

        keras.layers.Dense(32, activation=activation),
        keras.layers.Dropout(dropout),

        keras.layers.Dense(16, activation=activation),
        keras.layers.Dropout(dropout),

        keras.layers.Dense(10, activation=keras.activations.softmax)
    ])

    model.compile(optimizer=opt, loss=loss_func, metrics=keras.metrics.categorical_accuracy)

    logs = model.fit(x, y, validation_data=(val_x, val_y), epochs=epochs, batch_size=batch_size,
                     callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=30)])

    model.summary()

    return logs


if __name__ == "__main__":
    epochs = 500
    batch_size = 512

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

    data = [("relu", "0.1")]

    for activation, dropout in data:
        dropout = float(dropout)
        log = convolutional_neural_network(x_train, y_train, x_test, y_test, keras.optimizers.Adam(learning_rate=0.002),
                                           keras.losses.categorical_crossentropy, epochs, batch_size, activation, dropout)
        log.history['name'] = activation
        all_logs.append(log)

    plot_graphs(all_logs)
