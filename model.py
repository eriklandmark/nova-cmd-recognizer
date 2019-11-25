import tensorflow as tf

def get_model(num_labels):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=(118, 128, 1)))
    model.add(tf.keras.layers.Conv2D(filters=8, kernel_size=[4,2], activation="relu"))
    model.add(tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=[2, 2]))

    model.add(tf.keras.layers.Conv2D(filters=16, kernel_size=[4,2], activation="relu"))
    model.add(tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=[2, 2]))

    model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=[4,2], activation="relu"))
    model.add(tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=[2, 2]))

    model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=[4,2], activation="relu"))
    model.add(tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=[1, 2]))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dropout(0.25))
    model.add(tf.keras.layers.Dense(1000, activation="relu"))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(num_labels, activation="softmax"))

    model.compile(loss=tf.keras.losses.categorical_crossentropy,
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
        metrics=['accuracy']
    )

    return model

if __name__ == "__main__":
    model = get_model(2)
    model.summary()