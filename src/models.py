import tensorflow as tf



def blokus_model():
    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Dense(2000, activation='relu', input_shape=(2000,)))
    model.add(tf.keras.layers.Dense(1024, activation='relu'))
    model.add(tf.keras.layers.Dense(512, activation='relu'))
    model.add(tf.keras.layers.Dense(4, activation='softmax'))

    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=[
            tf.keras.metrics.CategoricalAccuracy(),
            tf.keras.metrics.MeanSquaredError(),
            tf.keras.metrics.Precision(),
            tf.keras.metrics.CategoricalCrossentropy()
        ]
    )

    return model