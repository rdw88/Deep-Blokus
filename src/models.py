import tensorflow as tf



'''
The model takes an encoded Blokus board as input and outputs
the predicted Q-value for the given state of the Blokus board
'''
def blokus_model():
    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Dense(2000, activation='relu', input_shape=(2000,)))
    model.add(tf.keras.layers.Dense(1024, activation='relu'))
    model.add(tf.keras.layers.Dense(512, activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001, amsgrad=True),
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=[
            tf.keras.metrics.MeanAbsoluteError()
        ]
    )

    return model