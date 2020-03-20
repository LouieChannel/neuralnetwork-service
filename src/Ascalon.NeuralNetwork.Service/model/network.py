import tensorflow as tf


def create_model(training=True):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input((50, 7,)))
    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(8, return_sequences=True, kernel_initializer=tf.keras.initializers.GlorotNormal(),
                                                                 trainable=training)))
    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(4, kernel_initializer=tf.keras.initializers.GlorotNormal(),
                                                                 trainable=training)))
    model.add(tf.keras.layers.Dense(8, activation='relu', kernel_initializer=tf.keras.initializers.GlorotNormal(),
                                    trainable=training))
    model.add(tf.keras.layers.Dense(5, activation="softmax", kernel_initializer=tf.keras.initializers.GlorotNormal(), trainable=training))
    return model
