
import tensorflow as tf
from numpy import moveaxis

def mobileNetV2(img_shape = (200,137,137),classes = 100, channel_first = True):
    if channel_first: tf.keras.backend.set_image_data_format('channels_first')
    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    prediction_layer = tf.keras.layers.Dense(classes)
    base_model = tf.keras.applications.MobileNetV2(input_shape=img_shape,
                                               include_top=False,
                                               weights=None, classes=classes)
    print(f'model trainable: {base_model.trainable}')
    inputs = tf.keras.Input(shape=img_shape)
    # x = moveaxis(inputs,2,0)
    x = base_model(inputs)
    x = global_average_layer(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = prediction_layer(x)
    model = tf.keras.Model(inputs, outputs)
    
    return model