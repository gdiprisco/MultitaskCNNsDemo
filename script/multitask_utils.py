import tensorflow as tf
import keras
from keras import backend as K
import numpy as np
from cv2 import cv2 as cv

MASK_VALUE = -1
# GENDER = ['Female', 'Male']
# ETHNICITY = ['African American','East Asian','Caucasian Latin','Asian Indian']
GENDER = ['♀F', '♂M']
GENDER_SYM = ["♀", "♂"]
ETHNICITY = ['Afro-American','Asian','Caucasian','Asian']
EMOTIONS = ['Surprised','Afraid','Disgusted','Happy','Sad','Angry','Neutral']

def Hswish(x):
    return x * tf.nn.relu6(x + 3) / 6

def HSigmoid(x):
    return tf.nn.relu6(x + 3) / 6

def age_relu(x):
    return keras.backend.relu(x, max_value=100)

custom_objects = {
    'age_relu': age_relu,
    'Hswish': Hswish,
    'HSigmoid': HSigmoid
}

available_versions = ["verA", "verB", "verC"]

def build_masked_loss(loss_function, mask_value=MASK_VALUE):
    """Builds a loss function that masks based on targets
    Args:       loss_function: The loss function to mask
                mask_value: The value to mask in the targets
    Returns:    function: a loss function that acts like loss_function with masked inputs
    """
    def masked_categorical_crossentropy(y_true, y_pred):
        mask = K.cast(K.not_equal(y_true, mask_value), K.floatx())
        return loss_function(y_true * mask, y_pred * mask)

    def masked_mean_squared_error(y_true, y_pred):
        mask = K.cast(K.not_equal(y_true, mask_value), K.floatx())
        masked_se_tensor = (y_true - y_pred) ** 2
        return K.sum(masked_se_tensor) / K.maximum(K.sum(mask), 1)

    if loss_function is keras.losses.mean_squared_error:
        return masked_mean_squared_error
    elif loss_function is keras.losses.binary_crossentropy or loss_function is keras.losses.categorical_crossentropy:
        return masked_categorical_crossentropy
    else:
        raise Exception("Masked loss: {} loss not supported.".format(loss_function.__name__))

def build_masked_acc(acc_function, mask_value=MASK_VALUE):

    def masked_categorical_accuracy(y_true, y_pred): #single_class_accuracy
        mask = K.cast(K.not_equal(y_true, mask_value), K.floatx())
        class_y_true = K.argmax(y_true, axis=-1)
        class_y_pred = K.argmax(y_pred, axis=-1)
        mask = K.cast(K.any(mask, axis=-1), K.floatx())
        masked_acc_tensor = K.cast(K.equal(class_y_true, class_y_pred), K.floatx()) * mask
        return K.sum(masked_acc_tensor) / K.maximum(K.sum(mask), 1)

    def masked_mean_absolute_error(y_true, y_pred):
        mask = K.cast(K.not_equal(y_true, mask_value), K.floatx())
        masked_ae_tensor = K.abs(y_true - y_pred)
        return K.sum(masked_ae_tensor) / K.maximum(K.sum(mask), 1)

    if acc_function is keras.metrics.categorical_accuracy:
        return masked_categorical_accuracy
    elif acc_function is keras.metrics.mean_absolute_error:
        return masked_mean_absolute_error
    else:
        raise Exception("Masked accuracy: {} metric not supported.".format(acc_function.__name__))

loss1 = {
    "gen1" : build_masked_loss(keras.losses.binary_crossentropy),
    "age1" : build_masked_loss(keras.losses.mean_squared_error),
    "eth1" : build_masked_loss(keras.losses.binary_crossentropy),
    "emo1" : build_masked_loss(keras.losses.binary_crossentropy),
}

loss2 = {
    "gen2" : build_masked_loss(keras.losses.binary_crossentropy),
    "age2" : build_masked_loss(keras.losses.mean_squared_error),
    "eth2" : build_masked_loss(keras.losses.binary_crossentropy),
    "emo2" : build_masked_loss(keras.losses.binary_crossentropy)
}

loss_weights1 = {
    "gen1" : 10.0,
    "age1" : 0.025,
    "eth1" : 10.0,
    "emo1" : 20.0,
}

loss_weights2 = {
    "gen2" : 10.0,
    "age2" : 0.025,
    "eth2" : 10.0,
    "emo2" : 50.0
}

accuracy1 = {
    "gen1" : build_masked_acc(keras.metrics.categorical_accuracy),
    "age1" : build_masked_acc(keras.metrics.mean_absolute_error),
    "eth1" : build_masked_acc(keras.metrics.categorical_accuracy),
    "emo1" : build_masked_acc(keras.metrics.categorical_accuracy),
}

accuracy2 = {
    "gen2" : build_masked_acc(keras.metrics.categorical_accuracy),
    "age2" : build_masked_acc(keras.metrics.mean_absolute_error),
    "eth2" : build_masked_acc(keras.metrics.categorical_accuracy),
    "emo2" : build_masked_acc(keras.metrics.categorical_accuracy)
}

def get_versioned_metrics(version):
    if version in available_versions[0:2]:
        return loss1, loss_weights1, accuracy1, 1
    elif version == available_versions[2]:
        return {**loss1, **loss2}, {**loss_weights1, **loss_weights2}, {**accuracy1, **accuracy2}, 2
    else:
        raise Exception("Version {} not supported: unable to get right losses and accuracies".format(version)) 

def get_gender(values):
    values = values.tolist()
    return GENDER[np.argmax(values)]

def get_gender_cat(genderstr):
    return GENDER.index(genderstr)

def get_age(value):
    return int(value)

def get_ethnicity(values):
    values = values.tolist()
    return ETHNICITY[np.argmax(values)]

def get_emotion(values):
    values = values.tolist()
    return EMOTIONS[np.argmax(values)]

def write_str(annImage, text, p, color=(0, 255, 0), strcolor=(255, 255, 255)):
    FONT = cv.FONT_HERSHEY_SIMPLEX
    SCALE = 0.5
    THICKNESS = 2
    siz = 0,0
    for i, line in enumerate(text.split('\n')):
        tmp_siz = cv.getTextSize(line, FONT, SCALE, THICKNESS)[0]
        siz = cv.getTextSize(line, FONT, SCALE, THICKNESS)[0] if siz[0] < tmp_siz[0] else siz
    # p1 = (p[0]-1, p[1]-siz[1]-10)
    p1 = (p[0]-1, p[1])
    if p1[1] < -5:
        p1 = (p1[0], p[1])
    p2 = (p1[0] + 10 + siz[0], p1[1]+10+siz[1])
    p_text = (p1[0]+5, p2[1]-5)
    y_offset = siz[1]+5
    # cv.rectangle(annImage, p1, (p2[0], p1[1]+3*25), color, cv.FILLED)
    # cv.rectangle(annImage, p1, p2, color, cv.FILLED)
    # cv.putText(annImage, text, p_text, FONT, SCALE, (255, 255, 255), THICKNESS)

    # for i, line in enumerate(text.split('\n')):
    #     p_text_flow = (p_text[0], p_text[1] + i*y_offset)
    #     cv.putText(annImage, line, p_text_flow, FONT, SCALE, strcolor, THICKNESS)

    
    from PIL import ImageFont, ImageDraw, Image
    annImage_array = Image.fromarray(annImage)
    draw = ImageDraw.Draw(annImage_array)
    fontpath = "script/apple-symbols-1.ttf"  
    font = ImageFont.truetype(fontpath, 25)
    size = (0,0)
    for i, line in enumerate(text.split('\n')):
        linesize = font.getsize(line)
        size = linesize if linesize[0] > size[0] else size
    draw.rectangle([p1, (p1[0]+size[0]+10, p1[1]+(i+2)*size[1]+5)], fill=color)
    draw.text((p1[0]+5, p1[1]+5), text, font=font, fill=strcolor)
    return np.array(annImage_array)

def top_right(f):
    return (f['roi'][0]+f['roi'][2], f['roi'][1])

def top_left(f):
    return (f['roi'][0], f['roi'][1])

def bottom_right(f):
    return (f['roi'][0]+f['roi'][2], f['roi'][1]+f['roi'][3])

def bottom_left(f):
    return (f['roi'][0], f['roi'][1]+f['roi'][3])

def coords(f):
    startX, startY = top_left(f)
    endX, endY = bottom_right(f)
    return startX, startY, endX, endY