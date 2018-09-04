from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf
from tensorflow.python.keras.layers import InputLayer, Conv2D, Activation, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.optimizers import RMSprop
from tensorflow.python.saved_model.signature_constants import PREDICT_INPUTS

from environment import create_trainer_environment
from keras import optimizers
from keras.models import *
import keras.backend as K

from utils.resnet_helpers import *
if K.backend() == 'tensorflow':
    from utils.BilinearUpSampling import *
from utils.SegDataGenerator import *

HEIGHT = 224
WIDTH = 224
DEPTH = 3
NUM_CLASSES = 22
NUM_DATA_BATCHES = 5
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 10000 * NUM_DATA_BATCHES
BATCH_SIZE = 128


def keras_model_fn(hyperparameters):
    """keras_model_fn receives hyperparameters from the training job and returns a compiled keras model.
    The model will be transformed into a TensorFlow Estimator before training and it will be saved in a
    TensorFlow Serving SavedModel at the end of training.

    Args:
        hyperparameters: The hyperparameters passed to the SageMaker TrainingJob that runs your TensorFlow
                         training script.
    Returns: A compiled Keras model
    """
    # the trainer environment contains useful information about
    env = create_trainer_environment()
    print('creating SageMaker trainer environment:\n%s' % str(env))

    # getting the hyperparameters
    batch_size = env.hyperparameters.get('batch_size', object_type=int)
    data_augmentation = env.hyperparameters.get('data_augmentation', default=True, object_type=bool)
    learning_rate = env.hyperparameters.get('learning_rate', default=.0001, object_type=float)
    width_shift_range = env.hyperparameters.get('width_shift_range', object_type=float)
    height_shift_range = env.hyperparameters.get('height_shift_range', object_type=float)
    EPOCHS = env.hyperparameters.get('epochs', default=10, object_type=int)

    weight_decay = 0.
    batch_momentum = 0.9
    batch_shape = x_train.shape
    classes = 22
    img_input = Input(batch_shape=batch_shape)
    image_size = batch_shape[2:4]
    print('batch_shape' + str(batch_shape))
    print('image_size' + str(image_size))
    bn_axis = 1  # TODO(Shun): Check '3' is ok or not. Documentation recommends '1'

    x = Conv2D(64, (7, 7), strides=(2, 2), padding='same', name='conv1', kernel_regularizer=l2(weight_decay))(img_input)
    x = BatchNormalization(axis=bn_axis, name='bn_conv1', momentum=batch_momentum)(x)
    x = Activation('relu')(x)
    # x = MaxPooling2D((3, 3), strides=(2, 2))(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)  # TODO(Shun): (2, 2) is just random. Further consideration needed

    x = conv_block(3, [64, 64, 256], stage=2, block='a', weight_decay=weight_decay, strides=(1, 1), batch_momentum=batch_momentum)(x)
    x = identity_block(3, [64, 64, 256], stage=2, block='b', weight_decay=weight_decay, batch_momentum=batch_momentum)(x)
    x = identity_block(3, [64, 64, 256], stage=2, block='c', weight_decay=weight_decay, batch_momentum=batch_momentum)(x)

    x = conv_block(3, [128, 128, 512], stage=3, block='a', weight_decay=weight_decay, batch_momentum=batch_momentum)(x)
    x = identity_block(3, [128, 128, 512], stage=3, block='b', weight_decay=weight_decay, batch_momentum=batch_momentum)(x)
    x = identity_block(3, [128, 128, 512], stage=3, block='c', weight_decay=weight_decay, batch_momentum=batch_momentum)(x)
    x = identity_block(3, [128, 128, 512], stage=3, block='d', weight_decay=weight_decay, batch_momentum=batch_momentum)(x)

    x = conv_block(3, [256, 256, 1024], stage=4, block='a', weight_decay=weight_decay, batch_momentum=batch_momentum)(x)
    x = identity_block(3, [256, 256, 1024], stage=4, block='b', weight_decay=weight_decay, batch_momentum=batch_momentum)(x)
    x = identity_block(3, [256, 256, 1024], stage=4, block='c', weight_decay=weight_decay, batch_momentum=batch_momentum)(x)
    x = identity_block(3, [256, 256, 1024], stage=4, block='d', weight_decay=weight_decay, batch_momentum=batch_momentum)(x)
    x = identity_block(3, [256, 256, 1024], stage=4, block='e', weight_decay=weight_decay, batch_momentum=batch_momentum)(x)
    x = identity_block(3, [256, 256, 1024], stage=4, block='f', weight_decay=weight_decay, batch_momentum=batch_momentum)(x)

    x = atrous_conv_block(3, [512, 512, 2048], stage=5, block='a', weight_decay=weight_decay, atrous_rate=(2, 2), batch_momentum=batch_momentum)(x)
    x = atrous_identity_block(3, [512, 512, 2048], stage=5, block='b', weight_decay=weight_decay, atrous_rate=(2, 2), batch_momentum=batch_momentum)(x)
    x = atrous_identity_block(3, [512, 512, 2048], stage=5, block='c', weight_decay=weight_decay, atrous_rate=(2, 2), batch_momentum=batch_momentum)(x)
    # classifying layer
    # x = Conv2D(classes, (3, 3), dilation_rate=(2, 2), kernel_initializer='normal', activation='linear', padding='same', strides=(1, 1), kernel_regularizer=l2(weight_decay))(x)
    x = Conv2D(classes, (1, 1), kernel_initializer='he_normal', activation='linear', padding='same', strides=(1, 1), kernel_regularizer=l2(weight_decay))(x)

    # TODO(Warning): UpSampling2D of Keras cannot accurately represent BilinearUpSampling2D of Tensorflow
    if K.backend() == 'tensorflow':
        x = BilinearUpSampling2D(target_size=tuple(image_size))(x)
    elif K.backend() == 'mxnet':
        # TODO(Warning): This method works only when the input image size is 224 * 224
        x = UpSampling2D(size=(16, 16), data_format=None)(x)
        # x = K.symbol.UpSampling(x, (image_size[0], image_size[1]), sample_type='bilinear')
    # x = K.resize_images(x, image_size[0], image_size[1], 'channels_first')

    _model = Model(img_input, x)

    # initiate RMSprop optimizer
    opt = optimizers.rmsprop(lr=learning_rate, decay=1e-6)

    # Let's train the model using RMSprop
    _model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    return _model


def serving_input_fn(hyperparameters):
    """This function defines the placeholders that will be added to the model during serving.
    The function returns a tf.estimator.export.ServingInputReceiver object, which packages the
    placeholders and the resulting feature Tensors together.
    For more information: https://github.com/aws/sagemaker-python-sdk/blob/master/src/sagemaker/tensorflow/README.rst#creating-a-serving_input_fn

    Args:
        hyperparameters: The hyperparameters passed to SageMaker TrainingJob that runs your TensorFlow
                        training script.
    Returns: ServingInputReceiver or fn that returns a ServingInputReceiver
    """

    # Notice that the input placeholder has the same input shape as the Keras model input
    tensor = tf.placeholder(tf.float32, shape=[None, HEIGHT, WIDTH, DEPTH])

    # The inputs key PREDICT_INPUTS matches the Keras InputLayer name
    inputs = {PREDICT_INPUTS: tensor}
    return tf.estimator.export.ServingInputReceiver(inputs, inputs)


def train_input_fn(training_dir, hyperparameters):
    """Returns input function that would feed the model during training"""
    return _input(tf.estimator.ModeKeys.TRAIN,
                  batch_size=BATCH_SIZE, data_dir=training_dir)


def eval_input_fn(training_dir, hyperparameters):
    """Returns input function that would feed the model during evaluation"""
    return _input(tf.estimator.ModeKeys.EVAL,
                  batch_size=BATCH_SIZE, data_dir=training_dir)


def _input(mode, batch_size, data_dir):
    """Uses the tf.data input pipeline for CIFAR-10 dataset.
    Args:
        mode: Standard names for model modes (tf.estimators.ModeKeys).
        batch_size: The number of samples per batch of input requested.
    """
    dataset = _record_dataset(_filenames(mode, data_dir))

    # For training repeat forever.
    if mode == tf.estimator.ModeKeys.TRAIN:
        dataset = dataset.repeat()

    dataset = dataset.map(_dataset_parser)
    dataset.prefetch(2 * batch_size)

    # For training, preprocess the image and shuffle.
    if mode == tf.estimator.ModeKeys.TRAIN:
        dataset = dataset.map(_train_preprocess_fn)
        dataset.prefetch(2 * batch_size)

        # Ensure that the capacity is sufficiently large to provide good random
        # shuffling.
        buffer_size = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN * 0.4) + 3 * batch_size
        dataset = dataset.shuffle(buffer_size=buffer_size)

    # Subtract off the mean and divide by the variance of the pixels.
    dataset = dataset.map(
        lambda image, label: (tf.image.per_image_standardization(image), label))
    dataset.prefetch(2 * batch_size)

    # Batch results by up to batch_size, and then fetch the tuple from the
    # iterator.
    iterator = dataset.batch(batch_size).make_one_shot_iterator()
    images, labels = iterator.get_next()

    # We must use the default input tensor name PREDICT_INPUTS
    return {PREDICT_INPUTS: images}, labels


def _train_preprocess_fn(image, label):
    """Preprocess a single training image of layout [height, width, depth]."""
    # Resize the image to add four extra pixels on each side.
    image = tf.image.resize_image_with_crop_or_pad(image, HEIGHT + 8, WIDTH + 8)

    # Randomly crop a [HEIGHT, WIDTH] section of the image.
    image = tf.random_crop(image, [HEIGHT, WIDTH, DEPTH])

    # Randomly flip the image horizontally.
    image = tf.image.random_flip_left_right(image)

    return image, label


def _dataset_parser(value):
    """Parse a CIFAR-10 record from value."""
    # Every record consists of a label followed by the image, with a fixed number
    # of bytes for each.
    label_bytes = 1
    image_bytes = HEIGHT * WIDTH * DEPTH
    record_bytes = label_bytes + image_bytes

    # Convert from a string to a vector of uint8 that is record_bytes long.
    raw_record = tf.decode_raw(value, tf.uint8)

    # The first byte represents the label, which we convert from uint8 to int32.
    label = tf.cast(raw_record[0], tf.int32)

    # The remaining bytes after the label represent the image, which we reshape
    # from [depth * height * width] to [depth, height, width].
    depth_major = tf.reshape(raw_record[label_bytes:record_bytes],
                             [DEPTH, HEIGHT, WIDTH])

    # Convert from [depth, height, width] to [height, width, depth], and cast as
    # float32.
    image = tf.cast(tf.transpose(depth_major, [1, 2, 0]), tf.float32)

    return image, tf.one_hot(label, NUM_CLASSES)


def _record_dataset(filenames):
    """Returns an input pipeline Dataset from `filenames`."""
    record_bytes = HEIGHT * WIDTH * DEPTH + 1
    return tf.data.FixedLengthRecordDataset(filenames, record_bytes)


def _filenames(mode, data_dir):
    """Returns a list of filenames based on 'mode'."""
    data_dir = os.path.join(data_dir, 'FileLists')

    assert os.path.exists(data_dir), '"FileLists" directory does not exist.'

    if mode == tf.estimator.ModeKeys.TRAIN:
        return os.path.join(data_dir, 'train.txt')

    elif mode == tf.estimator.ModeKeys.EVAL:
        return os.path.join(data_dir, 'val.txt')
    else:
        raise ValueError('Invalid mode: %s' % mode)
