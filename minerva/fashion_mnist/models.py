from keras.callbacks import EarlyStopping
from keras.layers import Flatten, Dense, Dropout, Input, Conv2D, MaxPool2D
from keras.models import Model
from keras.optimizers import SGD

from minerva.backend.models.keras.models import BasicKerasClassifier
from minerva.backend.models.keras.callbacks import NeptuneMonitor


class SimpleClassifier(BasicKerasClassifier):
    def __init__(self, architecture_config, training_config, callbacks_config):
        self.architecture_config = architecture_config
        self.training_config = training_config
        self.callbacks_config = callbacks_config

        self._build_model = build_model
        self._build_optimizer = build_optimizer
        self._build_loss = build_loss

        self.callbacks = self._create_callbacks(**self.callbacks_config)
        self.model = self._compile_model(**self.architecture_config)

    def _create_callbacks(self, patience, model_name, **kwargs):
        early_stopping = EarlyStopping(patience=patience)
        neptune = NeptuneMonitor()
        return [early_stopping, neptune]


def build_model(input_size, classes):
    input_shape = (input_size, input_size, 1)
    images = Input(shape=input_shape)

    x = Conv2D(16, (3, 3), activation='relu', name='block1_conv1')(images)
    x = Conv2D(16, (3, 3), activation='relu', name='block1_conv2')(x)

    x = Conv2D(64, (3, 3), activation='relu', name='block2_conv1')(x)
    x = Conv2D(64, (3, 3), activation='relu', name='block2_conv2')(x)
    x = MaxPool2D(name='block2_maxpool')(x)

    x = Conv2D(128, (3, 3), activation='relu', name='block3_conv1')(x)
    x = Conv2D(128, (3, 3), activation='relu', name='block3_conv2')(x)
    x = MaxPool2D(name='block3_maxpool')(x)

    x = Flatten()(x)
    x = Dense(512, activation='relu', name='fc1')(x)
    x = Dropout(0.25)(x)
    x = Dense(512, activation='relu', name='fc2')(x)
    x = Dropout(0.25)(x)
    predictions = Dense(classes, activation='softmax', name='output')(x)

    model = Model(inputs=images, outputs=predictions)
    return model


def build_optimizer(lr, momentum, nesterov):
    return SGD(lr=lr, momentum=momentum, nesterov=nesterov)


def build_loss():
    return 'categorical_crossentropy'
