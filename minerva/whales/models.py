import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from minerva.backend.models.pytorch.callbacks import CallbackList, TrainingMonitor, ValidationMonitor, ModelCheckpoint, \
    NeptuneMonitor, NeptuneMonitorLocalizer, ExperimentTiming, NeptuneMonitorKeypoints, ExponentialLRScheduler, \
    PlotBoundingBoxPredictions
from minerva.backend.models.pytorch.models import MultiOutputModel


class SimpleLocalizer(MultiOutputModel):
    def __init__(self, architecture_config, training_config, callbacks_config):
        super().__init__(architecture_config, training_config, callbacks_config)
        self.model = PyTorchLocalizer(**architecture_config['model_params'])
        self.weight_regularization = weight_regularization_localizer
        self.optimizer = optim.SGD(self.weight_regularization(self.model, **architecture_config['regularizer_params']),
                                   **architecture_config['optimizer_params'])
        self.loss_function = multi_output_cross_entropy
        self.callbacks = build_callbacks_localizer(self.callbacks_config)

    def transform(self, datagen, validation_datagen=None):
        prediction_coordinates = self._transform(datagen, validation_datagen)
        prediction_coordinates_ = np.squeeze(np.argmax(prediction_coordinates, axis=2))
        return {'prediction_coordinates': prediction_coordinates_}


class SimpleAligner(MultiOutputModel):
    def __init__(self, architecture_config, training_config, callbacks_config):
        super().__init__(architecture_config, training_config, callbacks_config)
        self.model = PyTorchAligner(**architecture_config['model_params'])
        self.weight_regularization = weight_regularization_aligner
        self.optimizer = optim.SGD(self.weight_regularization(self.model, **architecture_config['regularizer_params']),
                                   **architecture_config['optimizer_params'])
        self.loss_function = multi_output_cross_entropy
        self.callbacks = build_callbacks_aligner(self.callbacks_config)

    def transform(self, datagen, validation_datagen=None):
        prediction_coordinates = self._transform(datagen, validation_datagen)
        prediction_coordinates_ = np.squeeze(np.argmax(prediction_coordinates, axis=2))
        return {'prediction_coordinates': prediction_coordinates_}


class SimpleClassifier(MultiOutputModel):
    def __init__(self, architecture_config, training_config, callbacks_config):
        super().__init__(architecture_config, training_config, callbacks_config)
        self.model = PyTorchClassifierMultiOutput(**architecture_config['model_params'])
        self.weight_regularization = weight_regularization_classifier
        self.optimizer = optim.SGD(self.weight_regularization(self.model, **architecture_config['regularizer_params']),
                                   **architecture_config['optimizer_params'])
        self.loss_function = multi_output_cross_entropy
        self.callbacks = build_callbacks_classifier(self.callbacks_config)

    def transform(self, datagen, validation_datagen=None):
        prediction_proba = self._transform(datagen, validation_datagen)
        return {'prediction_probability': np.array(prediction_proba)}


class PyTorchLocalizer(nn.Module):
    def __init__(self, input_shape, classes):
        super(PyTorchLocalizer, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2)),

            nn.Conv2d(in_channels=16, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2)),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2)),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2)),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2)),

        )
        self.flat_features = self._get_flat_features(input_shape, self.features)

        self.point1_x = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(self.flat_features, classes),
            nn.LogSoftmax(dim=1)
        )

        self.point1_y = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(self.flat_features, classes),
            nn.LogSoftmax(dim=1)
        )

        self.point2_x = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(self.flat_features, classes),
            nn.LogSoftmax(dim=1)
        )

        self.point2_y = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(self.flat_features, classes),
            nn.LogSoftmax(dim=1)
        )

    def _get_flat_features(self, in_size, features):
        dummy_input = Variable(torch.ones(1, *in_size))
        f = features(dummy_input)
        return int(np.prod(f.size()[1:]))

    def forward(self, x):
        features = self.features(x)
        flat_features = features.view(-1, self.flat_features)
        pred_p1x = self.point1_x(flat_features)
        pred_p1y = self.point1_y(flat_features)
        pred_p2x = self.point2_x(flat_features)
        pred_p2y = self.point2_y(flat_features)
        return [pred_p1x, pred_p1y, pred_p2x, pred_p2y]

    def forward_target(self, x):
        return self.forward(x)


class PyTorchAligner(nn.Module):
    def __init__(self, input_shape, classes):
        super(PyTorchAligner, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2)),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2)),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2)),

            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2)),

            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2)),

            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2)),
        )
        self.flat_features = self._flatten_features(input_shape, self.features)

        self.point1_x = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(self.flat_features, classes['points']),
            nn.LogSoftmax(dim=1)
        )

        self.point1_y = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(self.flat_features, classes['points']),
            nn.LogSoftmax(dim=1)
        )

        self.point2_x = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(self.flat_features, classes['points']),
            nn.LogSoftmax(dim=1)
        )

        self.point2_y = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(self.flat_features, classes['points']),
            nn.LogSoftmax(dim=1)
        )

        self.callosity = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(self.flat_features, classes['callosity']),
            nn.LogSoftmax(dim=1)
        )

        self.whale_id = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(self.flat_features, classes['whale_id']),
            nn.LogSoftmax(dim=1)
        )

    def _flatten_features(self, in_size, features):
        f = features(Variable(torch.ones(1, *in_size)))
        return int(np.prod(f.size()[1:]))

    def forward(self, x):
        features = self.features(x)
        flat_features = features.view(-1, self.flat_features)
        pred_p1x = self.point1_x(flat_features)
        pred_p1y = self.point1_y(flat_features)
        pred_p2x = self.point2_x(flat_features)
        pred_p2y = self.point2_y(flat_features)
        pred_callosity = self.callosity(flat_features)
        pred_whale = self.whale_id(flat_features)

        return [pred_p1x, pred_p1y, pred_p2x, pred_p2y, pred_callosity, pred_whale]

    def forward_target(self, x):
        features = self.features(x)
        flat_features = features.view(-1, self.flat_features)
        pred_p1x = self.point1_x(flat_features)
        pred_p1y = self.point1_y(flat_features)
        pred_p2x = self.point2_x(flat_features)
        pred_p2y = self.point2_y(flat_features)
        return [pred_p1x, pred_p1y, pred_p2x, pred_p2y]


class PyTorchClassifier(nn.Module):
    def __init__(self, input_shape, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2)),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2)),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2)),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2)),

            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2)),

            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2)),
        )
        self.flat_features = self._flatten_features(input_shape, self.features)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(self.flat_features, num_classes),
            nn.LogSoftmax(dim=1)
        )

    def _flatten_features(self, in_size, features):
        f = features(Variable(torch.ones(1, *in_size)))
        return int(np.prod(f.size()[1:]))

    def forward(self, x):
        features = self.features(x)
        flat_features = features.view(-1, self.flat_features)
        out = self.classifier(flat_features)
        return out


class PyTorchClassifierMultiOutput(nn.Module):
    def __init__(self, input_shape, classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2)),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2)),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2)),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2)),

            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2)),

            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2)),
        )
        self.flat_features = self._flatten_features(input_shape, self.features)

        self.whale_id = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(self.flat_features, classes['whale_id']),
            nn.LogSoftmax(dim=1)
        )

        self.callosity = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(self.flat_features, classes['callosity']),
            nn.LogSoftmax(dim=1)
        )

    def _flatten_features(self, in_size, features):
        f = features(Variable(torch.ones(1, *in_size)))
        return int(np.prod(f.size()[1:]))

    def forward(self, x):
        features = self.features(x)
        flat_features = features.view(-1, self.flat_features)
        pred_whale_id = self.whale_id(flat_features)
        pred_callosity = self.callosity(flat_features)
        return [pred_whale_id, pred_callosity]

    def forward_target(self, x):
        features = self.features(x)
        flat_features = features.view(-1, self.flat_features)
        pred_whale_id = self.whale_id(flat_features)
        return [pred_whale_id]


def mse_loss(input_, target):
    return torch.sum((input_ - target) ** 2)


def cross_entropy(output, target):
    return F.nll_loss(output, target)


def multi_output_cross_entropy(outputs, targets):
    loss_seq = []
    for output, target in zip(outputs, targets):
        loss = cross_entropy(output, target)
        loss_seq.append(loss)
    return sum(loss_seq) / len(loss_seq)


def weight_regularization_localizer(model, regularize, weight_decay_conv2d, weight_decay_linear, *args, **kwargs):
    if regularize:
        parameter_list = [{'params': model.features.parameters(), 'weight_decay': weight_decay_conv2d},
                          {'params': model.point1_x.parameters(), 'weight_decay': weight_decay_linear},
                          {'params': model.point1_y.parameters(), 'weight_decay': weight_decay_linear},
                          {'params': model.point2_x.parameters(), 'weight_decay': weight_decay_linear},
                          {'params': model.point2_y.parameters(), 'weight_decay': weight_decay_linear}
                          ]
    else:
        parameter_list = model.parameters()
    return parameter_list


def weight_regularization_aligner(model, regularize, weight_decay_conv2d, weight_decay_linear, *args, **kwargs):
    if regularize:
        parameter_list = [{'params': model.features.parameters(), 'weight_decay': weight_decay_conv2d},
                          {'params': model.point1_x.parameters(), 'weight_decay': weight_decay_linear},
                          {'params': model.point1_y.parameters(), 'weight_decay': weight_decay_linear},
                          {'params': model.point2_x.parameters(), 'weight_decay': weight_decay_linear},
                          {'params': model.point2_y.parameters(), 'weight_decay': weight_decay_linear},
                          {'params': model.callosity.parameters(), 'weight_decay': weight_decay_linear},
                          {'params': model.whale_id.parameters(), 'weight_decay': weight_decay_linear}
                          ]
    else:
        parameter_list = model.parameters()
    return parameter_list


def weight_regularization_classifier(model, regularize, weight_decay_conv2d, weight_decay_linear, *args, **kwargs):
    if regularize:
        parameter_list = [{'params': model.features.parameters(), 'weight_decay': weight_decay_conv2d},
                          {'params': model.whale_id.parameters(), 'weight_decay': weight_decay_linear},
                          {'params': model.callosity.parameters(), 'weight_decay': weight_decay_linear}
                          ]
    else:
        parameter_list = model.parameters()
    return parameter_list


def build_callbacks_localizer(callbacks_config):
    experiment_timing = ExperimentTiming()
    model_checkpoints = ModelCheckpoint(**callbacks_config['model_checkpoint'])
    lr_scheduler = ExponentialLRScheduler(**callbacks_config['lr_scheduler'])
    training_monitor = TrainingMonitor(**callbacks_config['training_monitor'])
    validation_monitor = ValidationMonitor(**callbacks_config['validation_monitor'])
    neptune_monitor = NeptuneMonitorLocalizer(**callbacks_config['neptune_monitor'])
    plot_bounding_box = PlotBoundingBoxPredictions(**callbacks_config['bounding_box_predictions'])

    return CallbackList(
        callbacks=[experiment_timing, model_checkpoints, lr_scheduler, training_monitor, validation_monitor,
                   neptune_monitor, plot_bounding_box])


def build_callbacks_aligner(callbacks_config):
    experiment_timing = ExperimentTiming()
    model_checkpoints = ModelCheckpoint(**callbacks_config['model_checkpoint'])
    lr_scheduler = ExponentialLRScheduler(**callbacks_config['lr_scheduler'])
    training_monitor = TrainingMonitor(**callbacks_config['training_monitor'])
    validation_monitor = ValidationMonitor(**callbacks_config['validation_monitor'])
    neptune_monitor = NeptuneMonitorKeypoints(**callbacks_config['neptune_monitor'])

    return CallbackList(
        callbacks=[experiment_timing, model_checkpoints, lr_scheduler, training_monitor, validation_monitor,
                   neptune_monitor])


def build_callbacks_classifier(callbacks_config):
    experiment_timing = ExperimentTiming()
    model_checkpoints = ModelCheckpoint(**callbacks_config['model_checkpoint'])
    lr_scheduler = ExponentialLRScheduler(**callbacks_config['lr_scheduler'])
    validation_monitor = ValidationMonitor(**callbacks_config['validation_monitor'])
    training_monitor = TrainingMonitor(**callbacks_config['training_monitor'])
    neptune_monitor = NeptuneMonitor()

    return CallbackList(
        callbacks=[experiment_timing, model_checkpoints, lr_scheduler, training_monitor, validation_monitor,
                   neptune_monitor])
