import numpy as np

from sklearn.metrics import mean_squared_error, log_loss
from math import sqrt


def rmse_multi(y_true, y_pred):
    rmse = []
    for i in range(y_pred.shape[1]):
        pred = y_pred[:, i]
        true = y_true[:, i]
        rmse_chunk = sqrt(mean_squared_error(true, pred))
        rmse.append(rmse_chunk)
    return np.mean(rmse)


def log_loss_whales(y_true, y_pred):
    y_pred = np.squeeze(y_pred, axis=1)
    return log_loss(y_true, y_pred, labels=list(range(y_pred.shape[1])))


SCORE_FUNCTIONS = {'localization': rmse_multi,
                   'alignment': rmse_multi,
                   'classification': log_loss_whales,
                   'end_to_end': log_loss_whales
                   }