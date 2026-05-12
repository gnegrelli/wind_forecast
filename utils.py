import torch.nn as nn


def calculate_prediction_error(model, samples, targets):
    criterion = nn.L1Loss(reduction='none')
    predictions = model.predict(samples)
    loss = criterion(predictions, targets)

    return loss


def calculate_per_sample_error(model, samples, targets):
    loss = calculate_prediction_error(model, samples, targets)

    return loss.mean(dim=1)


def calculate_per_timestep_error(model, samples, targets):
    loss = calculate_prediction_error(model, samples, targets)

    return loss.mean(dim=0)
