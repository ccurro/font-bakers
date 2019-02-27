import torch


def model_saver(model, path):
    '''
    NOTE: WIP/TODO

    Takes in a generator or descriminator model and saves the model to an
    appropriate location for future generation/discrimination

    Parameters
    ----------
    model: a trained pytorch model
    path: path to where to save the model

    Returns
    -------
    True if it worked false otherwise, possibly something else to be decided
    later
    '''
    torch.save(model.state_dict(), location)
    return true


def model_loader(path):
    '''
    NOTE: WIP/TODO

    Model loader. The point is to be able to load a previously trained model for
    either continuing training or evaluating the generator model.

    Parameters
    ----------
    path: path to the model file we wish to open and load

    Returns
    -------
    pytorch model

    '''

    pass
