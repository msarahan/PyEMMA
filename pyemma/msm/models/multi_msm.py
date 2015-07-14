__author__ = 'noe'

from pyemma._base.model import Model as _Model

class MultiMSM(_Model):
    r"""A series of MSMs, e.g. for multiple thermodynamic states

    Parameters
    ----------
    models : list of MSM
        list of MSM objects

    """
    def __init__(self, models):
        self.set_model_params(models=models)


    def set_model_params(self, models=None):
        self.models = models

