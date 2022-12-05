import brainpy.dyn as dyn
from brainpy.connect import TwoEndConnector
from typing import Union

from .base import BaseSynapse, BaseNeuron, BaseNeuronSlice


class Exponential(BaseSynapse):

  def __init__(self,
               pre: Union[BaseNeuron, BaseNeuronSlice],
               post: Union[BaseNeuron, BaseNeuronSlice],
               conn: TwoEndConnector,
               g_max=1.0,
               tau=8.0,
               delay_step=None):
    kwargs = dict(g_max=g_max, tau=tau, delay_step=delay_step)
    super().__init__(pre, post, conn, **kwargs)
    self.model_class = dyn.synapses.Exponential
    # self.model_class_remote = bpl.synapses.RemoteExponential
