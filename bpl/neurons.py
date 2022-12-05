from .base import BaseNeuron
import brainpy.dyn as dyn


class LIF(BaseNeuron):

  def __init__(self, size: int, V_reset: float, V_th: float, V_rest: float, tau: float, tau_ref: float):
    kwargs = dict(V_reset=V_reset, V_th=V_th, V_rest=V_rest, tau=tau, tau_ref=tau_ref)
    super(LIF, self).__init__(size, **kwargs)
    self.model_class = dyn.LIF
