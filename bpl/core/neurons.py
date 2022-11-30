from typing import Union, Callable, Optional

import brainpy.math as bm
from bpl.core.neurons_base import ProxyNeuGroup
from brainpy.initialize import (ZeroInit, Initializer,
                                parameter, variable_, noise as init_noise)
from brainpy.modes import Mode, NormalMode, TrainingMode, normal, check_mode
from brainpy.tools.checking import check_initializer, check_callable
from brainpy.types import Shape, Array


class ProxyLIF(ProxyNeuGroup):
  r"""Leaky integrate-and-fire neuron model in multi-device enviornment.
  """

  def __init__(
      self,
      size: Shape,
      keep_size: bool = False,

      # other parameter
      V_rest: Union[float, Array, Initializer, Callable] = 0.,
      V_reset: Union[float, Array, Initializer, Callable] = -5.,
      V_th: Union[float, Array, Initializer, Callable] = 20.,
      R: Union[float, Array, Initializer, Callable] = 1.,
      tau: Union[float, Array, Initializer, Callable] = 10.,
      tau_ref: Optional[Union[float, Array, Initializer, Callable]] = None,
      V_initializer: Union[Initializer, Callable, Array] = ZeroInit(),
      noise: Optional[Union[float, Array, Initializer, Callable]] = None,
      method: str = 'exp_auto',
      name: Optional[str] = None,

      # training parameter
      mode: Mode = normal,
      spike_fun: Callable = bm.spike_with_sigmoid_grad,
  ):
    # initialization
    super(ProxyLIF, self).__init__(size=size,
                                   name=name,
                                   keep_size=keep_size,
                                   mode=mode)
    check_mode(self.mode, (TrainingMode, NormalMode), self.__class__)

    # parameters
    self.V_rest = parameter(V_rest, 0, allow_none=False)
    self.V_reset = parameter(V_reset, 0, allow_none=False)
    self.V_th = parameter(V_th, 0, allow_none=False)
    self.tau = parameter(tau, 0, allow_none=False)
    self.R = parameter(R, 0, allow_none=False)
    self.tau_ref = parameter(tau_ref, 0, allow_none=True)
    self.noise = init_noise(noise, 0)
    self.spike_fun = check_callable(spike_fun, 'spike_fun')

    # initializers
    check_initializer(V_initializer, 'V_initializer')
    self._V_initializer = V_initializer

    # variables
    self.V = variable_(self._V_initializer, 0, mode)
    self.input = variable_(bm.zeros, 0, mode)
    # Make sure the 'spike' attribute is consistent with real neuron group
    # It is useful for JIT in multi-device enviornment.
    sp_type = bm.dftype() if isinstance(mode, TrainingMode) else bool  # the gradient of spike is a float
    self.spike = variable_(lambda s: bm.zeros(s, dtype=sp_type), self.varshape, mode)

    if self.tau_ref is not None:
      self.t_last_spike = variable_(lambda s: bm.ones(s) * -1e7, 0, mode)
      self.refractory = variable_(lambda s: bm.zeros(s, dtype=bool), 0, mode)

  def update(self, tdi, x=None):
    pass
