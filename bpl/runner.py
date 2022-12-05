import brainpy.dyn as dyn
from typing import Union, Callable, List
import bpl
from .base import Network, input_transform, monitor_transform
from .device import Input, Monitor


class DSRunner:

  def __init__(
      self,
      target: Union[dyn.DynamicalSystem, Network],
      # inputs for target variables
      inputs: List[Input],
      monitors: List[Monitor],
      fun_inputs: Callable = None,
      # extra info
      dt: float = None,
      t0: Union[float, int] = 0.,
      spike_callback: Callable = None,
      volt_callback: Callable = None,
      **kwargs):
    if not isinstance(target, (Network, dyn.DynamicalSystem)):
      raise ValueError(type(target))

    inputs_l = []
    for input in inputs:
      inputs_l.append((input.pops, input.value))
    inputs_l = input_transform(inputs_l)
    monitors_d = {}
    for m in monitors:
      monitors_d.update(monitor_transform(m.pops, m.monitor_key))

    if isinstance(target, Network):
      target = target.lowref

    def _callback(t: float, d: dict):
      for k, v in d.items():
        if k == 'spike' and spike_callback:
          a = []
          for i, j in enumerate(v):
            if bool(j):
              a.append((i + 1, t))
          spike_callback(a)
        if k == 'V' and volt_callback:
          a = []
          for i, j in enumerate(v):
            a.append((i + 1, t, float(j)))
          volt_callback(a)

    c = _callback if spike_callback or volt_callback else None

    self.lowref = bpl.BplRunner(
      target=target, inputs=inputs_l, monitors=monitors_d, fun_inputs=fun_inputs, dt=dt, t0=t0, callback=c, **kwargs)

  def __getattr__(self, __name: str):
    return self.lowref.__getattribute__(__name)
