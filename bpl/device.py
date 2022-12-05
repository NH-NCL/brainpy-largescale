from typing import Sequence
from .base import BaseNeuron


class Input():

  def __init__(self, pops: Sequence[BaseNeuron], value: float) -> None:
    self.pops = pops
    self.value = value


class MonitorKey():
  spike = 'spike'
  volt = "V"


class Monitor():

  def __init__(self, pops: Sequence[BaseNeuron], monitor_key: MonitorKey.spike) -> None:
    self.pops = pops
    self.monitor_key = monitor_key
