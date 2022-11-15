from typing import List, Tuple, Union
from .base import BaseNeuron


def input_transform(pops: List[Tuple[BaseNeuron, float]]):
  if len(pops) > 0 and not isinstance(pops[0], (list, tuple)):
    pops = [pops]
  input_trans = []
  for pop in pops:
    try:
      tmp = pop[0].input
      input_trans.append((tmp, pop[1])+pop[2:])
    except Exception as e:
      continue
  return input_trans


def monitor_transform(pops: Union[List[BaseNeuron], Tuple[BaseNeuron]], attr: str = 'spike'):
  mon_var = {}
  for pop in pops:
    try:
      tmp = getattr(pop, attr)
      mon_var.update({attr: tmp})
    except Exception as e:
      continue
  return mon_var
