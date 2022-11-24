from typing import Sequence
from .base import BaseNeuron


def input_transform(pops: Sequence):
  """Select BaseNeurons which locate on current process and return its input.

  Parameters
  ----------
  pops : Sequence

  Returns
  -------
  The inputs for the target DynamicalSystem. It should be the format
    of `[(target, value, [type, operation])]
  """
  if len(pops) > 0 and not isinstance(pops[0], (list, tuple)):
    pops = [pops]
  input_trans = []
  for pop in pops:
    try:
      tmp = pop[0].input
      input_trans.append((tmp, pop[1]) + pop[2:])
    except Exception as e:
      continue
  return input_trans


def monitor_transform(pops: Sequence[BaseNeuron], attr: str = 'spike'):
  """generate monitor parameters dictionary

  Parameters
  ----------
  pops : Sequence[BaseNeuron]

  attr : str, optional
      attribute to monitor, by default 'spike'

  Returns
  -------
  {attribute : BaseNeuron.attribute}
  """
  mon_var = {}
  for pop in pops:
    try:
      tmp = getattr(pop, attr)
      mon_var.update({attr: tmp})
    except Exception as e:
      continue
  return mon_var
