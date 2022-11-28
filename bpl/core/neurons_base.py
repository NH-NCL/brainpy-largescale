from typing import Union, Sequence
import collections
import brainpy.math as bm
from brainpy import dyn
from brainpy.initialize import variable_
from brainpy.modes import Mode, TrainingMode, normal
from brainpy.types import Array, Shape


class ProxyNeuGroup(dyn.NeuGroup):
  """Base class to model neuronal groups in multi-device enviornment.
  """

  def __init__(
      self,
      size: Shape,
      keep_size: bool = False,
      name: str = None,
      mode: Mode = normal,
  ):
    # initialize
    super(ProxyNeuGroup, self).__init__(size=size,
                                        name=name,
                                        keep_size=keep_size,
                                        mode=mode)

  def __getitem__(self, item):
    return ProxyNeuGroupView(target=self, index=item, keep_size=self.keep_size)


class ProxyNeuGroupView(ProxyNeuGroup):
  """A view for a neuron group instance in multi-device enviornment."""

  def __init__(
      self,
      target: ProxyNeuGroup,
      index: Union[slice, Sequence, Array],
      name: str = None,
      mode: Mode = None,
      keep_size: bool = False
  ):
    # check target
    if not isinstance(target, dyn.DynamicalSystem):
      raise TypeError(f'Should be instance of dyn.DynamicalSystem, but we got {type(target)}.')

    # check slicing
    if isinstance(index, (int, slice)):
      index = (index,)
    self.index = index  # the slice

    # check slicing
    var_shapes = target.varshape
    if len(self.index) > len(var_shapes):
      raise ValueError(f"Length of the index should be less than "
                       f"that of the target's varshape. But we "
                       f"got {len(self.index)} > {len(var_shapes)}")

    # get size
    size = []
    for i, idx in enumerate(self.index):
      if isinstance(idx, int):
        size.append(1)
      elif isinstance(idx, slice):
        size.append(dyn.base._slice_to_num(idx, var_shapes[i]))
      else:
        # should be a list/tuple/array of int
        # do not check again
        if not isinstance(idx, collections.Iterable):
          raise TypeError('Should be an iterable object of int.')
        size.append(len(idx))
    size += list(var_shapes[len(self.index):])

    # initialization
    ProxyNeuGroup.__init__(self, tuple(size), keep_size=keep_size, name=name, mode=mode)
    # Proxy neuron needs 'input' attribute during distributed network updating.
    self.input = variable_(bm.zeros, 0, mode)
    # Make sure the 'spike' attribute is consistent with real neuron group
    # It is useful for JIT in multi-device enviornment.
    sp_type = bm.dftype() if isinstance(mode, TrainingMode) else bool  # the gradient of spike is a float
    self.spike = variable_(lambda s: bm.zeros(s, dtype=sp_type), self.varshape, mode)
    # Proxy neuron needs 'V' attribute during distributed network updating.
    self.V = variable_(bm.zeros, 0, mode)
