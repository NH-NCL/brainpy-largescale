from typing import Union, Dict, List, Tuple
import brainpy.math as bm
from brainpy.base.base import Base
import gc


class RemoteDynamicalSystem(Base):
  '''These variable is used in multi-device enviornment.'''
  remote_global_delay_data: Dict[str, Tuple[Union[bm.LengthDelay, None], bm.Variable]] = dict()

  remote_synapse_mark: List[str] = []

  def __del__(self):
    """Function for handling `del` behavior.

    This function is used to pop out the variables which registered in global delay data and remote_global_delay_data.
    """
    if not hasattr(self, 'comm'):
      if hasattr(self, 'local_delay_vars'):
        for key in tuple(self.local_delay_vars.keys()):
          val = self.remote_global_delay_data.pop(key)
          del val
          val = self.local_delay_vars.pop(key)
          del val
    else:
      if hasattr(self, 'local_delay_vars'):
        for key in tuple(self.local_delay_vars.keys()):
          val = self.global_delay_data.pop(key)
          del val
          val = self.local_delay_vars.pop(key)
          del val
    if hasattr(self, 'implicit_nodes'):
      for key in tuple(self.implicit_nodes.keys()):
        del self.implicit_nodes[key]
    if hasattr(self, 'implicit_vars'):
      for key in tuple(self.implicit_vars.keys()):
        del self.implicit_vars[key]
    for key in tuple(self.__dict__.keys()):
      del self.__dict__[key]
    gc.collect()
