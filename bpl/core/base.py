from typing import Union, Dict, List, Tuple
import brainpy.math as bm
from brainpy.base.base import Base


class RemoteDynamicalSystem(Base):
  '''These variable is used in multi-device enviornment.'''
  remote_global_delay_data: Dict[str, Tuple[Union[bm.LengthDelay, None], bm.Variable]] = dict()

  remote_first_send_mark: List[str] = []
