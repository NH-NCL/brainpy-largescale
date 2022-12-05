from typing import Union, Callable, Optional

import brainpy.math as bm
from brainpy import dyn
from brainpy.initialize import Initializer, parameter
from brainpy.types import Array
from brainpy.dyn.base import TwoEndConn
import numpy as np
from bpl.core.base import DynamicalSystem
from mpi4py import MPI
import jax.numpy as jnp
import platform

if platform.system() != 'Windows':
  import mpi4jax


class RemoteSynapse(DynamicalSystem, dyn.base.TwoEndConn):
  """Remote synapse model in multi-device environment.
  This class is a wrapper which can run the given specific synapse model between 2 ranks.

  Parameters
  -------
  synapse_class: TwoEndConn
      Given specific synapse model class.
  param_dict: dict(python3.7 and follow-up version), OrderedDict
      A dict contains parameters which is needed by synapse_class.
  source_rank: int
      The rank id which pre neuron group is located in.
  target_rank: int
      The rank id which post neuron group is located in.
  comm: mpi4py.MPI.Intracomm
      Communicators object in mpi4py package.
  """

  def __init__(self, synapse_class, param_dict, source_rank, target_rank, comm=MPI.COMM_WORLD):
    # Make sure RemoteSynapse can work correctly.
    if not issubclass(synapse_class, TwoEndConn):
      raise Exception('synapse_class should be a subclass (not a instance of class) of brainpy.dyn.base.TwoEndConn.')
    count = 3
    for k, v in param_dict.items():
      if (count == 3 and k != 'pre') or (count == 2 and k != 'post') or (count == 1 and k != 'conn'):
        raise Exception('First 3 keys in param_dict must be pre post and conn in order.')
      count -= 1
    if count > 0:
      raise Exception(
        'At least 3 items should be given to param_dict and first 3 keys in param_dict must be pre post and conn in order.'
      )

    super(RemoteSynapse, self).__init__(param_dict['pre'], param_dict['post'], param_dict['conn'])

    # Create a new class because some class method will change in the following code.
    # Below code can prevent origin class from being affected.

    class sub_synapse_class(DynamicalSystem, synapse_class):
      pass

    self.synapse_class = sub_synapse_class

    self.comm = comm
    self.source_rank = source_rank
    self.target_rank = target_rank
    self.rank = self.comm.Get_rank()
    self.rank_pair = 'rank' + str(self.source_rank) + 'rank' + str(self.target_rank)
    # Replace some function to save place or make synapse work in muti-device enviornment
    self.synapse_class.register_delay = self.remote_register_delay

    if self.rank == source_rank:
      # Replace some function to save place or make synapse work in muti-device enviornment
      def source_rank_init_weights(*para, **dict):
        return None, None

      self.synapse_class.init_weights = source_rank_init_weights

      def variable_(*param, **dict):
        return None

      def odeint(*param, **dict):
        return None

      # Make sure the same neuron group only deliver its spike one time
      # during one step network simulation between this two ranks
      check_name = param_dict['pre'].name + self.rank_pair
      if check_name not in self.remote_synapse_mark:
        self.remote_synapse_mark.append(check_name)
        # send
        if platform.system() == 'Windows':
          self.comm.send(len(param_dict['pre'].spike), dest=target_rank, tag=0)
          self.comm.Send(param_dict['pre'].spike.to_numpy(), dest=target_rank, tag=1)
        else:
          token = mpi4jax.send(param_dict['pre'].spike.value, dest=target_rank, tag=0, comm=self.comm)
      # initialize
      self.synapse_instance = self.synapse_class(**param_dict)

    elif self.rank == target_rank:
      # Replace some function to save place or make synapse work in muti-device enviornment
      self.synapse_class.get_delay_data = self.remote_get_delay_data
      check_name = param_dict['pre'].name + self.rank_pair
      if check_name not in self.remote_synapse_mark:
        self.remote_synapse_mark.append(check_name)
        # receive
        if platform.system() == 'Windows':
          pre_len = self.comm.recv(source=source_rank, tag=0)
          pre_spike = np.empty(pre_len, dtype=np.bool_)
          self.comm.Recv(pre_spike, source=source_rank, tag=1)
        else:
          pre_spike, token = mpi4jax.recv(param_dict['pre'].spike.value, source=source_rank, tag=0, comm=self.comm)
        # self.pre.spike should catch the spike data sent from source rank.
        param_dict['pre'].spike = bm.Variable(pre_spike)
      # initialize
      self.synapse_instance = self.synapse_class(**param_dict)

  def remote_register_delay(
    self,
    identifier: str,
    delay_step: Optional[Union[int, Array, Callable, Initializer]],
    delay_target: bm.Variable,
    initial_delay_data: Union[Initializer, Callable, Array, float, int, bool] = None,
  ):
    """Register delay variable in multi-device enviornmrnt.
    """
    identifier = identifier + self.rank_pair
    # delay steps
    if delay_step is None:
      delay_type = 'none'
    elif isinstance(delay_step, (int, np.integer, jnp.integer)):
      delay_type = 'homo'
    elif isinstance(delay_step, (bm.ndarray, jnp.ndarray, np.ndarray)):
      if delay_step.size == 1 and delay_step.ndim == 0:
        delay_type = 'homo'
      else:
        delay_type = 'heter'
        delay_step = bm.asarray(delay_step)
    elif callable(delay_step):
      delay_step = parameter(delay_step, delay_target.shape, allow_none=False)
      delay_type = 'heter'
    else:
      raise ValueError(f'Unknown "delay_steps" type {type(delay_step)}, only support '
                       f'integer, array of integers, callable function, brainpy.init.Initializer.')
    if delay_type == 'heter':
      if delay_step.dtype not in [bm.int32, bm.int64]:
        raise ValueError('Only support delay steps of int32, int64. If your '
                         'provide delay time length, please divide the "dt" '
                         'then provide us the number of delay steps.')
      if delay_target.shape[0] != delay_step.shape[0]:
        raise ValueError(f'Shape is mismatched: {delay_target.shape[0]} != {delay_step.shape[0]}')
    if delay_type != 'none':
      max_delay_step = int(bm.max(delay_step))

    # delay target
    if delay_type != 'none':
      if not isinstance(delay_target, bm.Variable):
        raise ValueError(f'"delay_target" must be an instance of Variable, but we got {type(delay_target)}')

    # delay variable
    if delay_type != 'none':
      if identifier not in self.remote_global_delay_data:
        delay = bm.LengthDelay(delay_target, max_delay_step, initial_delay_data)
        self.remote_global_delay_data[identifier] = (delay, delay_target)
        self.local_delay_vars[identifier] = delay
      else:
        delay = self.remote_global_delay_data[identifier][0]
        if delay is None:
          delay = bm.LengthDelay(delay_target, max_delay_step, initial_delay_data)
          self.remote_global_delay_data[identifier] = (delay, delay_target)
          self.local_delay_vars[identifier] = delay
        elif delay.num_delay_step - 1 < max_delay_step:
          self.remote_global_delay_data[identifier][0].reset(delay_target, max_delay_step, initial_delay_data)
    else:
      if identifier not in self.remote_global_delay_data:
        delay = bm.LengthDelay(delay_target, 0)
        self.remote_global_delay_data[identifier] = (delay, delay_target)
        self.local_delay_vars[identifier] = delay
        # Above row of code does not exist in 'register_delay' method.
        # In multi-device environment, spike of source neuron group need be delivered during local_delay_vars being traversed when Network is updating.
    self.register_implicit_nodes(self.local_delay_vars)
    return delay_step

  def remote_get_delay_data(
    self,
    identifier: str,
    delay_step: Optional[Union[int, bm.JaxArray, jnp.DeviceArray]],
    *indices: Union[int, slice, bm.JaxArray, jnp.DeviceArray],
  ):
    """Get delay data according to the provided delay steps in multi-device enviornment.
    """
    identifier = identifier + self.rank_pair
    if delay_step is None:
      if bm.ndim(delay_step) == 0:
        return self.remote_global_delay_data[identifier][0](0, *indices)
      else:
        if len(indices) == 0:
          indices = (jnp.arange(delay_step.size),)
        return self.remote_global_delay_data[identifier][0](0, *indices)

    if identifier in self.remote_global_delay_data:
      if bm.ndim(delay_step) == 0:
        return self.remote_global_delay_data[identifier][0](delay_step, *indices)
      else:
        if len(indices) == 0:
          indices = (jnp.arange(delay_step.size),)
        return self.remote_global_delay_data[identifier][0](delay_step, *indices)

    elif identifier in self.local_delay_vars:
      if bm.ndim(delay_step) == 0:
        return self.local_delay_vars[identifier](delay_step)
      else:
        if len(indices) == 0:
          indices = (jnp.arange(delay_step.size),)
        return self.local_delay_vars[identifier](delay_step, *indices)

    else:
      raise ValueError(f'{identifier} is not defined in delay variables.')

  def update(self, tdi, pre_spike=None):
    if self.rank == self.target_rank:
      self.synapse_instance.update(tdi, pre_spike)
    # pass
