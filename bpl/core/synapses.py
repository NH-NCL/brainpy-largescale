from typing import Union, Dict, Callable, Optional

from jax import vmap
from jax.lax import stop_gradient

import brainpy.math as bm
from brainpy.connect import TwoEndConnector, All2All, One2One
from brainpy import dyn
from brainpy.initialize import Initializer, variable_, parameter
from brainpy.integrators import odeint
from brainpy.modes import Mode, BatchingMode, normal
from brainpy.types import Array
from brainpy.dyn.synouts import CUBA
import numpy as np
from bpl.core.base import RemoteDynamicalSystem
from mpi4py import MPI
import jax.numpy as jnp
import platform

if platform.system() != 'Windows':
  import mpi4jax


class RemoteExponential(RemoteDynamicalSystem, dyn.synapses.Exponential):
  """Exponential decay synapse model in multi-device environment.
  """

  def __init__(
      self,
      source_rank,
      pre: dyn.NeuGroup,
      target_rank,
      post: dyn.NeuGroup,
      conn: Union[TwoEndConnector, Array, Dict[str, Array]],
      comm=MPI.COMM_WORLD,
      output: dyn.SynOut = CUBA(),
      stp: Optional[dyn.SynSTP] = None,
      comp_method: str = 'sparse',
      g_max: Union[float, Array, Initializer, Callable] = 1.,
      delay_step: Union[int, Array, Initializer, Callable] = None,
      tau: Union[float, Array] = 8.0,
      method: str = 'exp_auto',

      # other parameters
      name: str = None,
      mode: Mode = normal,
      stop_spike_gradient: bool = False,

  ):
    super(RemoteExponential, self).__init__(pre=pre,
                                            post=post,
                                            conn=conn,
                                            output=output,
                                            stp=stp,
                                            name=name,
                                            mode=mode,
                                            )
    # parameters
    self.stop_spike_gradient = stop_spike_gradient
    self.comp_method = comp_method
    self.tau = tau
    if bm.size(self.tau) != 1:
      raise ValueError(f'"tau" must be a scalar or a tensor with size of 1. But we got {self.tau}')

    # # connections and weights
    # self.g_max, self.conn_mask = self.init_weights(g_max, comp_method, sparse_data='csr')

    self.comm = comm
    self.source_rank = source_rank
    self.target_rank = target_rank
    self.rank = self.comm.Get_rank()
    self.rank_pair = 'rank' + str(self.source_rank) + 'rank' + str(self.target_rank)
    if self.rank == source_rank:
      # Make sure the same neuron group only deliver its spike one time
      # during one step network simulation between this two ranks
      if self.pre.name + self.rank_pair not in self.remote_synapse_mark:
        self.remote_synapse_mark.append(self.pre.name + self.rank_pair)
        if platform.system() == 'Windows':
          self.comm.send(len(self.pre.spike), dest=target_rank, tag=0)
          self.comm.Send(self.pre.spike.to_numpy(), dest=target_rank, tag=1)
        else:
          token = mpi4jax.send(self.pre.spike.value, dest=target_rank, tag=0, comm=self.comm)
        self.delay_step = self.remote_register_delay(
          f"{self.pre.name+self.rank_pair}.spike", delay_step, self.pre.spike)
    elif self.rank == target_rank:
      # connections and weights
      self.g_max, self.conn_mask = self.init_weights(g_max, comp_method, sparse_data='csr')

      if self.pre.name + self.rank_pair not in self.remote_synapse_mark:
        self.remote_synapse_mark.append(self.pre.name + self.rank_pair)
        if platform.system() == 'Windows':
          pre_len = self.comm.recv(source=source_rank, tag=0)
          pre_spike = np.empty(pre_len, dtype=np.bool_)
          self.comm.Recv(pre_spike, source=source_rank, tag=1)
        else:
          pre_spike, token = mpi4jax.recv(pre.spike, source=source_rank, tag=0, comm=self.comm)
        self.pre_spike = bm.Variable(pre_spike)
        # variables
        self.delay_step = self.remote_register_delay(
          f"{self.pre.name+self.rank_pair}.spike", delay_step, self.pre_spike)
      # variables
      self.g = variable_(bm.zeros, self.post.num, mode)
      self.integral = odeint(lambda g, t: -g / self.tau, method=method)

  def remote_register_delay(
      self,
      identifier: str,
      delay_step: Optional[Union[int, Array, Callable, Initializer]],
      delay_target: bm.Variable,
      initial_delay_data: Union[Initializer, Callable, Array, float, int, bool] = None,
  ):
    """Register delay variable in multi-device enviornmrnt.
    """
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

  def update(self, tdi, pre_spike=None):
    if self.rank == self.target_rank:
      t, dt = tdi['t'], tdi['dt']

      # delays
      if pre_spike is None:
        pre_spike = self.remote_get_delay_data(f"{self.pre.name+self.rank_pair}.spike", self.delay_step)
      if self.stop_spike_gradient:
        pre_spike = pre_spike.value if isinstance(pre_spike, bm.JaxArray) else pre_spike
        pre_spike = stop_gradient(pre_spike)

      # update sub-components
      self.output.update(tdi)
      if self.stp is not None:
        self.stp.update(tdi, pre_spike)

      # post values
      if isinstance(self.conn, All2All):
        syn_value = bm.asarray(pre_spike, dtype=bm.dftype())
        if self.stp is not None:
          syn_value = self.stp(syn_value)
        post_vs = self.syn2post_with_all2all(syn_value, self.g_max)
      elif isinstance(self.conn, One2One):
        syn_value = bm.asarray(pre_spike, dtype=bm.dftype())
        if self.stp is not None:
          syn_value = self.stp(syn_value)
        post_vs = self.syn2post_with_one2one(syn_value, self.g_max)
      else:
        if self.comp_method == 'sparse':
          def f(s):
            return bm.pre2post_event_sum(s, self.conn_mask, self.post.num, self.g_max)
          if isinstance(self.mode, BatchingMode):
            f = vmap(f)
          post_vs = f(pre_spike)
          # if not isinstance(self.stp, _NullSynSTP):
          #   raise NotImplementedError()
        else:
          syn_value = bm.asarray(pre_spike, dtype=bm.dftype())
          if self.stp is not None:
            syn_value = self.stp(syn_value)
          post_vs = self.syn2post_with_dense(syn_value, self.g_max, self.conn_mask)
      # updates
      self.g.value = self.integral(self.g.value, t, dt) + post_vs
      # output
      return self.output(self.g)

  def remote_get_delay_data(
      self,
      identifier: str,
      delay_step: Optional[Union[int, bm.JaxArray, jnp.DeviceArray]],
      *indices: Union[int, slice, bm.JaxArray, jnp.DeviceArray],
  ):
    """Get delay data according to the provided delay steps in multi-device enviornment.
    """
    if delay_step is None:
      if bm.ndim(delay_step) == 0:
        return self.remote_global_delay_data[identifier][0](0, *indices)
      else:
        if len(indices) == 0:
          indices = (jnp.arange(delay_step.size),)
        return self.remote_global_delay_data[identifier][0](0, *indices)

    if identifier in self.global_delay_data:
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
