from typing import Union, Dict, Sequence
import brainpy.math as bm
from brainpy import dyn
from brainpy.modes import Mode, normal
from .base import RemoteDynamicalSystem
from mpi4py import MPI
import platform

if platform.system() != 'Windows':
  import mpi4jax
import numpy as np


class RemoteNetwork(dyn.Network, RemoteDynamicalSystem):
  """Exponential decay synapse model in multi-device environment.
  """

  def __init__(
      self,
      *ds_tuple,
      comm=MPI.COMM_WORLD,
      name: str = None,
      mode: Mode = normal,
      **ds_dict
  ):
    super(RemoteNetwork, self).__init__(*ds_tuple,
                                        name=name,
                                        mode=mode,
                                        **ds_dict)
    self.comm = comm
    if self.comm is None:
      self.rank = None
    else:
      self.rank = self.comm.Get_rank()

  def update_local_delays(self, nodes: Union[Sequence, Dict] = None):
    """overwrite 'update_local_delays' method in Network.
    """
    # update delays
    if nodes is None:
      nodes = tuple(self.nodes(level=1, include_self=False).subset(dyn.DynamicalSystem).unique().values())
    elif isinstance(nodes, dyn.DynamicalSystem):
      nodes = (nodes,)
    elif isinstance(nodes, dict):
      nodes = tuple(nodes.values())
    if not isinstance(nodes, (tuple, list)):
      raise ValueError('Please provide nodes as a list/tuple/dict of dyn.DynamicalSystem.')
    for node in nodes:
      if hasattr(node, 'comm'):
        for name in node.local_delay_vars:
          if self.rank == node.source_rank:
            if platform.system() == 'Windows':
              self.comm.send(len(node.pre.spike), dest=node.target_rank, tag=2)
              self.comm.Send(node.pre.spike.to_numpy(), dest=node.target_rank, tag=3)
            else:
              token = mpi4jax.send(node.pre.spike.value, dest=node.target_rank, tag=3, comm=self.comm)
          elif self.rank == node.target_rank:
            delay = self.remote_global_delay_data[name][0]
            if platform.system() == 'Windows':
              pre_len = self.comm.recv(source=node.source_rank, tag=2)
              target = np.empty(pre_len, dtype=np.bool_)
              self.comm.Recv(target, source=node.source_rank, tag=3)
            else:
              target, token = mpi4jax.recv(node.pre.spike.value, source=node.source_rank, tag=3, comm=self.comm)
            target = bm.Variable(target)
            delay.update(target.value)
      else:
        for name in node.local_delay_vars:
          delay = self.global_delay_data[name][0]
          target = self.global_delay_data[name][1]
          delay.update(target.value)

  def reset_local_delays(self, nodes: Union[Sequence, Dict] = None):
    """overwrite 'reset_local_delays' method in Network.
    """
    # reset delays
    if nodes is None:
      nodes = self.nodes(level=1, include_self=False).subset(dyn.DynamicalSystem).unique().values()
    elif isinstance(nodes, dict):
      nodes = nodes.values()
    for node in nodes:
      if hasattr(node, 'comm'):
        for name in node.local_delay_vars:
          if self.rank == node.source_rank:
            if platform.system() == 'Windows':
              self.comm.send(len(node.pre.spike), dest=node.target_rank, tag=4)
              self.comm.Send(node.pre.spike.to_numpy(), dest=node.target_rank, tag=5)
            else:
              token = mpi4jax.send(node.pre.spike.value, dest=node.target_rank, tag=4, comm=self.comm)
          elif self.rank == node.target_rank:
            delay = self.remote_global_delay_data[name][0]
            if platform.system() == 'Windows':
              pre_len = self.comm.recv(source=node.source_rank, tag=4)
              target = np.empty(pre_len, dtype=np.bool_)
              self.comm.Recv(target, source=node.source_rank, tag=5)
            else:
              target, token = mpi4jax.recv(node.pre.spike.value, source=node.source_rank, tag=4, comm=self.comm)
            target = bm.Variable(target)
            delay.reset(target.value)
      else:
        for name in node.local_delay_vars:
          delay = self.global_delay_data[name][0]
          target = self.global_delay_data[name][1]
          delay.reset(target.value)
