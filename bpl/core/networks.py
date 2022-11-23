from typing import Union, Dict, Sequence
import brainpy.math as bm
from brainpy.dyn.base import Network, DynamicalSystem
from brainpy.modes import Mode, normal
from mpi4py import MPI
import mpi4jax


class RemoteNetwork(Network):
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
    if self.comm == None:
      self.rank = None
    else:
      self.rank = self.comm.Get_rank()

  def update_local_delays(self, nodes: Union[Sequence, Dict] = None):
    """overwrite 'update_local_delays' method in Network.
    """
    # update delays
    if nodes is None:
      nodes = tuple(self.nodes(level=1, include_self=False).subset(DynamicalSystem).unique().values())
    elif isinstance(nodes, DynamicalSystem):
      nodes = (nodes, )
    elif isinstance(nodes, dict):
      nodes = tuple(nodes.values())
    if not isinstance(nodes, (tuple, list)):
      raise ValueError('Please provide nodes as a list/tuple/dict of DynamicalSystem.')
    for node in nodes:
      if hasattr(node, 'comm'):
        for name in node.local_delay_vars:
          if self.rank == node.source_rank:
            token = mpi4jax.send(node.pre.spike.value, dest=node.target_rank, tag=3, comm=self.comm)
          elif self.rank == node.target_rank:
            delay = self.remote_global_delay_data[name][0]
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
      nodes = self.nodes(level=1, include_self=False).subset(DynamicalSystem).unique().values()
    elif isinstance(nodes, dict):
      nodes = nodes.values()
    for node in nodes:
      if hasattr(node, 'comm'):
        for name in node.local_delay_vars:
          if self.rank == node.source_rank:
            token = mpi4jax.send(node.pre.spike.value, dest=node.target_rank, tag=4, comm=self.comm)
          elif self.rank == node.target_rank:
            delay = self.remote_global_delay_data[name][0]
            target, token = mpi4jax.recv(node.pre.spike.value, source=node.source_rank, tag=4, comm=self.comm)
            target = bm.Variable(target)
            delay.reset(target.value)
      else:
        for name in node.local_delay_vars:
          delay = self.global_delay_data[name][0]
          target = self.global_delay_data[name][1]
          delay.reset(target.value)