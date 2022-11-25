import numpy as np
from brainpy.connect.random_conn import FixedProb
from typing import List, Tuple

from .res_manager import ResManager
from .base import mpi_size


class Optimizer():
  OPT_GREEDY_INIT = 1

  def __init__(self, opt_method=OPT_GREEDY_INIT, device_memory=40, device_capability=60) -> None:
    self.opt_method = opt_method
    self.device_memory = device_memory
    self.device_capability = device_capability

  def run(self):
    if self.opt_method == self.OPT_GREEDY_INIT:
      self.run_greedy_init()

  def get_edge_weight_matrix(self, syns=[], total_pop_num=0) -> List[List[int]]:
    """get edge weight matrix, traffic matrix between populations.

    Parameteresult
    ----------
    syns : List
        synapse List, `ResManager.syns`
    total_pop_num : int
        population total number

    Returns
    -------
    List
        two dimensional np array
    """
    edge_weight_matrix = np.zeros((total_pop_num, total_pop_num))
    for syn in syns:
      pre_id = syn.pre.id
      pre_num = syn.pre.shape
      post_id = syn.post.id
      post_num = syn.post.shape

      edge_weight = 0
      conn = syn.conn
      if isinstance(conn, FixedProb):
        edge_weight = pre_num * post_num * conn.prob

      last_edge_weight = edge_weight_matrix[pre_id - 1][post_id - 1]
      edge_weight_matrix[pre_id - 1][post_id - 1] = last_edge_weight + edge_weight
    return edge_weight_matrix

  def prepare_input(self) -> Tuple[List[List[int]], int, int]:
    """ prepare optimizer input

    Returns
    -------
    tuple(List[[int]], int, int)
        tuple(edge_weight_matrix, memory_used, memory_capacity)
    """

    pops = ResManager.pops
    total_num = len(pops)

    pop_neuron_count = np.zeros(total_num)
    sum_hardware_store = np.zeros((2, mpi_size))
    sum_hardware_store[0] = sum_hardware_store[0] + self.device_memory
    sum_hardware_store[1] = sum_hardware_store[1] + self.device_capability

    edge_weight_matrix = self.get_edge_weight_matrix(
      ResManager.syns, total_num=total_num)

    memory_used = int((edge_weight_matrix.sum(0) * 10 +
                       pop_neuron_count * 300) / (1024 ** 3))
    memory_capacity = int(sum_hardware_store[0])

    return (edge_weight_matrix, memory_used, memory_capacity)

  def run_greedy_init(self):
    """Greedy initialization algorithm. Each time find the method that can reduce
    the communication cost between devices after putting it in.
    """

    A, memory_used, memory_capacity = self.prepare_input()
    population_num = A.shape[0]
    assert A.shape == (population_num, population_num)
    assert memory_used.shape == (population_num,)
    device_num = len(memory_capacity)
    assert memory_capacity.shape == (device_num,)
    assert np.all(A >= 0)

    capacity_each = memory_used.sum() / device_num
    memory_capacity[np.where(memory_capacity > capacity_each)] = capacity_each * 1.01

    # each population assign to one device
    result = {}
    # memory remaining for each device
    device_available = memory_capacity.copy()

    # move population to decrease traffic cost
    move_gain = np.zeros((device_num, population_num))

    remote_conn = 0
    device_index_pre = 0

    for _ in range(population_num):
      max_index = move_gain.argmax()
      device_index = max_index // population_num
      if device_index_pre != device_index:
        remote_conn = 0
        device_index_pre = device_index
      population_index = max_index % population_num
      if move_gain[device_index, population_index] < 0:
        return result

      r = result.get(device_index, [])
      r.append(population_index)
      result[device_index] = r

      remote_conn_pre = remote_conn
      remote_conn = remote_conn + A[population_index, :].sum() - A[population_index, result.get(device_index)].sum()

      new_capacity = device_available[device_index] - memory_used[population_index] - (
        remote_conn - remote_conn_pre) * 280 / 1024 ** 3

      if new_capacity >= 0:
        device_available[device_index] = new_capacity
        move_gain[device_index] += A[:, population_index] + A[population_index, :]
        move_gain[device_index, memory_used > new_capacity] = -float('inf')
        move_gain[:, population_index] = -float('inf')
      else:
        move_gain[device_index, :] = -float('inf')

        if device_index + 1 > device_num - 1:
          print("resource not enough")
          break
        device_available[device_index + 1] -= memory_used[population_index]
        new_capacity = device_available[device_index + 1]
        remote_conn = 0
        move_gain[device_index + 1] += A[:, population_index] + A[population_index, :]
        move_gain[device_index + 1, memory_used > new_capacity] = -float('inf')
        move_gain[:, population_index] = -float('inf')
        r = result.get(device_index + 1, [])
        r.append(population_index)
        result[device_index + 1] = r

      return result

  def save_to_pops_by_rank(self, result: dict):
    d = {}
    for k, v in result.items():
      tmp = []
      for i in v:
        tmp.append(ResManager.pops_by_id(i))
      d[k] = tmp
    return d
