import sys
import brainpy.dyn as dyn
from brainpy.modes import Mode, normal
from brainpy.types import Shape, Array
from brainpy.connect import TwoEndConnector
from typing import Union, Sequence, Callable, Tuple, Dict
import jax.tree_util
from .res_manager import ResManager
import bpl
import copy

try:
  from mpi4py import MPI

  mpi_size = MPI.COMM_WORLD.Get_size()
  mpi_rank = MPI.COMM_WORLD.Get_rank()
except ImportError:
  mpi_size = 1
  mpi_rank = 0

pop_id = 0


def get_pop_id():
  global pop_id
  pop_id += 1
  return pop_id


def reset():
  global pop_id
  ResManager.clear()
  pop_id = 0


class BaseNeuron:
  proxy_neurons = {}

  def __init__(
      self,
      shape: Shape,
      *args,
      **kwargs
  ):
    self.args = args
    self.kwargs = kwargs
    self.shape = shape
    # self.model_class = None
    self.lowref = None
    # process id
    self.pid = None
    # population id, autoincrement
    self.id = get_pop_id()
    self.neuron_start_id = 0
    ResManager.pops.append(self)
    ResManager.pops_by_id[self.id] = self

  def __hash__(self):
    return self.id

  def __eq__(self, other):
    return self.id == other.id

  def __getitem__(self, index: Union[slice, Sequence, Array]):
    return BaseNeuronSlice(self, index)

  def __getattr__(self, __name: str):
    return self.lowref.__getattribute__(__name)

  def build(self):  # TODO check current pid
    if self.lowref is not None:
      return self.lowref
    if not hasattr(self, 'model_class'):
      raise Exception("model_class should be assigned")
    self.lowref = self.model_class(self.shape, *self.args, **self.kwargs)
    return self.lowref


class BaseNeuronSlice:
  def __init__(self, target: Union[BaseNeuron, 'BaseNeuronSlice'], index: Union[slice, Sequence, Array]):
    if isinstance(target, BaseNeuron):
      self.target = target
      self.index = [index]
    elif isinstance(target, BaseNeuronSlice):
      self.target = target.target
      self.index = copy.deepcopy(target.index)
      self.index.append(index)
    else:
      raise ValueError("target should be a BaseNeuronSlice or a BaseNeuron")

  def build(self):
    if '_cache' not in self.__dict__:
      self._cache = self.target.build()
      for idx in self.index:
        self._cache = self._cache[idx]
    return self._cache

  def __getattr__(self, __name: str):
    if __name in self.target.__dict__:
      return self.target.__getattribute__(__name)
    else:
      return self.build().__getattribute__(__name)

  def __getitem__(self, index: Union[slice, Sequence, Array]):
    return BaseNeuronSlice(self, index)


class register():
  """decorator class to register user defined neuron model and synapse model to respa and base module

  **Decorator Examples**

    >>> import bpl
    >>> from brainpy.dyn import channels
    >>> @bpl.register()
    >>> class HH(bp.dyn.CondNeuGroup):
    >>>   def __init__(self, size):
    >>>     super(HH, self).__init__(size, )
    >>>     self.INa = channels.INa_TM1991(size, g_max=100., V_sh=-63.)
    >>>     self.IK = channels.IK_TM1991(size, g_max=30., V_sh=-63.)
    >>>     self.IL = channels.IL(size, E=-60., g_max=0.05)
    >>> pop = bpl.HH(100)

  Parameters
  ----------
  name : str = None
    The name of the class in the bpl, respa and base module
    User should use this class to create their model instance.
  model : str = None
    The type of the model
    If model is None the type will be inferred from the class.
  """

  def __init__(self, name: str = None, model: str = None):
    self.name = name
    self.model = model.lower() if model is not None else None

  def __call__(self, cls):
    if self.model is None and issubclass(cls, dyn.NeuGroup) or self.model == 'neuron':
      if self.name is None:
        name = cls.__name__
      else:
        name = self.name
      bases = (BaseNeuron,)
      attrs = {'__qualname__': name, 'model_class': cls}
    elif self.model is None and issubclass(cls, dyn.SynConn) or self.model == 'synapse':
      if self.name is None:
        name = cls.__name__
      else:
        name = self.name
      bases = (BaseSynapse,)
      attrs = {'__qualname__': name,
               'model_class': cls, 'model_class_remote': None}
    else:
      raise RuntimeError(f'custom {self.model} is not supported')
    respa_type = type(name, bases, attrs)
    name_split = __name__.split('.')
    if len(name_split) > 1:
      setattr(sys.modules['.'.join(name_split[:-1])], name, respa_type)
    if len(name_split) > 2:
      setattr(sys.modules['.'.join(name_split[:-2])], name, respa_type)
    setattr(sys.modules[__name__], name, respa_type)
    return cls


class BaseSynapse:

  def __init__(
      self,
      pre: Union[BaseNeuron, Tuple[BaseNeuron, Union[slice, Sequence, Array]]],
      post: Union[BaseNeuron, Tuple[BaseNeuron, Union[slice, Sequence, Array]]],
      conn: Union[TwoEndConnector, Array, Dict[str, Array]],
      *args,
      **kwargs
  ):
    self.pre = pre
    self.post = post
    self.conn = conn
    self.args = args
    self.kwargs = kwargs
    # self.model_class = None
    self.lowref = None
    ResManager.syns.append(self)

  def __getattr__(self, __name: str):
    return self.lowref.__getattribute__(__name)

  def build(self):
    # assert self.pre.lowref is not None and self.post.lowref is not None
    if self.pre.pid != mpi_rank and self.post.pid != mpi_rank or self.lowref is not None:
      return self.lowref

    if not hasattr(self, 'model_class'):
      raise Exception("model_class should be assigned")

    pre_pid = self.pre.pid
    pre_shape = self.pre.shape
    if isinstance(self.pre, BaseNeuronSlice):
      pre_slice = self.pre.index
      if pre_pid == mpi_rank:
        pre = self.pre.build()
      else:
        pre = self.pre.target
    elif isinstance(self.pre, BaseNeuron):
      pre_slice = None
      if pre_pid == mpi_rank:
        pre = self.pre.build()
      else:
        pre = self.pre
    else:
      raise ValueError(type(self.pre))

    post_pid = self.post.pid
    post_shape = self.post.shape
    if isinstance(self.post, BaseNeuronSlice):
      post_slice = self.post.index
      if post_pid == mpi_rank:
        post = self.post.build()
      else:
        post = self.post.target
    elif isinstance(self.post, BaseNeuron):
      post_slice = None
      if post_pid == mpi_rank:
        post = self.post.build()
      else:
        post = self.post
    else:
      raise ValueError(type(self.post))

    if pre_pid == post_pid and pre_pid == mpi_rank:
      self.lowref = self.model_class(pre, post, self.conn, *self.args, **self.kwargs)
    elif pre_pid == mpi_rank:
      if post not in BaseNeuron.proxy_neurons:
        tmp_ = bpl.neurons.ProxyLIF(post_shape)
        BaseNeuron.proxy_neurons.update({post: tmp_})
      else:
        tmp_ = BaseNeuron.proxy_neurons[post]
      if post_slice is not None:
        for sli in post_slice:
          tmp_ = tmp_[sli]
      self.lowref = self.model_class_remote(pre_pid, pre, post_pid, tmp_, conn=self.conn, *self.args, **self.kwargs)
    elif post_pid == mpi_rank:
      if pre not in BaseNeuron.proxy_neurons:
        tmp_ = bpl.neurons.ProxyLIF(pre_shape)
        BaseNeuron.proxy_neurons.update({pre: tmp_})
      else:
        tmp_ = BaseNeuron.proxy_neurons[pre]
      if pre_slice is not None:
        for sli in pre_slice:
          tmp_ = tmp_[sli]
      self.lowref = self.model_class_remote(pre_pid, tmp_, post_pid, post, conn=self.conn, *self.args, **self.kwargs)
    return self.lowref


class LIF(BaseNeuron):
  def __init__(
      self,
      shape: Shape,
      *args,
      **kwargs
  ):
    super(LIF, self).__init__(shape, *args, **kwargs)
    self.model_class = dyn.LIF


# another way to define respa LIF
# BaseNeuron.register(dyn.LIF)


class Exponential(BaseSynapse):
  def __init__(
      self,
      pre: Union[BaseNeuron, Tuple[BaseNeuron, Union[slice, Sequence, Array]]],
      post: Union[BaseNeuron, Tuple[BaseNeuron, Union[slice, Sequence, Array]]],
      conn: Union[TwoEndConnector, Array, Dict[str, Array]],
      *args,
      **kwargs
  ):
    super().__init__(pre, post, conn, *args, **kwargs)
    self.model_class = dyn.synapses.Exponential
    self.model_class_remote = bpl.synapses.RemoteExponential


class Network:
  def __init__(self, *ds_tuple, name: str = None, mode: Mode = normal, **ds_dict):
    self.ds_tuple = ds_tuple
    self.ds_dict = ds_dict
    self.lowref = bpl.RemoteNetwork((), name=name, mode=mode)

  def __getattr__(self, __name: str):
    return self.lowref.__getattribute__(__name)

  def build_all_population_synapse(self):
    for pop in ResManager.pops:
      pop.build()
    for syn in ResManager.syns:
      syn.build()
    self.lowref.register_implicit_nodes(*map(lambda x: x.lowref, ResManager.pops))
    self.lowref.register_implicit_nodes(*map(lambda x: x.lowref, ResManager.syns))

  def build(self):
    self.pops_ = []
    self.syns_ = []

    def reg_pop_syn(v):
      if isinstance(v, BaseNeuron):
        self.pops_.append(v)
      elif isinstance(v, BaseSynapse):
        self.syns_.append(v)

    jax.tree_util.tree_map(reg_pop_syn, self.__dict__)

    def simple_split(pops_):
      res = [[] for i in range(mpi_size)]
      avg = len(pops_) // mpi_size
      for i in range(mpi_size):
        res[i].extend(pops_[i * avg:i * avg + avg])
      res[-1].extend(pops_[avg * mpi_size:])
      return res

    self.pops_by_rank = simple_split(self.pops_)
    offset = 1
    for i in range(mpi_size):
      for __pops in self.pops_by_rank[i]:
        __pops.pid = i
    for node in self.pops_by_rank[mpi_rank]:
      node.neuron_start_id = offset
      offset += node.shape

      node.build()
      self.lowref.register_implicit_nodes(node.lowref)
    for node in self.syns_:
      if node.build() is not None:
        self.lowref.register_implicit_nodes(node.lowref)

  def update(self, *args, **kwargs):
    self.lowref.update(*args, **kwargs)

  def register_nodes(self, *nodes, **named_nodes):
    self.lowref.register_implicit_nodes(*nodes, **named_nodes)

  def register_vars(self, *variables, **named_variables):
    self.lowref.register_implicit_vars(*variables, **named_variables)


class DSRunner:

  def __init__(
      self,
      target: Union[dyn.DynamicalSystem, Network],
      # inputs for target variables
      inputs: Sequence = (),
      fun_inputs: Callable = None,
      # extra info
      dt: float = None,
      t0: Union[float, int] = 0.,
      spike_callback: Callable = None,
      volt_callback: Callable = None,
      **kwargs
  ):
    if not isinstance(target, (Network, dyn.DynamicalSystem)):
      raise ValueError(type(target))

    if isinstance(target, Network):
      target = target.lowref

    def _callback(t: float, d: dict):
      for k, v in d.items():
        if k == 'spike' and spike_callback:
          tmp = ''
          for i, j in enumerate(v):
            if j is True:
              tmp += '{},{:.2f}\n'.format(i + 1, t)
          spike_callback(tmp)
        if k == 'V' and volt_callback:
          tmp = ''
          for i, j in enumerate(v):
            tmp += '{},{:.2f},{:.2f}\n'.format(i + 1, t, j)
          volt_callback(tmp)

    c = _callback if spike_callback or volt_callback else None

    self.lowref = bpl.BplRunner(target=target, inputs=inputs, fun_inputs=fun_inputs, dt=dt, t0=t0, callback=c, **kwargs)

  def __getattr__(self, __name: str):
    return self.lowref.__getattribute__(__name)
