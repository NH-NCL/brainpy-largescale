import sys
import brainpy.dyn as dyn
from brainpy.modes import Mode, normal
from brainpy.types import Shape, Array
from typing import Union, Sequence, Callable, Tuple
import jax.tree_util
from brainpy import tools
try:
  from mpi4py import MPI
  mpi_size = MPI.COMM_WORLD.Get_size()
  mpi_rank = MPI.COMM_WORLD.Get_rank()
except:
  mpi_size = 1
  mpi_rank = 0


class BaseNeuron:
  pops = []

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
    self.pid = None
    self.pops.append(self)

  def __getitem__(self, index: Union[slice, Sequence, Array]):
    return (self, index)

  def __getattr__(self, __name: str):
    return self.lowref.__getattribute__(__name)

  def build(self):  # TODO check current pid
    if self.lowref is not None:
      return self.lowref
    if not hasattr(self, 'model_class'):
      raise Exception("model_class should be assigned")
    self.lowref = self.model_class(self.shape, *self.args, **self.kwargs)
    return self.lowref


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
  syns = []

  def __init__(
      self,
      pre: Union[BaseNeuron, Tuple[BaseNeuron, Union[slice, Sequence, Array]]],
      post: Union[BaseNeuron, Tuple[BaseNeuron, Union[slice, Sequence, Array]]],
      *args,
      **kwargs
  ):
    self.pre = pre
    self.post = post
    self.args = args
    self.kwargs = kwargs
    # self.model_class = None
    self.lowref = None
    self.syns.append(self)

  def __getattr__(self, __name: str):
    return self.lowref.__getattribute__(__name)

  def build(self):
    # assert self.pre.lowref is not None and self.post.lowref is not None
    if self.lowref is not None:
      return self.lowref
    if not hasattr(self, 'model_class'):
      raise Exception("model_class should be assigned")

    if isinstance(self.pre, tuple):
      pre_pid = self.pre[0].pid
      pre_shape = self.pre[0].shape
      pre_slice = self.pre[1]
      if pre_pid == mpi_rank:
        pre = self.pre[0].lowref[self.pre[1]]
    elif isinstance(self.pre, BaseNeuron):
      pre_pid = self.pre.pid
      pre_shape = self.pre.shape
      pre_slice = None
      if pre_pid == mpi_rank:
        pre = self.pre.lowref
    else:
      raise ValueError(type(self.pre))

    if isinstance(self.post, tuple):
      post_pid = self.post[0].pid
      post_shape = self.post[0].shape
      post_slice = self.post[1]
      if post_pid == mpi_rank:
        post = self.post[0].lowref[self.post[1]]
    elif isinstance(self.post, BaseNeuron):
      post_pid = self.post.pid
      post_shape = self.post.shape
      post_slice = None
      if post_pid == mpi_rank:
        post = self.post.lowref
    else:
      raise ValueError(type(self.post))

    if pre_pid == post_pid and pre_pid == mpi_rank:
      self.lowref = self.model_class(
          pre, post, *self.args, **self.kwargs)
    elif pre_pid == mpi_rank:
      tmp_ = dyn.LIF(0)
      tmp_.size = post_shape
      tmp_.num = tools.size2num(tmp_.size)
      if post_slice is not None:
        tmp_ = tmp_[post_slice]
      self.lowref = self.model_class_remote(
          pre_pid, pre, post_pid, tmp_, *self.args, **self.kwargs)
    elif post_pid == mpi_rank:
      tmp_ = dyn.LIF(0)
      tmp_.size = pre_shape
      tmp_.num = tools.size2num(tmp_.size)
      if pre_slice is not None:
        tmp_ = tmp_[pre_slice]
      self.lowref = self.model_class_remote(
          pre_pid, tmp_, post_pid, post, *self.args, **self.kwargs)
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
      *args,
      **kwargs
  ):
    super().__init__(pre, post, *args, **kwargs)
    self.model_class = dyn.synapses.Exponential
    self.model_class_remote = None  # dyn.synapses.RemoteExponential


class Network:
  def __init__(self, *ds_tuple, name: str = None, mode: Mode = normal, **ds_dict):
    self.ds_tuple = ds_tuple
    self.ds_dict = ds_dict
    self.lowref = dyn.Network((), name=name, mode=mode)

  def __getattr__(self, __name: str):
    return self.lowref.__getattribute__(__name)

  def build_all_population_synapse(self):
    for pop in BaseNeuron.pops:
      pop.build()
    for syn in BaseSynapse.syns:
      syn.build()
    self.lowref.register_implicit_nodes(
        *map(lambda x: x.lowref, BaseNeuron.pops))
    self.lowref.register_implicit_nodes(
        *map(lambda x: x.lowref, BaseSynapse.syns))

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
        res[i].extend(pops_[i*avg:i*avg+avg])
      res[-1].extend(pops_[avg*mpi_size:])
      return res
    self.pops_split = simple_split(self.pops_)
    for i in range(mpi_size):
      for __pops in self.pops_split[i]:
        __pops.pid = i
    for node in self.pops_split[mpi_rank]:
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
      **kwargs
  ):
    if isinstance(target, Network):
      self.lowref = dyn.DSRunner(
          target=target.lowref, inputs=inputs, fun_inputs=fun_inputs, dt=dt, t0=t0, **kwargs)
    elif isinstance(target, dyn.DynamicalSystem):
      self.lowref = dyn.DSRunner(
          target=target, inputs=inputs, fun_inputs=fun_inputs, dt=dt, t0=t0, **kwargs)
    else:
      raise ValueError(type(target))

  def __getattr__(self, __name: str):
    return self.lowref.__getattribute__(__name)
