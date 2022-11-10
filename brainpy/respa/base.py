import brainpy.dyn as dyn
from brainpy.modes import Mode, TrainingMode, BatchingMode, normal
from typing import Dict, Union, Sequence, Callable
import jax.tree_util
from mpi4py import MPI
mpi_size = MPI.COMM_WORLD.Get_size()
mpi_rank = MPI.COMM_WORLD.Get_rank()
class BaseNeuron:
	pops = []
	pop_dist = []
	def __init__(self, shape, *args, **kwargs):
		self.args = args
		self.kwargs = kwargs
		self.shape = shape
		self.model_class = None
		self.lowref = None
		self.pid = None
		self.pops.append(self)
	
	def __getitem__(self, index):
		return (self, index)

	def __getattr__(self, __name: str):
		return self.lowref.__getattribute__(__name)

	def build(self): #TODO check current pid
		if self.lowref is not None:
			return self.lowref
		if not self.model_class:
			raise Exception("model_class should not be None")
		self.lowref = self.model_class(self.shape, *self.args, **self.kwargs)
		return self.lowref

class BaseSynapse:
	syns = []
	def __init__(self, pre, post, *args, **kwargs):
		self.pre = pre
		self.post = post
		self.args = args
		self.kwargs = kwargs
		self.model_class = None
		self.lowref = None
		self.syns.append(self)
	
	def __getattr__(self, __name: str):
		return self.lowref.__getattribute__(__name)

	def build(self):
		# assert self.pre.lowref is not None and self.post.lowref is not None
		if self.lowref is not None:
			return self.lowref
		if not self.model_class:
			raise Exception("model_class should not be None")
		if isinstance(self.pre, tuple):
			pre = self.pre[0].lowref[self.pre[1]]
			pre_pid = self.pre[0].pid
		else:
			pre = self.pre.lowref
			pre_pid = self.pre.pid	
		if isinstance(self.post, tuple):
			post = self.post[0].lowref[self.post[1]]
			post_pid = self.post[0].pid
		else:
			post = self.post.lowref
			post_pid = self.post.pid
		if pre_pid == post_pid and pre_pid == mpi_rank:
			self.lowref = self.model_class(pre, post, *self.args, **self.kwargs)
		elif pre_pid == mpi_rank or post_pid == mpi_rank:
			self.lowref = self.model_class_remote(pre_pid, pre, pre_pid, post, *self.args, **self.kwargs)
		return self.lowref

class LIF(BaseNeuron):
	def __init__(self, shape, *args, **kwargs):
		super(LIF, self).__init__(shape, *args, **kwargs)
		self.model_class = dyn.LIF

class Exponential(BaseSynapse):
	def __init__(self, pre, post, *args, **kwargs):
		super().__init__(pre, post, *args, **kwargs)
		self.model_class = dyn.synapses.Exponential
		self.model_class_remote = None

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
		self.lowref.register_implicit_nodes(*map(lambda x: x.lowref, BaseNeuron.pops))
		self.lowref.register_implicit_nodes(*map(lambda x: x.lowref, BaseSynapse.syns))

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

class DSRunner:
	def __init__(
			self,
			target: Network,
			# inputs for target variables
			inputs: Sequence = (),
			fun_inputs: Callable = None,
			# extra info
			dt: float = None,
			t0: Union[float, int] = 0.,
			**kwargs
	):
		self.lowref = dyn.DSRunner(target=target.lowref, inputs=inputs, fun_inputs=fun_inputs, dt=dt, t0=t0, **kwargs)

	def __getattr__(self, __name: str):
		return self.lowref.__getattribute__(__name)






