import brainpy.dyn as dyn
from brainpy.modes import Mode, TrainingMode, BatchingMode, normal
from typing import Dict, Union, Sequence, Callable
class BaseNeuron:
	pops = []
	pop_dist = []
	def __init__(self, shape, *args, **kwargs):
		self.args = args
		self.kwargs = kwargs
		self.shape = shape
		self.model_class = None
		self.pid = None
		self.pops.append(self)
  
	def __getitem__(self, index):
		return (self, index)

	def __getattr__(self, __name: str):
		return self.lowref.__getattribute__(__name)

	def build(self): #TODO check current pid
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
		self.syns.append(self)
  
	def __getattr__(self, __name: str):
		return self.lowref.__getattribute__(__name)

	def build(self):
		# assert self.pre.lowref is not None and self.post.lowref is not None
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
		if pre_pid == post_pid and post_pid != None or post_pid is None and pre_pid is None: #TODO check current pid
			self.lowref = self.model_class(pre, post, *self.args, **self.kwargs)
		else:
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
    self.lowref = dyn.Network(*map(lambda x: x.lowref, ds_tuple), name=name, mode=mode, **ds_dict)
  
  def build(self):
    for pop in BaseNeuron.pops:
      pop.build()
    for syn in BaseSynapse.syns:
      syn.build()
    self.lowref.register_implicit_nodes(*map(lambda x: x.lowref, BaseNeuron.pops))
    self.lowref.register_implicit_nodes(*map(lambda x: x.lowref, BaseSynapse.syns))
  
  def update(self, *args, **kwargs):
    self.lowref.update(*args, **kwargs)

class DSRunner:
	def __init__(self):
		pass
	
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
		inputs_trans = inputs
		# for i in inputs:
		# 	inputs_trans.append((i[0].lowref.input, i[1]))
		self.lowref = dyn.DSRunner(target=target.lowref, inputs=inputs_trans, fun_inputs=fun_inputs, dt=dt, t0=t0, **kwargs)

	def __getattr__(self, __name: str):
		return self.lowref.__getattribute__(__name)






