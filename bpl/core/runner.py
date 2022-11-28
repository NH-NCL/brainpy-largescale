from typing import Dict, Union, Sequence, Callable
from jax.tree_util import tree_map
from jax.experimental.host_callback import id_tap

from brainpy import math as bm
from brainpy import dyn
from brainpy.tools.checking import serialize_kwargs
from brainpy.tools.others.dicts import DotDict


class BplRunner(dyn.DSRunner):
  def __init__(
      self,
      target: dyn.DynamicalSystem,

      # inputs for target variables
      inputs: Sequence = (),
      fun_inputs: Callable = None,

      # extra info
      dt: float = None,
      t0: Union[float, int] = 0.,
      callback: Callable = None,
      **kwargs
  ):
    super(BplRunner, self).__init__(target=target, inputs=inputs, fun_inputs=fun_inputs, dt=dt, t0=t0, **kwargs)
    self.callback = callback

  def f_predict(self, shared_args: Dict = None):
    if shared_args is None:
      shared_args = dict()

    shared_kwargs_str = serialize_kwargs(shared_args)
    if shared_kwargs_str not in self._f_predict_compiled:

      monitor_func = self.build_monitors(self._mon_info[0], self._mon_info[1], shared_args)

      def _step_func(t, i, x):
        self.target.clear_input()
        # input step
        shared = DotDict(t=t, i=i, dt=self.dt)
        self._input_step(shared)
        # dynamics update step
        shared.update(shared_args)
        args = (shared,) if x is None else (shared, x)
        out = self.target(*args)
        # monitor step
        mon = monitor_func(shared)
        if self.callback:
          self.callback(t, mon)
        # finally
        if self.progress_bar:
          id_tap(lambda *arg: self._pbar.update(), ())
        return out, mon

      if self.jit['predict']:
        dyn_vars = self.target.vars()
        dyn_vars.update(self.dyn_vars)

        def run_func(all_inputs):
          return bm.for_loop(_step_func, dyn_vars.unique(), all_inputs)

      else:
        def run_func(xs):
          # total data
          times, indices, xs = xs

          outputs = []
          monitors = {key: [] for key in (set(self.mon.var_names) | set(self.fun_monitors.keys()))}
          for i in range(times.shape[0]):
            # data at time i
            x = tree_map(lambda x: x[i], xs, is_leaf=lambda x: isinstance(x, bm.JaxArray))

            # step at the i
            output, mon = _step_func(times[i], indices[i], x)

            # append output and monitor
            outputs.append(output)
            for key, value in mon.items():
              monitors[key].append(value)

          # final work
          if outputs[0] is None:
            outputs = None
          else:
            outputs = bm.asarray(outputs)
          for key, value in monitors.items():
            monitors[key] = bm.asarray(value)
          return outputs, monitors
      self._f_predict_compiled[shared_kwargs_str] = run_func
    return self._f_predict_compiled[shared_kwargs_str]
