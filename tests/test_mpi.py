import unittest
import bpl
import brainpy as bp
import brainpy.math as bm
import pytest


class EINet_V1(bpl.RemoteNetwork):
  def __init__(self, scale=1.0, method='exp_auto', delay_step=None):
    super(EINet_V1, self).__init__()

    # network size
    num_exc = int(3200 * scale)
    num_inh = int(800 * scale)

    # neurons
    pars = dict(V_rest=-60., V_th=-50., V_reset=-60., tau=20., tau_ref=5.)

    # synapses
    we = 0.6 / scale  # excitatory synaptic weight (voltage)
    wi = 6.7 / scale  # inhibitory synaptic weight
    if self.rank == 0:
      self.E1 = bp.neurons.LIF(num_exc, **pars, method=method)
      self.I1 = bp.neurons.LIF(num_inh, **pars, method=method)
      self.E12I1 = bp.synapses.Exponential(self.E1, self.I1,
                                           bp.conn.FixedProb(0.02, seed=1),
                                           output=bp.synouts.COBA(E=0.), g_max=we,
                                           tau=5.,
                                           method=method, delay_step=delay_step)
      self.I2 = bpl.neurons.ProxyLIF(num_inh, **pars, method=method)
      self.I3 = bpl.neurons.ProxyLIF(num_inh, **pars, method=method)
      self.I4 = bpl.neurons.ProxyLIF(num_inh, **pars, method=method)
      self.I5 = bpl.neurons.ProxyLIF(num_inh, **pars, method=method)
    elif self.rank == 1:
      self.E1 = bpl.neurons.ProxyLIF(num_exc, **pars, method=method)
      self.I1 = bpl.neurons.ProxyLIF(num_inh, **pars, method=method)
      self.I2 = bp.neurons.LIF(num_inh, **pars, method=method)
      self.I3 = bp.neurons.LIF(num_inh, **pars, method=method)
      self.I4 = bpl.neurons.ProxyLIF(num_inh, **pars, method=method)
      self.I5 = bpl.neurons.ProxyLIF(num_inh, **pars, method=method)
    elif self.rank == 2:
      self.E1 = bpl.neurons.ProxyLIF(num_exc, **pars, method=method)
      self.I1 = bpl.neurons.ProxyLIF(num_inh, **pars, method=method)
      self.I2 = bpl.neurons.ProxyLIF(num_inh, **pars, method=method)
      self.I3 = bpl.neurons.ProxyLIF(num_inh, **pars, method=method)
      self.I4 = bp.neurons.LIF(num_inh, **pars, method=method)
      self.I5 = bpl.neurons.ProxyLIF(num_inh, **pars, method=method)
    elif self.rank == 3:
      self.E1 = bpl.neurons.ProxyLIF(num_exc, **pars, method=method)
      self.I1 = bpl.neurons.ProxyLIF(num_inh, **pars, method=method)
      self.I2 = bpl.neurons.ProxyLIF(num_inh, **pars, method=method)
      self.I3 = bpl.neurons.ProxyLIF(num_inh, **pars, method=method)
      self.I4 = bpl.neurons.ProxyLIF(num_inh, **pars, method=method)
      self.I5 = bp.neurons.LIF(num_inh, **pars, method=method)
    self.remoteE12I2 = bpl.synapses.RemoteExponential(0, self.E1, 1, self.I2,
                                                      bp.conn.FixedProb(0.02, seed=1),
                                                      output=bp.synouts.COBA(E=0.), g_max=we,
                                                      tau=5.,
                                                      method=method,
                                                      delay_step=delay_step
                                                      )
    self.remoteE12I3 = bpl.synapses.RemoteExponential(0, self.E1, 1, self.I3,
                                                      bp.conn.FixedProb(0.02, seed=1),
                                                      output=bp.synouts.COBA(E=0.), g_max=we,
                                                      tau=5.,
                                                      method=method,
                                                      delay_step=delay_step
                                                      )
    self.remoteE12I4 = bpl.synapses.RemoteExponential(0, self.E1, 2, self.I4,
                                                      bp.conn.FixedProb(0.02, seed=1),
                                                      output=bp.synouts.COBA(E=0.), g_max=we,
                                                      tau=5.,
                                                      method=method,
                                                      delay_step=delay_step
                                                      )
    self.remoteE12I5 = bpl.synapses.RemoteExponential(0, self.E1[:100], 3, self.I5[:100],
                                                      bp.conn.FixedProb(0.02, seed=1),
                                                      output=bp.synouts.COBA(E=0.), g_max=we,
                                                      tau=5.,
                                                      method=method,
                                                      delay_step=delay_step
                                                      )
    self.remoteI42I5 = bpl.synapses.RemoteExponential(2, self.I4, 3, self.I5[:100],
                                                      bp.conn.FixedProb(0.02, seed=1),
                                                      output=bp.synouts.COBA(E=0.), g_max=wi,
                                                      tau=5.,
                                                      method=method,
                                                      delay_step=delay_step
                                                      )


class MPITestCase(unittest.TestCase):
  def run_model(self, delay_step):
    # no delay
    net = EINet_V1(scale=1., method='exp_auto', delay_step=delay_step)
    runner = bp.dyn.DSRunner(net, monitors={'I5.spike': net.I5.spike}, inputs=[(net.E1.input, 20.)], jit=True)
    runner.run(20.)

  @pytest.mark.skip(reason="Unable to specify several processes")
  def test_run_model_v1(self):
    # the delay of all the synapse is None
    self.run_model(delay_step=None)

  @pytest.mark.skip(reason="Unable to specify several processes")
  def test_run_model_v2(self):
    # the delay of all the synapse is same
    delay_step = bm.random.randint(1, 10)
    self.run_model(delay_step=delay_step)


if __name__ == '__main__':
  unittest.main()
