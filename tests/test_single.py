import unittest
import sys
sys.path.append('../')
import bpl
import brainpy as bp
import brainpy.math as bm


class EINet_V1(bpl.Network):
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
    self.E1 = bp.neurons.LIF(num_exc, **pars, method=method)
    self.I1 = bp.neurons.LIF(num_inh, **pars, method=method)
    self.I2 = bp.neurons.LIF(num_inh, **pars, method=method)
    self.I3 = bp.neurons.LIF(num_inh, **pars, method=method)
    self.I4 = bp.neurons.LIF(num_inh, **pars, method=method)
    self.I5 = bp.neurons.LIF(num_inh, **pars, method=method)
    self.E12I1 = bp.synapses.Exponential(self.E1, self.I1,
                                         bp.conn.FixedProb(0.02, seed=1),
                                         output=bp.synouts.COBA(E=0.), g_max=we,
                                         tau=5.,
                                         method=method, delay_step=delay_step)
    self.E12I2 = bp.synapses.Exponential(self.E1, self.I2,
                                         bp.conn.FixedProb(0.02, seed=1),
                                         output=bp.synouts.COBA(E=0.), g_max=we,
                                         tau=5.,
                                         method=method,
                                         delay_step=delay_step
                                         )
    self.E12I3 = bp.synapses.Exponential(self.E1, self.I3,
                                         bp.conn.FixedProb(0.02, seed=1),
                                         output=bp.synouts.COBA(E=0.), g_max=we,
                                         tau=5.,
                                         method=method,
                                         delay_step=delay_step
                                         )
    self.E12I4 = bp.synapses.Exponential(self.E1, self.I4,
                                         bp.conn.FixedProb(0.02, seed=1),
                                         output=bp.synouts.COBA(E=0.), g_max=we,
                                         tau=5.,
                                         method=method,
                                         delay_step=delay_step
                                         )
    self.E12I5 = bp.synapses.Exponential(self.E1, self.I5,
                                         bp.conn.FixedProb(0.02, seed=1),
                                         output=bp.synouts.COBA(E=0.), g_max=we,
                                         tau=5.,
                                         method=method,
                                         delay_step=delay_step
                                         )
    self.I42I5 = bp.synapses.Exponential(self.I4, self.I5,
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

  def test_run_model_v1(self):
    # the delay of all the synapse is None
    self.run_model(delay_step=None)

  def test_run_model_v2(self):
    # the delay of all the synapse is same
    delay_step = bm.random.randint(1, 10)
    self.run_model(delay_step=delay_step)


if __name__ == '__main__':
  unittest.main()
