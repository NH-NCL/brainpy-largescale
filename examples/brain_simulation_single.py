import sys
sys.path.append('../')
import brainpy as bp

bp.math.set_platform('cpu')


class EINet_V1(bp.dyn.Network):
  def __init__(self, scale=1.0, method='exp_auto'):
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
    self.E12I1 = bp.synapses.Exponential(self.E1, self.I1, 
                                      bp.conn.FixedProb(0.02,seed=1),
                                       output=bp.synouts.COBA(E=0.), g_max=we,
                                       tau=5., 
                                       method=method,
                                       delay_step=1)
    self.I2 = bp.neurons.LIF(num_inh, **pars, method=method)
    self.E12I2 = bp.synapses.Exponential(self.E1, self.I2, 
                                                    bp.conn.FixedProb(0.02,seed=1), 
                                                      output=bp.synouts.COBA(E=0.), g_max=we,
                                                      tau=5., 
                                                      method=method,
                                                      delay_step=1
                                                      )

def run_model_v1():
  net = EINet_V1(scale=1., method='exp_auto')
  runner = bp.dyn.DSRunner(
    net,
    monitors={'I2.spike': net.I2.spike},
    inputs=[(net.E1.input, 200.), (net.I1.input, 200.)],
    jit=True
  )
  runner.run(10.)

  # visualization
  bp.visualize.raster_plot(runner.mon.ts, runner.mon['I2.spike'], show=True)


if __name__ == '__main__':
  run_model_v1()
  