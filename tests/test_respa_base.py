import unittest
import bpl
import brainpy as bp
from brainpy.dyn import channels, synouts


class BaseFunctionsTestCase(unittest.TestCase):
  def testbasefunc(self):
    class MyNetwork(bpl.Network):
      def __init__(self, *ds_tuple):
        super(MyNetwork, self).__init__(ds_tuple)
        self.a = bpl.LIF(3200, V_rest=-60., V_th=-50., V_reset=-60., tau=20.,
                           tau_ref=5., method='exp_auto', V_initializer=bp.initialize.Normal(-55., 2.))
        self.b = bpl.LIF(800, V_rest=-60., V_th=-50., V_reset=-60., tau=20.,
                           tau_ref=5., method='exp_auto', V_initializer=bp.initialize.Normal(-55., 2.))
        # self.c = bpl.Exponential(ds_tuple[0], self.a, bp.conn.FixedProb(
        #     0.02), g_max=10, tau=5., output=bp.synouts.COBA(E=0.), method='exp_auto')
        self.d = bpl.Exponential(self.a[100:], self.b, bp.conn.FixedProb(
            0.02, seed=123), g_max=10, tau=5., output=bp.synouts.COBA(E=0.), method='exp_auto', delay_step=1)

    net = MyNetwork()
    net.build()
    # from mpi4py import MPI
    # if MPI.COMM_WORLD.Get_size() == 2:
    #   if MPI.COMM_WORLD.Get_rank() == 1:
    #     monitors = {'spikes': net.b.spike}
    #     # inputs = []
    #   else:
    #     monitors = {}
    #     # inputs = [(net.a.input, 20.)]
    # else:
    #   monitors = {'spikes': net.b.spike}
    #   inputs = [(net.a.input, 20.)]
    inputs = bpl.input_transform([(net.a, 20)])
    monitors = bpl.monitor_transform([net.b])
    runner = bpl.DSRunner(
        net,
        monitors=monitors,
        inputs=inputs,
        jit=False
    )
    runner.run(10.)
    if 'spike' in runner.mon:
      bp.visualize.raster_plot(
          runner.mon.ts, runner.mon['spike'], show=True)
      print(net.pops_)
      print(net.pops_split)
      print(net.syns_)
      # print(net.nodes())

  def testBaseNeuronregister(self):
    @bpl.BaseNeuron.register
    class HH(bp.dyn.CondNeuGroup):
      def __init__(self, size):
        super(HH, self).__init__(size, )
        self.INa = channels.INa_TM1991(size, g_max=100., V_sh=-63.)
        self.IK = channels.IK_TM1991(size, g_max=30., V_sh=-63.)
        self.IL = channels.IL(size, E=-60., g_max=0.05)

    class EINet_v1(bpl.Network):
      def __init__(self, scale=1.):
        super(EINet_v1, self).__init__()
        self.E = bpl.HH(int(3200 * scale))
        self.I = bpl.HH(int(800 * scale))
        prob = 0.02
        self.E2E = bpl.Exponential(self.E, self.E, bp.conn.FixedProb(prob),
                                     g_max=0.03 / scale, tau=5,
                                     output=synouts.COBA(E=0.))
        self.E2I = bpl.Exponential(self.E, self.I, bp.conn.FixedProb(prob),
                                     g_max=0.03 / scale, tau=5.,
                                     output=synouts.COBA(E=0.))
        self.I2E = bpl.Exponential(self.I, self.E, bp.conn.FixedProb(prob),
                                     g_max=0.335 / scale, tau=10.,
                                     output=synouts.COBA(E=-80))
        self.I2I = bpl.Exponential(self.I, self.I, bp.conn.FixedProb(prob),
                                     g_max=0.335 / scale, tau=10.,
                                     output=synouts.COBA(E=-80.))

    def run_ei_v1():
      net = EINet_v1(scale=1)
      net.build()
      runner = bpl.DSRunner(net, monitors={'E.spike': net.E.spike})
      runner.run(100.)
      bp.visualize.raster_plot(runner.mon.ts, runner.mon['E.spike'], show=True)

    run_ei_v1()


if __name__ == '__main__':
  unittest.main()
