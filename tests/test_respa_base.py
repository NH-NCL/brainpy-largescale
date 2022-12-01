import unittest
import bpl
import brainpy as bp
from brainpy.dyn import channels, synouts
import brainpy.math as bm
from .base import BaseTest
import pytest


class BaseFunctionsTestCase(BaseTest):
  # @pytest.mark.skip(reason="Unable to specify several processes")
  def testbasefunc(self):
    class MyNetwork(bpl.respa.Network):
      def __init__(self, *ds_tuple):
        super(MyNetwork, self).__init__(ds_tuple)
        self.a = bpl.LIF(3200, V_rest=-60., V_th=-50., V_reset=-60., tau=20.,
                         tau_ref=5., method='exp_auto', V_initializer=bp.initialize.Normal(-55., 2.))
        self.b = bpl.LIF(800, V_rest=-60., V_th=-50., V_reset=-60., tau=20.,
                         tau_ref=5., method='exp_auto', V_initializer=bp.initialize.Normal(-55., 2.))
        # self.c = bpl.Exponential(ds_tuple[0], self.a, bp.conn.FixedProb(
        #     0.02), g_max=10, tau=5., output=bp.synouts.COBA(E=0.), method='exp_auto')
        self.d = bpl.Exponential(self.a, self.b, bp.conn.FixedProb(
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
    runner.run(5.)
    if 'spike' in runner.mon:
      # bp.visualize.raster_plot(
      #   runner.mon.ts, runner.mon['spike'], show=True)
    #   print(net.pops_)
      print(net.pops_by_rank)
    #   print(net.syns_)
    #   print(net.nodes())

  # @pytest.mark.skip(reason="Unable to specify several processes")
  def testbaseregister(self):
    @bpl.register()
    class HH(bp.dyn.CondNeuGroup):
      def __init__(self, size):
        super(HH, self).__init__(size, )
        self.INa = channels.INa_TM1991(size, g_max=100., V_sh=-63.)
        self.IK = channels.IK_TM1991(size, g_max=30., V_sh=-63.)
        self.IL = channels.IL(size, E=-60., g_max=0.05)

    @bpl.register()
    class ExpCOBA(bp.dyn.TwoEndConn):
      def __init__(self, pre, post, conn, g_max=1., delay=0., tau=8.0, E=0.,
                   method='exp_auto'):
        super(ExpCOBA, self).__init__(pre=pre, post=post, conn=conn)
        self.check_pre_attrs('spike')
        self.check_post_attrs('input', 'V')

        # parameters
        self.E = E
        self.tau = tau
        self.delay = delay
        self.g_max = g_max
        self.pre2post = self.conn.require('pre2post')

        # variables
        self.g = bm.Variable(bm.zeros(self.post.num))

        # function
        self.integral = bp.odeint(lambda g, t: -g / self.tau, method=method)

      def update(self, tdi):
        self.g.value = self.integral(self.g, tdi.t, tdi.dt)
        self.g += bm.pre2post_event_sum(self.pre.spike,
                                        self.pre2post, self.post.num, self.g_max)
        self.post.input += self.g * (self.E - self.post.V)

    class EINet_v1(bpl.respa.Network):
      def __init__(self, scale=1.):
        super(EINet_v1, self).__init__()
        self.E = bpl.HH(int(3200 * scale))
        self.I = bpl.HH(int(800 * scale))
        prob = 0.02
        self.E2E = bpl.ExpCOBA(self.E, self.E, bp.conn.FixedProb(prob),
                               g_max=0.03 / scale, tau=5)
        self.E2I = bpl.ExpCOBA(self.E, self.I, bp.conn.FixedProb(prob),
                               g_max=0.03 / scale, tau=5.)
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
      runner.run(50.)
      # bp.visualize.raster_plot(runner.mon.ts, runner.mon['E.spike'], show=True)

    run_ei_v1()


if __name__ == '__main__':
  unittest.main()
