import unittest
import brainpy.respa as respa
import brainpy as bp


class BaseFunctionsTestCase(unittest.TestCase):
  def testbasefunc(self):
    class MyNetwork(respa.Network):
      def __init__(self, *ds_tuple):
        super(MyNetwork, self).__init__(ds_tuple)
        self.a = respa.LIF(3200, V_rest=-60., V_th=-50., V_reset=-60., tau=20.,
                           tau_ref=5., method='exp_auto', V_initializer=bp.initialize.Normal(-55., 2.))
        self.b = respa.LIF(800, V_rest=-60., V_th=-50., V_reset=-60., tau=20.,
                           tau_ref=5., method='exp_auto', V_initializer=bp.initialize.Normal(-55., 2.))
        # self.c = respa.Exponential(ds_tuple[0], self.a, bp.conn.FixedProb(
        #     0.02), g_max=10, tau=5., output=bp.synouts.COBA(E=0.), method='exp_auto')
        self.d = respa.Exponential(self.a[100:], self.b, bp.conn.FixedProb(
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
    inputs = respa.input_transform([(net.a, 20)])
    monitors = respa.monitor_transform([net.b])
    runner = respa.DSRunner(
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


if __name__ == '__main__':
  unittest.main()
