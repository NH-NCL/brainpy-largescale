import bpl
import brainpy as bp

class MyNetwork(bpl.Network):
  def __init__(self, *ds_tuple):
    super(MyNetwork, self).__init__(ds_tuple)
    self.a = bpl.LIF(20, V_rest=-60., V_th=-50., V_reset=-60., tau=20.,
                      tau_ref=5., method='exp_auto', V_initializer=bp.initialize.Normal(-55., 2.))
    self.b = bpl.LIF(10, V_rest=-60., V_th=-50., V_reset=-60., tau=20.,
                      tau_ref=5., method='exp_auto', V_initializer=bp.initialize.Normal(-55., 2.))
    # self.c = bpl.Exponential(ds_tuple[0], self.a, bp.conn.FixedProb(
    #     0.02), g_max=10, tau=5., output=bp.synouts.COBA(E=0.), method='exp_auto')
    self.d = bpl.Exponential(self.a[100:], self.b, bp.conn.FixedProb(
        0.2, seed=123), g_max=10, tau=5., output=bp.synouts.COBA(E=0.), method='exp_auto', delay_step=1)

net = MyNetwork()
net.build()
inputs = bpl.input_transform([(net.a, 20)])
monitor_spike = bpl.monitor_transform([net.a], attr='spike')
monitor_volt = bpl.monitor_transform([net.a], attr='V')
monitors = {}
monitors.update(monitor_spike)
monitors.update(monitor_volt)
def spike(s: str):
  print(s)

def volt(s: str):
  print(s)

runner = bpl.DSRunner(
    net,
    monitors=monitors,
    inputs=inputs,
    jit=False,
    spike_callback=spike,
    volt_callback=volt,
)
runner.run(10.)

if 'spike' in runner.mon:
  bp.visualize.raster_plot(runner.mon.ts, runner.mon['spike'], show=True)
elif 'V' in runner.mon:
  bp.visualize.raster_plot(runner.mon.ts, runner.mon['V'], show=True)
