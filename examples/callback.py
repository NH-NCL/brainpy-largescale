import sys

sys.path.append('../')

import bpl
import brainpy as bp
from typing import List, Tuple

a = bpl.LIF(
  200,
  V_rest=-60.,
  V_th=-50.,
  V_reset=-60.,
  tau=20.,
  tau_ref=5.,
  method='exp_auto',
  V_initializer=bp.initialize.Normal(-55., 2.))
b = bpl.LIF(
  100,
  V_rest=-60.,
  V_th=-50.,
  V_reset=-60.,
  tau=20.,
  tau_ref=5.,
  method='exp_auto',
  V_initializer=bp.initialize.Normal(-55., 2.))
d = bpl.Exponential(
  a,
  b,
  bp.conn.FixedProb(0.04, seed=123),
  g_max=10,
  tau=5.,
  output=bp.synouts.COBA(E=0.),
  method='exp_auto',
  delay_step=1)

net = bpl.respa.Network(a, b, d)
net.build()

inputs = bpl.input_transform([(a, 20)])
monitor_spike = bpl.monitor_transform([a], attr='spike')
monitor_volt = bpl.monitor_transform([a], attr='V')
monitors = {}
monitors.update(monitor_spike)
monitors.update(monitor_volt)


def spike(a: List[Tuple[int, float]]):
  if a:
    print(a)


def volt(a: List[Tuple[int, float, float]]):
  # print(a)
  pass


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
