import unittest
import bpl
import brainpy as bp
from .base import BaseTest
# import pytest


class BaseFunctionsTestCase(BaseTest):
  # @pytest.mark.skip(reason="Unable to specify several processes")
  def testbasefunc(self):
    a = bpl.neurons.LIF(300, V_rest=-60., V_th=-50., V_reset=-60., tau=20., tau_ref=5.)
    b = bpl.neurons.LIF(100, V_rest=-60., V_th=-50., V_reset=-60., tau=20., tau_ref=5.)
    d = bpl.synapses.Exponential(a, b, bp.conn.FixedProb(0.4, seed=123), g_max=10, tau=5., delay_step=1)

    net = bpl.Network(a, b, d)
    net.build()

    inputs = [bpl.device.Input(a, 20), bpl.device.Input(b, 10)]
    monitor_spike = bpl.device.Monitor([a, b], bpl.device.MonitorKey.spike)
    monitor_volt = bpl.device.Monitor([b], bpl.device.MonitorKey.volt)
    monitors = [monitor_spike, monitor_volt]

    runner = bpl.runner.DSRunner(
      net,
      monitors=monitors,
      inputs=inputs,
      jit=False,
    )
    runner.run(10.)


if __name__ == '__main__':
  unittest.main()
