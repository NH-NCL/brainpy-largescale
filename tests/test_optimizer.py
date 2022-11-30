from bpl.respa.optimizer import Optimizer
from bpl.respa.res_manager import ResManager
import bpl
import brainpy as bp
import numpy as np
from .base import BaseTest


class TestOptimizer(BaseTest):
  def test_get_edge_weight_matrix(self):
    a = bpl.LIF(2, V_rest=-60., V_th=-50., V_reset=-60., tau=20.,
                tau_ref=5., method='exp_auto')
    b = bpl.LIF(3, V_rest=-60., V_th=-50., V_reset=-60., tau=20.,
                tau_ref=5., method='exp_auto')
    c = bpl.LIF(4, V_rest=-60., V_th=-50., V_reset=-60., tau=20., tau_ref=5., method='exp_auto')
    d = bpl.LIF(5, V_rest=-60., V_th=-50., V_reset=-60., tau=20., tau_ref=5., method='exp_auto')
    bpl.Exponential(a, b, bp.conn.FixedProb(1, seed=123), g_max=10, tau=5.,
                    output=bp.synouts.COBA(E=0.), method='exp_auto', delay_step=1)
    bpl.Exponential(a, c, bp.conn.FixedProb(1, seed=123), g_max=10, tau=5.,
                    output=bp.synouts.COBA(E=0.), method='exp_auto', delay_step=1)
    bpl.Exponential(b, c, bp.conn.FixedProb(1, seed=123), g_max=10, tau=5.,
                    output=bp.synouts.COBA(E=0.), method='exp_auto', delay_step=1)
    bpl.Exponential(b, d, bp.conn.FixedProb(1, seed=123), g_max=10, tau=5.,
                    output=bp.synouts.COBA(E=0.), method='exp_auto', delay_step=1)
    bpl.Exponential(d, a, bp.conn.FixedProb(1, seed=123), g_max=10, tau=5.,
                    output=bp.synouts.COBA(E=0.), method='exp_auto', delay_step=1)
    opt = Optimizer()
    matrix = opt.get_edge_weight_matrix(ResManager.syns, total_pop_num=len(ResManager.pops))
    self.assertTrue(np.array_equal(matrix, [[0., 6., 8., 0.],
                                            [0., 0., 12., 15.],
                                            [0., 0., 0., 0.],
                                            [10., 0., 0., 0.]]))
