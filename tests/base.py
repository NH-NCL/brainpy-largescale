import unittest
import bpl


class BaseTest(unittest.TestCase):

  def setUp(self) -> None:
    bpl.ResManager.clear()
    return super().setUp()
