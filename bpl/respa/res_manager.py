class ResManager():
  # [BaseNeuron(), ]
  pops = []

  # {id: BaseNeuron()}, id is BaseNeuron's id
  pops_by_id = {}

  # [BaseSynapse(), ]
  syns = []

  # element is an array, like [BaseNeuron(), ], pops_by_rank can be expressed as: {device_index: [BaseNeuron(), ] }
  pops_by_rank = {}

  @classmethod
  def clear(cls):
    cls.pops = []
    cls.pops_by_id = {}
    cls.syns = []
    cls.pops_by_rank = {}
