# @Time    : 2024/5/6 01:16
# @Author  : zhangchenming

from stereo.modeling.trainer_template import TrainerTemplate
from .hagvstereo import HAGVstereo

__all__ = {
    'HAGVstereo': HAGVstereo,
}


class Trainer(TrainerTemplate):
    def __init__(self, args, cfgs, local_rank, global_rank, logger, tb_writer):
        model = __all__[cfgs.MODEL.NAME](cfgs.MODEL)
        super().__init__(args, cfgs, local_rank, global_rank, logger, tb_writer, model)
