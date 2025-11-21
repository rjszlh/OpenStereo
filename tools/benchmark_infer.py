import sys
import argparse
import time

import torch
import torch.nn as nn
from easydict import EasyDict

sys.path.insert(0, './')

from stereo.utils import common_utils
from stereo.modeling import build_trainer


def parse_config():
    parser = argparse.ArgumentParser(description='Model inference benchmark')
    parser.add_argument('--cfg_file', type=str, required=True,
                        help='path to config yaml')
    parser.add_argument('--pretrained_model', type=str, default=None,
                        help='path to .pth checkpoint (overrides cfg)')
    parser.add_argument('--height', type=int, default=256,
                        help='input height, e.g. 256')
    parser.add_argument('--width', type=int, default=512,
                        help='input width, e.g. 512')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='device to run on, e.g. cuda:0')
    parser.add_argument('--warmup_iters', type=int, default=10,
                        help='number of warmup iterations')
    parser.add_argument('--iters', type=int, default=1000,
                        help='number of timed iterations')

    args = parser.parse_args()

    yaml_config = common_utils.config_loader(args.cfg_file)
    cfgs = EasyDict(yaml_config)

    if args.pretrained_model is not None:
        cfgs.MODEL.PRETRAINED_MODEL = args.pretrained_model

    # minimal args required by build_trainer / TrainerTemplate
    args.dist_mode = False
    args.run_mode = 'infer'

    return args, cfgs


@torch.no_grad()
def main():
    args, cfgs = parse_config()

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for this benchmark but is not available.')

    device = torch.device(args.device)
    if device.type != 'cuda':
        raise RuntimeError('Please use a CUDA device, e.g. --device cuda:0')

    torch.cuda.set_device(device)
    local_rank = device.index if device.index is not None else 0
    global_rank = 0

    common_utils.set_random_seed(0)

    logger = common_utils.create_logger(log_level='INFO', rank=local_rank)

    logger.info('Benchmark configuration:')
    for key, val in vars(args).items():
        logger.info('{:16} {}'.format(key, val))
    common_utils.log_configs(cfgs, logger=logger)

    logger.info('Building model and loading weights...')
    trainer = build_trainer(args, cfgs, local_rank, global_rank, logger, tb_writer=None)
    model = trainer.model

    # For RLightStereo with ACIR aggregation, optionally re-parameterize
    # ACIR blocks into their deploy form before benchmarking.
    # if getattr(cfgs, 'MODEL', None) is not None \
    #         and cfgs.MODEL.get('NAME', None) == 'RLightStereo' \
    #         and cfgs.MODEL.get('AGGREGATION_TYPE', None) == 'ACIR':
    #     from stereo.modeling.models.rlightstereo.acir_block import ACIRBlockECA
    #
    #     base_model = model.module if isinstance(model, nn.parallel.DistributedDataParallel) else model
    #     num_blocks = 0
    #     for m in base_model.modules():
    #         if isinstance(m, ACIRBlockECA):
    #             m.switch_to_deploy()
    #             num_blocks += 1
    #     logger.info(f'Switched {num_blocks} ACIRBlockECA modules to deploy mode for benchmarking.')

    model.eval()

    h, w = args.height, args.width
    logger.info(f'Creating dummy input: shape = [1, 3, {h}, {w}]')
    left = torch.randn(1, 3, h, w, device=device)
    right = torch.randn(1, 3, h, w, device=device)
    sample = {
        'left': left,
        'right': right,
    }

    use_amp = cfgs.OPTIMIZATION.AMP

    # warmup
    if args.warmup_iters > 0:
        logger.info(f'Warmup for {args.warmup_iters} iterations...')
        for _ in range(args.warmup_iters):
            with torch.cuda.amp.autocast(enabled=use_amp):
                _ = model(sample)
            torch.cuda.synchronize(device)

    # timed iterations
    iters = max(1, args.iters)
    logger.info(f'Start timing for {iters} iterations...')
    torch.cuda.synchronize(device)
    start = time.perf_counter()
    for _ in range(iters):
        with torch.cuda.amp.autocast(enabled=use_amp):
            _ = model(sample)
        torch.cuda.synchronize(device)
    end = time.perf_counter()

    total_time = end - start
    avg_time = total_time / iters
    fps = 1.0 / avg_time if avg_time > 0 else float('inf')

    logger.info('========== Benchmark Result ==========')
    logger.info(f'Total time:      {total_time * 1000:.2f} ms for {iters} iters')
    logger.info(f'Average latency: {avg_time * 1000:.2f} ms / iter')
    logger.info(f'Throughput:      {fps:.2f} FPS')


if __name__ == '__main__':
    main()
