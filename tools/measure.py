# @Time    : 2024/3/1 11:17
# @Author  : zhangchenming
import argparse
import sys
import time
from collections import defaultdict

import thop
import torch
from easydict import EasyDict
from tqdm import tqdm

sys.path.insert(0, './')
from stereo.utils import common_utils
from stereo.modeling import build_trainer


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--dist_mode', action='store_true', default=False, help='torchrun ddp multi gpu')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config for training')
    parser.add_argument('--repetitions', type=int, default=100, help='iterations for avg inference time')
    parser.add_argument('--module_time', action='store_true', help='report per-module inference time')
    parser.add_argument('--module_repetitions', type=int, default=50, help='iterations for per-module breakdown')
    parser.add_argument('--module_prefix', type=str, default=None, help='only show aggregated time for top-level module prefix (e.g. backbone)')

    args = parser.parse_args()
    yaml_config = common_utils.config_loader(args.cfg_file)
    cfgs = EasyDict(yaml_config)
    args.run_mode = 'measure'
    return args, cfgs


def main():
    args, cfgs = parse_config()
    model = build_trainer(args, cfgs, local_rank=0, global_rank=0, logger=None, tb_writer=None).model

    shape = [1, 3, 544, 960]
    infer_time(model, shape, repetitions=args.repetitions)
    measure(model, shape)
    if args.module_time:
        measure_module_time(model, shape, repetitions=args.module_repetitions, module_prefix=args.module_prefix)


@torch.no_grad()
def measure(model, shape):
    model.eval()

    inputs = {'left': torch.randn(shape).cuda(),
              'right': torch.randn(shape).cuda()}

    flops, params = thop.profile(model, inputs=(inputs,))
    print("Number of calculates:%.2fGFlops" % (flops / 1e9))
    print("Number of parameters:%.2fM" % (params / 1e6))


@torch.no_grad()
def infer_time(model, shape, repetitions):
    model.eval()

    inputs = {'left': torch.randn(shape).cuda(),
              'right': torch.randn(shape).cuda()}

    # 预热, GPU 平时可能为了节能而处于休眠状态, 因此需要预热
    print('warm up ...\n')
    with torch.no_grad():
        for _ in range(10):
            _ = model(inputs)

    # synchronize 等待所有 GPU 任务处理完才返回 CPU 主线程
    # torch.cuda.synchronize()

    # 设置用于测量时间的 cuda Event, 这是PyTorch 官方推荐的接口,理论上应该最靠谱
    # starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    # 初始化一个时间容器
    # timings = np.zeros((repetitions, 1))

    all_time = 0
    print('testing ...\n')
    with torch.no_grad():
        for _ in tqdm(range(repetitions)):
            # starter.record()
            infer_start = time.perf_counter()
            # infer_start = time.time()
            result = model(inputs)
            print(result.keys())
            # ender.record()
            all_time += time.perf_counter() - infer_start
            # torch.cuda.synchronize()  # 等待GPU任务完成

            # curr_time = starter.elapsed_time(ender)  # 从 starter 到 ender 之间用时,单位为毫秒
            # timings[rep] = curr_time

    # avg = timings.sum() / repetitions
    # print('\navg_time=%.3fms\n' % avg)
    print('\navg_time=%.3fms\n' % (all_time / repetitions * 1000))


@torch.no_grad()
def measure_module_time(model, shape, repetitions=50, module_prefix=None):
    """
    粗粒度统计每个叶子模块的平均耗时，可选按顶层前缀聚合，按单次前向总耗时排序。
    """
    model.eval()
    inputs = {'left': torch.randn(shape).cuda(),
              'right': torch.randn(shape).cuda()}

    module_time = defaultdict(float)
    module_calls = defaultdict(int)
    module_starts = {}
    handles = []

    def is_leaf(module):
        return len(list(module.children())) == 0

    def pre_hook(name):
        def hook(module, inputs):
            torch.cuda.synchronize()
            module_starts[name] = time.perf_counter()
        return hook

    def post_hook(name):
        def hook(module, inputs, output):
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - module_starts.pop(name, 0.0)
            module_time[name] += elapsed
            module_calls[name] += 1
        return hook

    for name, module in model.named_modules():
        if is_leaf(module):
            handles.append(module.register_forward_pre_hook(pre_hook(name)))
            handles.append(module.register_forward_hook(post_hook(name)))

    # 预热
    with torch.no_grad():
        for _ in range(5):
            _ = model(inputs)

    print('module time testing ...\n')
    with torch.no_grad():
        for _ in tqdm(range(repetitions)):
            _ = model(inputs)

    for handle in handles:
        handle.remove()

    print('\nmodule avg time per forward (sorted by total per iteration):')
    grouped_time = defaultdict(float)
    grouped_calls = defaultdict(int)
    for name in module_time:
        top = name.split('.')[0] if '.' in name else name
        grouped_time[top] += module_time[name]
        grouped_calls[top] += module_calls[name]

    def emit(name, total, calls):
        total_ms = total / repetitions * 1000
        per_call_ms = total / calls * 1000 if calls else 0.0
        calls_per_iter = calls / repetitions
        print(f'{name:60s} total/iter: {total_ms:8.3f} ms | per_call: {per_call_ms:8.3f} ms | calls/iter: {calls_per_iter:5.1f}')

    if module_prefix:
        if module_prefix not in grouped_time:
            print(f'prefix "{module_prefix}" not found in module names')
        else:
            emit(module_prefix, grouped_time[module_prefix], grouped_calls[module_prefix])
    else:
        for name in sorted(grouped_time, key=lambda k: grouped_time[k], reverse=True):
            emit(name, grouped_time[name], grouped_calls[name])


if __name__ == '__main__':
    main()
