# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Benchmark script: compare inference performance of base FusedMoE vs FP4-dispatch FusedMoE.

Each method is warmed up 20 times, then timed over 100 iterations.

Launch with 8 GPUs:
    PYTHONPATH="" python -m paddle.distributed.launch --gpus 0,1,2,3,4,5,6,7 \
        tests/quantization/bench_nvfp4_ep_prefill.py
"""

import json
import os
import shutil
import time

os.environ.setdefault("FD_MOE_BACKEND", "flashinfer-cutedsl")

import paddle
from paddle.distributed import fleet

from fastdeploy.config import (
    CacheConfig,
    EPLBConfig,
    FDConfig,
    GraphOptimizationConfig,
    LoadConfig,
    ModelConfig,
    ParallelConfig,
    RoutingReplayConfig,
)
from fastdeploy.model_executor.layers.linear import ReplicatedLinear
from fastdeploy.model_executor.layers.moe.moe import FusedMoE
from fastdeploy.model_executor.layers.quantization.nvfp4 import ModelOptNvFp4Config
from fastdeploy.scheduler import SchedulerConfig
from fastdeploy.worker.worker_process import init_distributed_environment

paddle.set_default_dtype("bfloat16")

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
NUM_EXPERTS = 160
EP_SIZE = 8
NUM_LOCAL_EXPERTS = NUM_EXPERTS // EP_SIZE
HIDDEN_SIZE = 7168
MOE_INTERMEDIATE_SIZE = 3584
NUM_TOKENS = 4096 * 8
TOP_K = 4
GROUP_SIZE = 16

WARMUP_ITERS = 20
BENCH_ITERS = 100


class MockForwardMeta:
    def __init__(self):
        self.moe_num_chunk = 1
        self.max_moe_num_chunk = 1
        self.routing_replay_table = None


class NvFp4MoEWrapper(paddle.nn.Layer):
    def __init__(self, model_config, ep_size, ep_rank, prefix="layer0", use_fp4_dispatch=False):
        super().__init__()
        self.model_config = model_config
        self.ep_size = ep_size
        self.ep_rank = ep_rank
        self.prefix = prefix

        quant_config = ModelOptNvFp4Config(
            is_checkpoint_nvfp4_serialized=True,
            kv_cache_quant_algo=None,
            exclude_modules=[],
            group_size=GROUP_SIZE,
            use_fp4_dispatch=use_fp4_dispatch,
        )

        nnodes = (ep_size + 7) // 8

        self.fd_config = FDConfig(
            model_config=self.model_config,
            parallel_config=ParallelConfig(
                {
                    "tensor_parallel_size": 1,
                    "expert_parallel_size": self.ep_size,
                    "expert_parallel_rank": self.ep_rank,
                    "data_parallel_size": self.ep_size,
                    "ep_prefill_use_worst_num_tokens": True,
                }
            ),
            quant_config=quant_config,
            scheduler_config=SchedulerConfig(
                {
                    "splitwise_role": "prefill",
                    "max_num_batched_tokens": NUM_TOKENS,
                }
            ),
            eplb_config=EPLBConfig({}),
            cache_config=CacheConfig({}),
            graph_opt_config=GraphOptimizationConfig({}),
            load_config=LoadConfig({}),
            ips=",".join(["0"] * nnodes),
            routing_replay_config=RoutingReplayConfig({}),
        )

        self.fd_config.parallel_config.tp_group = None
        self.fd_config.parallel_config.tensor_parallel_rank = 0
        self.fd_config.parallel_config.expert_parallel_size = self.ep_size
        self.fd_config.parallel_config.ep_group = fleet.get_hybrid_communicate_group().get_model_parallel_group()
        self.fd_config.scheduler_config.splitwise_role = "prefill"
        self.fd_config.model_config.moe_phase.phase = "prefill"

        weight_key_map = {
            "gate_weight_key": f"{self.prefix}.gate.weight",
            "gate_correction_bias_key": f"{self.prefix}.moe_statics.e_score_correction_bias",
            "up_gate_proj_expert_weight_key": f"{self.prefix}.experts.{{}}.up_gate_proj.weight",
            "down_proj_expert_weight_key": f"{self.prefix}.experts.{{}}.down_proj.weight",
        }

        self.gating = ReplicatedLinear(
            fd_config=self.fd_config,
            prefix=f"{self.prefix}.gate",
            input_size=HIDDEN_SIZE,
            output_size=NUM_EXPERTS,
            with_bias=False,
            skip_quant=True,
            weight_dtype="float32",
        )

        self.fused_moe = FusedMoE(
            fd_config=self.fd_config,
            moe_intermediate_size=MOE_INTERMEDIATE_SIZE,
            num_experts=NUM_EXPERTS,
            top_k=TOP_K,
            layer_idx=666,
            weight_key_map=weight_key_map,
            topk_method="noaux_tc",
            topk_group=4,
            n_group=8,
            gate_correction_bias=paddle.zeros([NUM_EXPERTS], paddle.float32),
        )

        self._init_random_weights()

    def _init_random_weights(self):
        moe_layer = self.fused_moe
        quant_method = moe_layer.quant_method

        up_gate_w = getattr(moe_layer, quant_method.added_weight_attrs[0])
        down_w = getattr(moe_layer, quant_method.added_weight_attrs[1])
        up_gate_w.set_value(paddle.randint(0, 256, up_gate_w.shape, dtype="int32").cast("uint8"))
        down_w.set_value(paddle.randint(0, 256, down_w.shape, dtype="int32").cast("uint8"))

        up_gate_scale = getattr(moe_layer, quant_method.added_scale_attrs[0])
        down_scale = getattr(moe_layer, quant_method.added_scale_attrs[1])
        up_gate_scale.set_value(paddle.full(up_gate_scale.shape, 0.01, dtype="float32").cast("float8_e4m3fn"))
        down_scale.set_value(paddle.full(down_scale.shape, 0.01, dtype="float32").cast("float8_e4m3fn"))

        moe_layer.up_gate_proj_weight_scale_2.set_value(
            paddle.full(moe_layer.up_gate_proj_weight_scale_2.shape, 1.0, dtype="float32")
        )
        moe_layer.down_proj_weight_scale_2.set_value(
            paddle.full(moe_layer.down_proj_weight_scale_2.shape, 1.0, dtype="float32")
        )
        moe_layer.up_gate_proj_input_scale.set_value(
            paddle.full(moe_layer.up_gate_proj_input_scale.shape, 1.0, dtype="float32")
        )
        moe_layer.down_proj_input_scale.set_value(
            paddle.full(moe_layer.down_proj_input_scale.shape, 1.0, dtype="float32")
        )

        quant_method.process_weights_after_loading(moe_layer)
        self.gating.weight.set_value(paddle.randn(self.gating.weight.shape, dtype=paddle.float32) * 0.01)


PROFILE_ITERS = 5


def bench_method(wrapper, x, label, rank, profile=False):
    """Benchmark a single fused_moe method: warmup + timed iterations."""
    meta = MockForwardMeta()

    # Warmup
    for _ in range(WARMUP_ITERS):
        _ = wrapper.fused_moe(x, wrapper.gating, forward_meta=meta)
    paddle.device.cuda.synchronize()

    if profile:
        # Profiling: trace exactly PROFILE_ITERS calls of fused_moe
        safe_label = label.replace(" ", "_").replace("(", "").replace(")", "").replace("=", "")
        profile_dir = f"./profiler_output/{safe_label}_rank{rank}"
        profiler = paddle.profiler.Profiler(
            targets=[paddle.profiler.ProfilerTarget.CPU, paddle.profiler.ProfilerTarget.GPU],
            scheduler=paddle.profiler.make_scheduler(
                closed=0,
                ready=0,
                record=PROFILE_ITERS,
                repeat=1,
            ),
            on_trace_ready=paddle.profiler.export_chrome_tracing(profile_dir),
        )

        if profiler.profiler is None:
            # _Profiler.create can return None after a previous profiling session;
            # re-create the internal C++ profiler object.
            from paddle.framework import core

            profileoption = core.ProfilerOptions()
            profileoption.trace_switch = 0
            profileoption.trace_switch |= 1  # CPU
            profileoption.trace_switch |= 1 << 1  # GPU
            profiler.profiler = core._Profiler.create(profileoption, [])

        profiler.start()
        for i in range(PROFILE_ITERS):
            _ = wrapper.fused_moe(x, wrapper.gating, forward_meta=meta)
            paddle.device.cuda.synchronize()
            profiler.step()
        profiler.stop()

        if rank == 0:
            print(f"\n  [Profiler] {PROFILE_ITERS} calls traced -> {profile_dir}/")

    # Timed runs
    latencies = []
    for _ in range(BENCH_ITERS):
        paddle.device.cuda.synchronize()
        t0 = time.perf_counter()
        _ = wrapper.fused_moe(x, wrapper.gating, forward_meta=meta)
        paddle.device.cuda.synchronize()
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000)  # ms

    avg = sum(latencies) / len(latencies)
    p50 = sorted(latencies)[len(latencies) // 2]
    p90 = sorted(latencies)[int(len(latencies) * 0.9)]
    p99 = sorted(latencies)[int(len(latencies) * 0.99)]
    min_lat = min(latencies)
    max_lat = max(latencies)

    if rank == 0:
        print(f"\n{'='*60}")
        print(f"  {label}")
        print(f"{'='*60}")
        print(f"  Warmup iters : {WARMUP_ITERS}")
        print(f"  Bench  iters : {BENCH_ITERS}")
        print(f"  Avg latency  : {avg:.3f} ms")
        print(f"  P50 latency  : {p50:.3f} ms")
        print(f"  P90 latency  : {p90:.3f} ms")
        print(f"  P99 latency  : {p99:.3f} ms")
        print(f"  Min latency  : {min_lat:.3f} ms")
        print(f"  Max latency  : {max_lat:.3f} ms")

    return avg


def main():
    init_distributed_environment()

    ep_size = paddle.distributed.get_world_size()
    ep_rank = paddle.distributed.get_rank()

    assert ep_size == EP_SIZE, (
        f"This benchmark requires {EP_SIZE} GPUs, got {ep_size}. "
        f"Launch with: python -m paddle.distributed.launch "
        f"--gpus 0,1,2,3,4,5,6,7 {__file__}"
    )

    paddle.seed(ep_rank + 42)

    # Build model config
    config_dict = {
        "architectures": ["DeepseekV3ForCausalLM"],
        "hidden_size": HIDDEN_SIZE,
        "moe_intermediate_size": MOE_INTERMEDIATE_SIZE,
        "moe_num_experts": NUM_EXPERTS,
        "moe_k": TOP_K,
        "num_attention_heads": -1,
        "dtype": "bfloat16",
    }
    tmp_dir = f"/tmp/nvfp4_ep_bench_rank{ep_rank}"
    os.makedirs(tmp_dir, exist_ok=True)
    config_path = os.path.join(tmp_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config_dict, f)
    model_config = ModelConfig({"model": tmp_dir, "max_model_len": 2048})

    # Build wrappers
    if ep_rank == 0:
        print("\n>>> Building base wrapper (use_fp4_dispatch=False) ...")
    wrapper = NvFp4MoEWrapper(
        model_config=model_config,
        ep_size=ep_size,
        ep_rank=ep_rank,
        use_fp4_dispatch=False,
    )

    if ep_rank == 0:
        print(">>> Building FP4-dispatch wrapper (use_fp4_dispatch=True) ...")
    wrapper_fp4_dispatch = NvFp4MoEWrapper(
        model_config=model_config,
        ep_size=ep_size,
        ep_rank=ep_rank,
        use_fp4_dispatch=True,
    )

    # Synchronize weights
    src_state = wrapper.state_dict()
    dst_state = wrapper_fp4_dispatch.state_dict()
    for key in dst_state:
        if key in src_state:
            dst_state[key] = src_state[key]
    wrapper_fp4_dispatch.set_state_dict(dst_state)

    # Prepare input
    paddle.seed(0)
    x = paddle.randn([NUM_TOKENS, HIDDEN_SIZE], dtype="bfloat16")

    if ep_rank == 0:
        print(f"\n>>> Input shape: [{NUM_TOKENS}, {HIDDEN_SIZE}], dtype=bfloat16")
        print(f">>> Config: num_experts={NUM_EXPERTS}, ep_size={EP_SIZE}, top_k={TOP_K}")
        print(f">>> Warmup={WARMUP_ITERS}, Bench={BENCH_ITERS}")

    # Benchmark
    avg_base = bench_method(wrapper, x, "Base FusedMoE (use_fp4_dispatch=False)", ep_rank, profile=True)
    avg_fp4 = bench_method(
        wrapper_fp4_dispatch, x, "FP4-Dispatch FusedMoE (use_fp4_dispatch=True)", ep_rank, profile=False
    )

    # Summary
    if ep_rank == 0:
        speedup = avg_base / avg_fp4 if avg_fp4 > 0 else float("inf")
        print(f"\n{'='*60}")
        print("  Summary")
        print(f"{'='*60}")
        print(f"  Base avg         : {avg_base:.3f} ms")
        print(f"  FP4-Dispatch avg : {avg_fp4:.3f} ms")
        print(f"  Speedup          : {speedup:.3f}x")
        if speedup > 1.0:
            print(f"  FP4-Dispatch is {(speedup - 1) * 100:.1f}% faster")
        else:
            print(f"  FP4-Dispatch is {(1 - speedup) * 100:.1f}% slower")
        print(f"{'='*60}\n")

    # Cleanup
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)


if __name__ == "__main__":
    main()
