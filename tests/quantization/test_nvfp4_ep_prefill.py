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
Test script for ModelOptNvFp4FusedMoE.apply_ep_prefill accuracy.

No mocks - uses real EP communication (DeepEP), real flashinfer CuteDSL kernels,
and real _run_cutedsl_grouped_masked on 8 GPUs.

Configuration:
    - num_experts = 160, EP8 => num_local_experts = 20
    - hidden_size = 7168, moe_intermediate_size = 3584
    - input tokens = 1024, top_k = 4
    - num_worst_tokens > 0

Launch with 8 GPUs:
    PYTHONPATH="" python -m paddle.distributed.launch --gpus 0,1,2,3,4,5,6,7 \\
        tests/quantization/test_nvfp4_ep_prefill.py

Python environment:
    /root/paddlejob/workspace/output/lizexu/miniconda3/envs/whs_fd/bin/python
"""

import json
import os
import shutil
import unittest

# Ensure we use the installed fastdeploy package (with compiled custom ops)
# rather than the source tree, which may not have compiled ops for this GPU arch.
# _SOURCE_TREE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# sys.path = [p for p in sys.path if os.path.abspath(p) != _SOURCE_TREE]

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
# Test parameters
# ---------------------------------------------------------------------------
NUM_EXPERTS = 160
EP_SIZE = 8
NUM_LOCAL_EXPERTS = NUM_EXPERTS // EP_SIZE  # 20
HIDDEN_SIZE = 7168
MOE_INTERMEDIATE_SIZE = 3584
NUM_TOKENS = 1024
TOP_K = 4
GROUP_SIZE = 16


class MockForwardMeta:
    """Minimal ForwardMeta to drive FusedMoE.forward."""

    def __init__(self):
        self.moe_num_chunk = 1
        self.max_moe_num_chunk = 1
        self.routing_replay_table = None


class NvFp4MoEWrapper(paddle.nn.Layer):
    """
    Wrapper that creates a real FusedMoE layer with NVFP4 quantization config
    for EP8 prefill testing. Follows the pattern in tests/layers/test_fusedmoe.py.

    Initializes:
        - FDConfig with real ParallelConfig (EP8, prefill phase)
        - FusedMoE with ModelOptNvFp4Config quant
        - Real ep_prefill_runner via init_ep() (DeepEP buffer)
        - Random NVFP4 weights + process_weights_after_loading
    """

    def __init__(
        self,
        model_config: ModelConfig,
        ep_size: int,
        ep_rank: int,
        prefix: str = "layer0",
        use_fp4_dispatch: bool = False,
    ):
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

        # Use Fleet's model-parallel group as ep_group (same pattern as test_fusedmoe.py)
        self.fd_config.parallel_config.ep_group = fleet.get_hybrid_communicate_group().get_model_parallel_group()
        self.fd_config.scheduler_config.splitwise_role = "prefill"
        self.fd_config.model_config.moe_phase.phase = "prefill"

        weight_key_map = {
            "gate_weight_key": f"{self.prefix}.gate.weight",
            "gate_correction_bias_key": f"{self.prefix}.moe_statics.e_score_correction_bias",
            "up_gate_proj_expert_weight_key": f"{self.prefix}.experts.{{}}.up_gate_proj.weight",
            "down_proj_expert_weight_key": f"{self.prefix}.experts.{{}}.down_proj.weight",
        }

        # Real gating layer
        self.gating = ReplicatedLinear(
            fd_config=self.fd_config,
            prefix=f"{self.prefix}.gate",
            input_size=HIDDEN_SIZE,
            output_size=NUM_EXPERTS,
            with_bias=False,
            skip_quant=True,
            weight_dtype="float32",
        )

        # Real FusedMoE: this calls quant_method.init_ep(self) internally
        # which creates ep_prefill_runner with real DeepEP buffer
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
        """Initialize MoE weights with random values for testing."""
        moe_layer = self.fused_moe
        quant_method = moe_layer.quant_method

        # Fill NVFP4 weights (uint8) with random bytes
        up_gate_w = getattr(moe_layer, quant_method.added_weight_attrs[0])
        down_w = getattr(moe_layer, quant_method.added_weight_attrs[1])
        up_gate_w.set_value(paddle.randint(0, 256, up_gate_w.shape, dtype="int32").cast("uint8"))
        down_w.set_value(paddle.randint(0, 256, down_w.shape, dtype="int32").cast("uint8"))

        # Fill weight_scale (float8_e4m3fn) with small positive values via float32
        up_gate_scale = getattr(moe_layer, quant_method.added_scale_attrs[0])
        down_scale = getattr(moe_layer, quant_method.added_scale_attrs[1])
        up_gate_scale.set_value(paddle.full(up_gate_scale.shape, 0.01, dtype="float32").cast("float8_e4m3fn"))
        down_scale.set_value(paddle.full(down_scale.shape, 0.01, dtype="float32").cast("float8_e4m3fn"))

        # Fill weight_scale_2 and input_scale with 1.0
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

        # Run the real process_weights_after_loading:
        # creates blockscale_swizzled, g1_alphas, g2_alphas, input_scale_quant, etc.
        quant_method.process_weights_after_loading(moe_layer)

        # Init gating weights with random values
        self.gating.weight.set_value(paddle.randn(self.gating.weight.shape, dtype=paddle.float32) * 0.01)


class TestNvFp4EpPrefill(unittest.TestCase):
    """
    Test MoE computation accuracy of ModelOptNvFp4FusedMoE.apply_ep_prefill
    with real 8-GPU EP8. No mocks.

    Uses real:
        - DeepEP for EP dispatch/combine
        - flashinfer CuteDSL for _run_cutedsl_grouped_masked
        - NVFP4 quantized weights
    """

    @classmethod
    def setUpClass(cls):
        """Initialize distributed environment and build model config."""
        init_distributed_environment()

        cls.ep_size = paddle.distributed.get_world_size()
        cls.ep_rank = paddle.distributed.get_rank()

        assert cls.ep_size == EP_SIZE, (
            f"This test requires {EP_SIZE} GPUs, got {cls.ep_size}. "
            f"Launch with: python -m paddle.distributed.launch "
            f"--gpus 0,1,2,3,4,5,6,7 {__file__}"
        )

        # Seed per rank for reproducibility
        paddle.seed(cls.ep_rank + 42)

        cls.model_config = cls._build_model_config()

    @classmethod
    def _build_model_config(cls) -> ModelConfig:
        """Create ModelConfig from a temporary config.json."""
        config_dict = {
            "architectures": ["DeepseekV3ForCausalLM"],
            "hidden_size": HIDDEN_SIZE,
            "moe_intermediate_size": MOE_INTERMEDIATE_SIZE,
            "moe_num_experts": NUM_EXPERTS,
            "moe_k": TOP_K,
            "num_attention_heads": -1,
            "dtype": "bfloat16",
        }

        tmp_dir = f"/tmp/nvfp4_ep_test_rank{cls.ep_rank}"
        os.makedirs(tmp_dir, exist_ok=True)
        config_path = os.path.join(tmp_dir, "config.json")
        with open(config_path, "w") as f:
            json.dump(config_dict, f)

        cls._tmp_dir = tmp_dir
        return ModelConfig({"model": cls._tmp_dir, "max_model_len": 2048})

    @classmethod
    def tearDownClass(cls):
        """Clean up temporary files."""
        if hasattr(cls, "_tmp_dir") and os.path.exists(cls._tmp_dir):
            shutil.rmtree(cls._tmp_dir)

    def test_output_shape_topk4_optimized_path(self):
        """
        Test that apply_ep_prefill produces correct output shape
        with top_k=4, num_worst_tokens > 0, using 8 real GPUs.

        Uses real DeepEP dispatch/combine and real flashinfer CuteDSL kernels.

        Verifies:
        1. Output shape is [NUM_TOKENS, HIDDEN_SIZE]
        2. Output dtype is bfloat16
        3. Output contains non-zero values (data flow correctness)
        4. No NaN or Inf in output (numerical stability)
        """
        wrapper = NvFp4MoEWrapper(
            model_config=self.model_config,
            ep_size=self.ep_size,
            ep_rank=self.ep_rank,
            use_fp4_dispatch=False,
        )
        wrapper_fp4_dispatch = NvFp4MoEWrapper(
            model_config=self.model_config,
            ep_size=self.ep_size,
            ep_rank=self.ep_rank,
            use_fp4_dispatch=True,
        )

        # Synchronize weights: copy all fused_moe and gating weights from
        # wrapper to wrapper_fp4_dispatch so both instances are identical.
        src_state = wrapper.state_dict()
        dst_state = wrapper_fp4_dispatch.state_dict()
        for key in dst_state:
            if key in src_state:
                dst_state[key] = src_state[key]
        wrapper_fp4_dispatch.set_state_dict(dst_state)

        # All ranks use the same input for deterministic EP testing
        paddle.seed(0)
        x = paddle.randn([NUM_TOKENS, HIDDEN_SIZE], dtype="bfloat16")

        base_out = wrapper.fused_moe(x, wrapper.gating, forward_meta=MockForwardMeta())
        out = wrapper_fp4_dispatch.fused_moe(x, wrapper_fp4_dispatch.gating, forward_meta=MockForwardMeta())

        # 1. Shape check
        self.assertEqual(
            list(out.shape),
            [NUM_TOKENS, HIDDEN_SIZE],
            f"[Rank {self.ep_rank}] Output shape mismatch: {list(out.shape)}",
        )

        # 2. Dtype check
        self.assertEqual(
            out.dtype,
            paddle.bfloat16,
            f"[Rank {self.ep_rank}] Output dtype mismatch: {out.dtype}",
        )

        # 3. Non-zero check
        out_f32 = out.cast("float32")
        abs_sum = float(out_f32.abs().sum())
        self.assertGreater(
            abs_sum,
            0.0,
            f"[Rank {self.ep_rank}] Output is all zeros - data flow broken",
        )

        # 4. NaN/Inf check
        has_nan = bool(paddle.isnan(out_f32).any())
        has_inf = bool(paddle.isinf(out_f32).any())
        self.assertFalse(
            has_nan,
            f"[Rank {self.ep_rank}] Output contains NaN values",
        )
        self.assertFalse(
            has_inf,
            f"[Rank {self.ep_rank}] Output contains Inf values",
        )

        # 5. check diff of base_out and out
        diff = paddle.max(paddle.abs(base_out - out))
        self.assertFalse(
            diff.item() > 1e-4,
            f"[Rank {self.ep_rank}] max diff: {diff.item():.4f}",
        )

        print(f"[Rank {self.ep_rank}] PASSED: shape={list(out.shape)}, " f"dtype={out.dtype}, abs_sum={abs_sum:.4f}")


if __name__ == "__main__":
    unittest.main()
