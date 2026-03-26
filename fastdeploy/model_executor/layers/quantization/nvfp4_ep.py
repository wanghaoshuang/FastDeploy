"""
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

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Callable

import paddle
from paddle import nn
from paddleformers.utils.log import logger

import fastdeploy
from fastdeploy import envs
from fastdeploy.model_executor.layers.moe.fused_moe_backend_base import MoEMethodBase
from fastdeploy.model_executor.utils import (
    create_parameter_and_copy,
    free_tensor,
    set_weight_attrs,
)

# Import helper functions and config from nvfp4.
# nvfp4.py appends `from .nvfp4_ep import ModelOptNvFp4FusedMoEDispatch` at its
# end (after all helpers are defined), so these names are available in the
# partially-loaded module when this module is first imported.
from .nvfp4 import (
    ModelOptNvFp4Config,
    _get_cute_dtype,
    _perm,
    _process_scale_interleaved,
    call_depermute_prefill_combine,
    call_prefill_permute_to_masked_gemm,
    next_power_of_2,
)

if TYPE_CHECKING:
    pass


class ModelOptNvFp4FusedMoEDispatch(MoEMethodBase):
    """Fused MoE method for Model Optimizer NVFP4.
    Supports loading NVFP4 checkpoints with the following structure:

    input_scale: paddle.float32, scalar ,
    weight: NVFP4(represented as byte) Shape: [1, X, y/2]
    weight_scale: FP8-E4M3, Shape: [X, Y], aka per block scale,
    weight_scale_2: paddle.float32, scalar,
    Args:
    quant_config: The ModelOpt quantization config.
    moe_config: The MoE configuration.
    layer: The linear layer.
    """

    def __init__(self, quant_config: ModelOptNvFp4Config):
        self.quant_config = quant_config
        self.added_weight_attrs = ["up_gate_proj_weight", "down_proj_weight"]
        self.added_scale_attrs = [
            "up_gate_proj_weight_scale",
            "down_proj_weight_scale",
        ]
        self.backend = "none"

        if envs.FD_MOE_BACKEND is None:
            # currently support flashinfer-cutlass,  flashinfer-trtllm will support in the future
            self.backend = "flashinfer-cutlass"
        elif envs.FD_MOE_BACKEND.startswith("flashinfer-"):
            self.backend = envs.FD_MOE_BACKEND

        if self.backend == "none":
            raise ValueError(
                "No valid NVFP4 flashinfer MoE backend found. Please check your platform capability and installtion of FlashInfer."
            )

        logger.info(f"Using {self.backend} for NVFP4 FusedMoE")

    def create_weights(self, layer, **extra_weight_attrs):
        """
        NVFP4 MoE create weight.
        """
        self.up_gate_proj_weight_shape = [
            layer.num_local_experts,
            layer.moe_intermediate_size * 2,
            layer.hidden_size // 2,
        ]
        self.down_proj_weight_shape = [
            layer.num_local_experts,
            layer.hidden_size,
            layer.moe_intermediate_size // 2,
        ]
        self.up_gate_proj_scale_shape = self.up_gate_proj_weight_shape[0:2] + [
            layer.hidden_size // self.quant_config.group_size
        ]
        self.down_proj_scale_shape = self.down_proj_weight_shape[0:2] + [
            layer.moe_intermediate_size // self.quant_config.group_size
        ]

        self.weight_scale_dtype = paddle.float8_e4m3fn
        self.weight_dtype = paddle.uint8
        up_gate_proj_weight_name = self.added_weight_attrs[0]
        down_proj_weight_name = self.added_weight_attrs[1]
        up_gate_proj_scale_name = self.added_scale_attrs[0]
        down_proj_scale_name = self.added_scale_attrs[1]
        setattr(
            layer,
            up_gate_proj_weight_name,
            layer.create_parameter(
                shape=self.up_gate_proj_weight_shape,
                dtype=self.weight_dtype,
                default_initializer=paddle.nn.initializer.Constant(0),
            ),
        )
        setattr(
            layer,
            down_proj_weight_name,
            layer.create_parameter(
                shape=self.down_proj_weight_shape,
                dtype=self.weight_dtype,
                default_initializer=paddle.nn.initializer.Constant(0),
            ),
        )
        # weight_scale
        setattr(
            layer,
            up_gate_proj_scale_name,
            layer.create_parameter(
                shape=self.up_gate_proj_scale_shape,
                dtype=self.weight_scale_dtype,
                default_initializer=paddle.nn.initializer.Constant(0),
            ),
        )
        setattr(
            layer,
            down_proj_scale_name,
            layer.create_parameter(
                shape=self.down_proj_scale_shape,
                dtype=self.weight_scale_dtype,
                default_initializer=paddle.nn.initializer.Constant(0),
            ),
        )
        # weight_scale_2
        layer.up_gate_proj_weight_scale_2 = layer.create_parameter(
            shape=[layer.num_local_experts, 2],
            dtype="float32",
            default_initializer=paddle.nn.initializer.Constant(0),
        )
        layer.down_proj_weight_scale_2 = layer.create_parameter(
            shape=[layer.num_local_experts],
            dtype="float32",
            default_initializer=paddle.nn.initializer.Constant(0),
        )
        # input_scale
        layer.up_gate_proj_input_scale = layer.create_parameter(
            shape=[layer.num_local_experts, 2],
            dtype="float32",
            default_initializer=paddle.nn.initializer.Constant(0),
        )
        layer.down_proj_input_scale = layer.create_parameter(
            shape=[layer.num_local_experts],
            dtype="float32",
            default_initializer=paddle.nn.initializer.Constant(0),
        )

        set_weight_attrs(
            getattr(layer, up_gate_proj_weight_name),
            {**extra_weight_attrs, "SHARD_ID_TO_SHARDED_DIM": {"gate": 0, "down": 1, "up": 0}},
        )
        set_weight_attrs(
            getattr(layer, up_gate_proj_scale_name),
            {**extra_weight_attrs, "SHARD_ID_TO_SHARDED_DIM": {"gate": 0, "down": 1, "up": 0}},
        )

        set_weight_attrs(
            getattr(layer, down_proj_weight_name),
            {**extra_weight_attrs, "SHARD_ID_TO_SHARDED_DIM": {"gate": 0, "down": 1, "up": 0}},
        )
        set_weight_attrs(
            getattr(layer, down_proj_scale_name),
            {**extra_weight_attrs, "SHARD_ID_TO_SHARDED_DIM": {"gate": 0, "down": 1, "up": 0}},
        )

        set_weight_attrs(layer.up_gate_proj_weight_scale_2, {**extra_weight_attrs, "weight_type": "weight_scale_2"})
        set_weight_attrs(layer.down_proj_weight_scale_2, {**extra_weight_attrs, "weight_type": "weight_scale_2"})
        set_weight_attrs(layer.up_gate_proj_input_scale, {**extra_weight_attrs, "weight_type": "input_scale"})
        set_weight_attrs(layer.down_proj_input_scale, {**extra_weight_attrs, "weight_type": "input_scale"})

    def process_weights_after_loading(self, layer):
        """ """

        # FlashInfer CUTLASS kernel assumes [Up, Gate] Proj as W13

        if self.backend == "flashinfer-cutlass":
            [a, b] = layer.up_gate_proj_weight.split(2, axis=1)
            layer.up_gate_proj_weight.set_value(paddle.concat([b, a], axis=1))
            [a, b] = layer.up_gate_proj_weight_scale.split(2, axis=1)
            layer.up_gate_proj_weight_scale.set_value(paddle.concat([b, a], axis=1))

        up_gate_proj_weight_scale_2 = layer.up_gate_proj_weight_scale_2[:, 0]
        free_tensor(layer.up_gate_proj_weight_scale_2)
        create_parameter_and_copy(layer, name="up_gate_proj_weight_scale_2", weight=up_gate_proj_weight_scale_2)
        up_gate_proj_input_scale = paddle.max(layer.up_gate_proj_input_scale).cast("float32")
        down_proj_input_scale = paddle.max(layer.down_proj_input_scale).cast("float32")

        # Create shared parameters
        create_parameter_and_copy(
            layer, "g1_alphas", (up_gate_proj_input_scale * up_gate_proj_weight_scale_2).cast("float32")
        )
        create_parameter_and_copy(
            layer, "g2_alphas", (down_proj_input_scale * layer.down_proj_weight_scale_2).cast("float32")
        )
        create_parameter_and_copy(
            layer, "up_gate_proj_input_scale_quant", (1 / up_gate_proj_input_scale).cast("float32")
        )
        create_parameter_and_copy(layer, "down_proj_input_scale_quant", (1 / down_proj_input_scale).cast("float32"))

        for name, weight_scale in [
            ("up_gate", layer.up_gate_proj_weight_scale),
            ("down", layer.down_proj_weight_scale),
        ]:
            assert weight_scale.shape[2] % 16 == 0, f"Expected {name}_weight_scale.dim(2) to be divisible by 16"
            assert (
                weight_scale.dtype == paddle.float8_e4m3fn
            ), f"{name} Weight Blockscale must be represented as FP8-E4M3"

        up_gate_proj_blockscale_swizzled = _process_scale_interleaved(layer.up_gate_proj_weight_scale)
        free_tensor(layer.up_gate_proj_weight_scale)
        layer.up_gate_proj_weight_scale = None
        create_parameter_and_copy(
            layer, name="up_gate_proj_blockscale_swizzled", weight=up_gate_proj_blockscale_swizzled
        )
        down_proj_blockscale_swizzled = _process_scale_interleaved(layer.down_proj_weight_scale)
        free_tensor(layer.down_proj_weight_scale)
        layer.down_proj_weight_scale = None
        create_parameter_and_copy(layer, name="down_proj_blockscale_swizzled", weight=down_proj_blockscale_swizzled)

    def _run_cutedsl_grouped_masked(self, layer, hidden_states_3d, masked_m):

        if self.backend != "flashinfer-cutedsl":
            raise NotImplementedError("NVFP4 EP backend only supports CuteDSL implementation.")

        # flashinfer cutedsl blockscaled gemm takes long time to complie, it may not be imported in function.
        # we will add flashinfer.cutedsl.blockscaled_gemm into setup.py by AOT.
        from flashinfer import (
            scaled_fp4_grouped_quantize,
            silu_and_mul_scaled_nvfp4_experts_quantize,
        )
        from flashinfer.cute_dsl.blockscaled_gemm import grouped_gemm_nt_masked

        masked_m = masked_m.cast(paddle.int32)
        num_experts = int(layer.num_local_experts)

        def _to_expert_scale_vec(scale: paddle.Tensor, name: str) -> paddle.Tensor:
            scale = scale.cast("float32")
            if len(scale.shape) == 0:
                return paddle.ones([num_experts], dtype="float32") * scale
            if len(scale.shape) == 1:
                if scale.shape[0] == num_experts:
                    return scale
                if scale.shape[0] == 1:
                    return paddle.tile(scale, [num_experts])
                raise ValueError(f"{name} shape mismatch: {scale.shape}, expected ({num_experts},)")
            if len(scale.shape) == 2 and scale.shape[1] == 2:
                return scale.max(axis=1).values.cast("float32")
            raise ValueError(f"{name} rank not supported: shape={scale.shape}")

        w1_alpha = _to_expert_scale_vec(layer.g1_alphas, "g1_alphas")
        w2_alpha = _to_expert_scale_vec(layer.g2_alphas, "g2_alphas")
        input_global_scale = _to_expert_scale_vec(
            layer.up_gate_proj_input_scale_quant, "up_gate_proj_input_scale_quant"
        )
        a2_global_scale = _to_expert_scale_vec(layer.down_proj_input_scale_quant, "down_proj_input_scale_quant")

        n = layer.down_proj_weight.shape[-1] * 2

        if isinstance(hidden_states_3d, tuple) and hidden_states_3d[1] is not None:
            a_q = hidden_states_3d[0].view(paddle.uint8)
            m, k_by_2, _ = a_q.shape
            k = k_by_2 * 2

            # hidden_states_3d[1] is non-swizzled scale: [M, scale_dim, E] float8_e4m3fn
            # grouped_gemm_nt_masked expects swizzled layout, so we must swizzle here.
            from flashinfer import block_scale_interleave

            sf_vec_size_local = 16
            scale_dim = k // sf_vec_size_local  # e.g. 7168/16 = 448

            # Transpose to [E, M, scale_dim] and view as uint8 for block_scale_interleave
            a_q_sf_ems = hidden_states_3d[1].transpose([2, 0, 1]).contiguous().view(paddle.uint8)
            # block_scale_interleave expects [E, M, scale_dim] uint8, returns flat swizzled buffer
            swizzled_flat = block_scale_interleave(a_q_sf_ems)

            # Reshape swizzled output to match scaled_fp4_grouped_quantize format:
            # physical (E, rm, rk, 32, 4, 4) -> logical (32, 4, rm, 4, rk, E)
            padded_k = (scale_dim + 3) // 4 * 4
            padded_m = (m + 127) // 128 * 128
            a_q_sf = (
                swizzled_flat.view(paddle.float8_e4m3fn)
                .reshape([num_experts, padded_m // 128, padded_k // 4, 32, 4, 4])
                .transpose([3, 4, 1, 5, 2, 0])
            )
        else:
            hidden_states = hidden_states_3d[0] if isinstance(hidden_states_3d, tuple) else hidden_states_3d
            _, m, k = hidden_states.shape
            a_q, a_q_sf = scaled_fp4_grouped_quantize(hidden_states, masked_m, input_global_scale)

        ab_dtype = "float4_e2m1fn"
        sf_dtype = "float8_e4m3fn"
        c_dtype = "bfloat16"
        sf_vec_size = 16

        gateup_output = paddle.empty([num_experts, m, n * 2], dtype=paddle.bfloat16).transpose([1, 2, 0])
        grouped_gemm_nt_masked(
            (a_q, a_q_sf),
            (_perm(layer.up_gate_proj_weight, 1, 2, 0), layer.up_gate_proj_blockscale_swizzled),
            gateup_output,
            masked_m,
            ab_dtype=ab_dtype,
            sf_dtype=sf_dtype,
            c_dtype=c_dtype,
            sf_vec_size=sf_vec_size,
            alpha=w1_alpha.reshape([1, 1, num_experts]),
            alpha_dtype=_get_cute_dtype(w1_alpha),
        )

        diq, diq_sf = silu_and_mul_scaled_nvfp4_experts_quantize(
            gateup_output.transpose([2, 0, 1]),
            masked_m,
            a2_global_scale,
        )

        out = paddle.empty([num_experts, m, k], dtype=paddle.bfloat16).transpose([1, 2, 0])
        grouped_gemm_nt_masked(
            (diq, diq_sf),
            (_perm(layer.down_proj_weight, 1, 2, 0), layer.down_proj_blockscale_swizzled),
            out,
            masked_m,
            ab_dtype=ab_dtype,
            sf_dtype=sf_dtype,
            c_dtype=c_dtype,
            sf_vec_size=sf_vec_size,
            alpha=w2_alpha.reshape([1, 1, num_experts]),
            alpha_dtype=_get_cute_dtype(w2_alpha),
        )

        return out.transpose([2, 0, 1])

    def apply_ep_prefill(
        self,
        layer: nn.Layer,
        x: paddle.Tensor,
        gate: nn.Layer,
        topk_ids_hookfunc: Callable = None,
        shared_experts: nn.Layer = None,
    ) -> paddle.Tensor:

        # 1. top experts and weights
        gate_out = gate(x.cast("float32"))
        topk_idx, topk_weights = self.ep_prefill_runner.moe_select(layer, gate_out)

        hidden_size = x.shape[1]

        if topk_ids_hookfunc is not None:
            topk_ids_hookfunc(topk_ids=topk_idx)

        from fastdeploy.model_executor.layers.moe.ep import deep_ep

        # 1.5. Quantize x to NVFP4 before dispatch
        # Compute a single global input scale (same for all experts in modelopt calibration)
        input_scale_raw = layer.up_gate_proj_input_scale_quant.cast("float32")
        if len(input_scale_raw.shape) == 0:
            input_global_scale_single = input_scale_raw.reshape([1])
        elif len(input_scale_raw.shape) == 1:
            input_global_scale_single = input_scale_raw[:1]
        elif len(input_scale_raw.shape) == 2 and input_scale_raw.shape[1] == 2:
            input_global_scale_single = input_scale_raw.max(axis=1).values[:1].cast("float32")
        else:
            input_global_scale_single = input_scale_raw.reshape([-1])[:1]

        from flashinfer import fp4_quantize

        x_q_2d, x_q_sf_2d = fp4_quantize(x, input_global_scale_single, sf_vec_size=16, is_sf_swizzled_layout=False)
        # x_q_2d layout: [num_tokens, hidden_packed], x_q_sf_2d layout: [num_tokens, sf_packed]

        # Pack 4 uint8 scale factors into float32 (bit-preserving) to satisfy
        # Deep EP's float32 dtype requirement without increasing communication volume.
        x_q_sf_2d = x_q_sf_2d.view(paddle.float32)

        event = deep_ep.Buffer.capture()

        # 2. ep dispatch (dispatching NVFP4 quantized data)
        (
            recv_x,
            recv_topk_idx,
            recv_topk_weights,
            recv_num_tokens_per_expert_list,
            handle,
            event,
        ) = self.ep_prefill_runner.dispatch(
            x_q_2d,
            topk_idx,
            topk_weights,
            x_scale_tensor=x_q_sf_2d,
            expert_alignment=128,
            previous_event=event,
        )

        if self.ep_prefill_runner.ep_engine.async_finish:
            event.current_stream_wait()

        # NVFP4 dispatch returns a tuple (quantized_value, block_scales)
        if isinstance(recv_x, tuple):
            recv_x_value, recv_x_scale = recv_x
            # recv_x_value is uint8 (FP4 packed); view as float8_e4m3fn so that
            # prefill_permute_to_masked_gemm CUDA kernel can dispatch correctly.
            recv_x_value = recv_x_value.view(paddle.float8_e4m3fn)
            # recv_x_scale stays float32 (packed from 4 uint8 before dispatch);
            # prefill_permute_to_masked_gemm supports float32 scale dtype.
        else:
            recv_x_value = recv_x
            recv_x_scale = None

        # 3. compute ffn
        if self.ep_prefill_runner.num_worst_tokens > 0:
            top_k = layer.top_k
            num_local_experts = layer.num_local_experts

            if top_k in (4, 8):
                token_split_factor = 2 if int(os.getenv("USE_TBO", "0")) == 1 else 1
                max_tokens_per_rank = (
                    layer.fd_config.scheduler_config.max_num_batched_tokens
                    // layer.fd_config.parallel_config.tensor_parallel_size
                    // token_split_factor
                )

                if recv_x_scale is None:
                    recv_x_scale = paddle.zeros([recv_x_value.shape[0], 1], dtype=paddle.int32)

                permute_input, permute_scale, permuted_indice_map, token_nums_per_expert = (
                    call_prefill_permute_to_masked_gemm(
                        x=recv_x_value,
                        scale=recv_x_scale,
                        topk_ids=recv_topk_idx,
                        num_local_experts=num_local_experts,
                        max_token_num=layer.ep_size * max_tokens_per_rank,
                    )
                )

                # Transpose from [num_experts, max_tokens, dim] to [max_tokens, dim, num_experts]
                # to match the layout expected by grouped_gemm_nt_masked
                permute_input_t = permute_input.transpose([1, 2, 0])
                # Unpack float32 scale back to float8_e4m3fn (bit-preserving) before
                # passing to grouped_gemm which expects 1-byte scale factors.
                # Must view before transpose so the scale dim (not expert dim) is expanded.
                permute_scale_t = permute_scale.contiguous().view(paddle.float8_e4m3fn).transpose([1, 2, 0])

                # token_nums_per_expert: [num_local_experts, 1] -> [num_local_experts]
                # Pass pre-quantized NVFP4 data as tuple to skip quantization inside
                ffn_out = self._run_cutedsl_grouped_masked(
                    layer, (permute_input_t, permute_scale_t), token_nums_per_expert.reshape([-1])
                )

                tmp_ffn_out = call_depermute_prefill_combine(
                    x=ffn_out,
                    indice_map=permuted_indice_map,
                    topk_weights=recv_topk_weights,
                    num_worst_tokens=recv_x_value.shape[0],
                )
            else:
                expert_token_lists = [[] for _ in range(num_local_experts)]
                recv_n = recv_x_value.shape[0]
                recv_topk_idx_numpy = recv_topk_idx.numpy()  # [N_recv, top_k]
                for ti in range(recv_n):
                    for ki in range(top_k):
                        ei = int(recv_topk_idx_numpy[ti, ki])
                        if ei >= 0:
                            expert_token_lists[ei].append(ti)

                token_counts = [len(lst) for lst in expert_token_lists]
                max_expert_tokens = max(token_counts) if token_counts else 0

                # Build blocked input: [num_local_experts, max_expert_tokens, H]
                H = recv_x_value.shape[1]
                if max_expert_tokens > 0:
                    blocked = paddle.zeros(
                        [num_local_experts, max_expert_tokens, H],
                        dtype=recv_x_value.dtype,
                    )
                    H_sf = recv_x_scale.shape[1] if recv_x_scale is not None else 0
                    blocked_scale = (
                        paddle.zeros(
                            [num_local_experts, max_expert_tokens, H_sf],
                            dtype=recv_x_scale.dtype,
                        )
                        if recv_x_scale is not None
                        else None
                    )
                    for ei, indices in enumerate(expert_token_lists):
                        if indices:
                            idx_tensor = paddle.to_tensor(indices, dtype=paddle.int64)
                            blocked[ei, : len(indices)] = recv_x_value[idx_tensor]
                            if blocked_scale is not None:
                                blocked_scale[ei, : len(indices)] = recv_x_scale[idx_tensor]
                    masked_m = paddle.to_tensor(token_counts, dtype=paddle.int32)

                    # Transpose from [experts, tokens, dim] to [tokens, dim, experts]
                    blocked_t = blocked.transpose([1, 2, 0])
                    # Unpack float32 scale back to float8_e4m3fn (bit-preserving).
                    # view before transpose so the scale dim is expanded correctly.
                    blocked_scale_t = (
                        blocked_scale.contiguous().view(paddle.float8_e4m3fn).transpose([1, 2, 0])
                        if blocked_scale is not None
                        else None
                    )

                    ffn_out_blocked = self._run_cutedsl_grouped_masked(layer, (blocked_t, blocked_scale_t), masked_m)
                    # ffn_out_blocked: [num_local_experts, max_expert_tokens, H_out]

                    # De-permute: accumulate weighted expert outputs back to
                    # [N_recv, H_out] in-place.
                    H_out = ffn_out_blocked.shape[2]
                    tmp_ffn_out = paddle.zeros([recv_n, H_out], dtype=paddle.float32)
                    recv_topk_weights_np = recv_topk_weights.cast(paddle.float32).numpy()
                    ffn_out_f32 = ffn_out_blocked.cast(paddle.float32)
                    for ti in range(recv_n):
                        for ki in range(top_k):
                            ei = int(recv_topk_idx_numpy[ti, ki])
                            if ei >= 0:
                                slot = expert_token_lists[ei].index(ti)
                                w = float(recv_topk_weights_np[ti, ki])
                                tmp_ffn_out[ti] += w * ffn_out_f32[ei, slot]
                    tmp_ffn_out = tmp_ffn_out.cast(paddle.bfloat16)
                else:
                    tmp_ffn_out = paddle.zeros([recv_n, H], dtype=paddle.bfloat16)

        else:
            tmp_ffn_out = paddle.empty([0, hidden_size], paddle.bfloat16)
            logger.warning("num_worst_tokens is disabled!")

        if shared_experts is not None:
            s_x = shared_experts(x)

        # 4. EP combine
        event = deep_ep.Buffer.capture()
        tmp_ffn_out, event = self.ep_prefill_runner.combine(tmp_ffn_out, handle, recv_topk_weights, event)

        if self.ep_prefill_runner.ep_engine.async_finish:
            event.current_stream_wait()

        if shared_experts is not None:
            tmp_ffn_out += s_x

        return tmp_ffn_out

    def apply_ep_decode(
        self,
        layer: nn.Layer,
        x: paddle.Tensor,
        gate: nn.Layer,
        topk_ids_hookfunc: Callable = None,
        shared_experts: nn.Layer = None,
    ) -> paddle.Tensor:
        if layer.fd_config.parallel_config.use_internode_ll_two_stage:
            raise NotImplementedError("NVFP4 CuteDSL EP decode does not support DeepEP two-stage low-latency.")

        gate_out = gate(x.cast("float32"))
        topk_idx, topk_weights = self.ep_decoder_runner.moe_select(layer, gate_out)

        if topk_ids_hookfunc is not None:
            topk_ids_hookfunc(topk_ids=topk_idx)

        recv_x, token_nums_per_expert, handle = self.ep_decoder_runner.dispatch(
            x,
            topk_idx,
            topk_weights,
            use_fp8=False,
        )

        ffn_out = self._run_cutedsl_grouped_masked(layer, recv_x, token_nums_per_expert)

        if shared_experts is not None:
            s_x = shared_experts(x)

        out = self.ep_decoder_runner.combine(ffn_out, topk_idx, topk_weights, handle)

        if shared_experts is not None:
            out += s_x

        return out

    def apply_tp(
        self,
        layer: nn.Layer,
        x: paddle.Tensor,
        gate: nn.Layer,
        topk_ids_hookfunc: Callable = None,
    ) -> paddle.Tensor:
        pass

    def apply(
        self,
        layer,
        x,
        gate,
        topk_ids_hookfunc: Callable = None,
        shared_experts: nn.Layer = None,
    ):
        """
        flashinfer nvfp4 fusedmoe for Model Optimizer
        """
        if self.backend == "flashinfer-cutlass":
            gate_out = gate(x.cast("float32"))
            topk_ids, topk_weights = fastdeploy.model_executor.ops.gpu.moe_topk_select(
                gate_out,
                layer.gate_correction_bias,
                layer.top_k,
                True,  # apply_norm_weight,
                False,
            )

            if topk_ids_hookfunc is not None:
                topk_ids_hookfunc(topk_ids)

            output_dtype = x.dtype
            x_sf = None
            output = paddle.empty_like(x)

            # flashinfer cutlass
            from flashinfer.fused_moe import (
                cutlass_fused_moe as flashinfer_cutlass_fused_moe,
            )

            _ = flashinfer_cutlass_fused_moe(
                input=x,
                token_selected_experts=topk_ids.to(paddle.int),
                token_final_scales=topk_weights,
                fc1_expert_weights=getattr(layer, self.added_weight_attrs[0]).view(paddle.long),
                fc2_expert_weights=getattr(layer, self.added_weight_attrs[1]).view(paddle.long),
                output_dtype=output_dtype,
                input_sf=x_sf,
                quant_scales=[
                    layer.up_gate_proj_input_scale_quant,
                    layer.up_gate_proj_blockscale_swizzled.view(paddle.int32),
                    layer.g1_alphas,
                    layer.down_proj_input_scale_quant,
                    layer.down_proj_blockscale_swizzled.view(paddle.int32),
                    layer.g2_alphas,
                ],
                ep_size=layer.ep_size,
                ep_rank=layer.ep_rank,
                tp_size=layer.tp_size,
                tp_rank=layer.tp_rank,
                tune_max_num_tokens=next_power_of_2(x.shape[0]),
                output=output,
            )

            return output

        elif self.backend == "flashinfer-cutedsl" and layer.ep_size > 1:
            if layer.fd_config.model_config.moe_phase.phase == "prefill":
                return self.apply_ep_prefill(
                    layer, x, gate, topk_ids_hookfunc=topk_ids_hookfunc, shared_experts=shared_experts
                )
            else:
                return self.apply_ep_decode(
                    layer, x, gate, topk_ids_hookfunc=topk_ids_hookfunc, shared_experts=shared_experts
                )

        # flashinfer-trtllm
        return paddle.empty_like(x)
