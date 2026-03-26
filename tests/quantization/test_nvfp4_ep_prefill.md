为ModelOptNvFp4FusedMoE的apply_ep_prefill生成一个测试脚本，测试MoE的计算准确性。
1. MoE layer的权重随机初始化；
2. 权重：专家数=160; hidden_size=7168; moe_intermediate_size=3584
3. 输入激活: 输入tokens数量=1K; hidden_size=7168; MoE topk=4
4. EP8专家并行，每张卡20个专家。
5. num_worst_tokens > 0
