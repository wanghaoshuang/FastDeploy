CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
FD_MOE_BACKEND=flashinfer-cutedsl
NCCL_DEBUG=INFO
python \
    -m paddle.distributed.launch \
    --gpus 0,1,2,3,4,5,6,7 \
    bench_nvfp4_ep_prefill.py
