#PYTHONPATH=/root/paddlejob/workspace/output/lizexu/FastDeploy:$PYTHONPATH
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
FD_MOE_BACKEND=flashinfer-cutedsl
python \
    -m paddle.distributed.launch \
    --gpus 0,1,2,3,4,5,6,7 \
    tests/quantization/test_nvfp4_ep_prefill.py
