# tgi_client_gradio

```sh
docker run -it --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --device=/dev/kfd -p 9000:80 --device=/dev/dri --group-add video --ipc=host --shm-size 256g -v $(pwd):/data --entrypoint /bin/bash --env HUGGINGFACE_HUB_CACHE=/data --name tgi_wip_demo2 tgi_wip_0902:latest
```

```sh
PYTORCH_TUNABLEOP_ENABLED=1 ROCM_USE_FLASH_ATTN_V2_TRITON=0 text-generation-launcher --model-id meta-llama/Meta-Llama-3.1-405B-Instruct --num-shard 8 --cuda-graphs 1 --max-batch-prefill-tokens 131072 --max-total-tokens 8192 &
```

```sh
what is the fastest way from SFO to san jose convention center? 

distinguish bears from racoon

write me a Verilog code that describes 3:2 multiplexer
```
