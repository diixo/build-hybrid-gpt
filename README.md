# build-hybrid-llama

* run `wikipedia.py` to download and prepare the wikipedia dataset
* run `main_ddp.py` for a distributed data parallel (DDP) pre-training loop across multiple GPUs


Tune params in `main_ddp.py`:
```python
#total_batch_size = 524288 # 2**19, ~0.5M, in number of tokens
#B = 64 # micro batch size
total_batch_size = 540672 # 2**19, ~0.5M, in number of tokens
B = 16 # micro batch size
```

The trained model will be located in `train_products` directory.


## GPT-R (GPT-RoPE)
GPT-R is hybrid of nano-GPT model with RoPE technique.
