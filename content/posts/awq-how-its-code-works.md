---
title: "AWQ: How Its Code Works"
date: 2024-04-04
draft: false
math: false
---

#### A walkthrough of the AutoAWQ library

Memory is king.

Modern large language models (LLMs), with almost no exception, are transformer-based models. It means the inference speed of LLM is typically determined by the speed of memory loading and storing. Also, a larger memory allows for a broader context window due to the use of KV cache.

However, large language models by its name have a huge size of parameters, which place a heavy burden on memory and cause low efficiency. This makes GPUs with large and fast memory, like the H100, popular and hence expensive. That’s how the market works.

The good news is we can significantly reduce the size of model parameters, typically by a factor of 4, simply by converting them to lower precisions. This method is called quantization. LLMs are robust to noise, and therefore suffer little degradation after being quantized.

There are many quantization methods. Activation-aware Weight Quantization (AWQ) is one of them. It is easy to use, easy to understand, and easy to modify.

### How to Use AWQ

AutoAWQ is a handy implementation of AWQ. We can install it using pip.

```
pip install autoawq
```

Import the package. Get the path of your Hugging Face model and put it in.

```
from awq import AutoAWQForCausalLM  
model = AutoAWQForCausalLM.from_pretrained(model_dir)
```

Optionally, if you want to have your own quantization configuration other than the default one, you can create it here. For example, we could use a group size of 64 instead of the default 128.

```
my_quant_config=dict(q_group_size=64)
```

Then we start to quantize the model.

```
from transformers import AutoTokenizer  
tokenizer = AutoTokenizer.from_pretrained(model_dir)  
model.quantize(  
    tokenizer,  
    quant_config=my_quant_config,  
)
```

It will run for about 20 minutes depending on your GPU and model size. Then, we can save the quantized model.

```
model.save_quantized(output_dir)
```

That’s it. The quantization is done. When you want to use your quantized model, just load it like it is a normal Hugging Face model. I’ve tried using it in vLLM and it works well. No need to change any setting. vLLM can automatically detect that it is a quantized model and run it in the right way.

### AutoAWQ Code Walkthrough

Now it’s time to take a deeper look inside the library.

By running the *from\_pretrained* method, it detects what structure the model is and creates a corresponding AWQ model. The supported model structures are listed below.

```
AWQ_CAUSAL_LM_MODEL_MAP = {  
    "mpt": MptAWQForCausalLM,  
    "llama": LlamaAWQForCausalLM,  
    "opt": OptAWQForCausalLM,  
    "RefinedWeb": FalconAWQForCausalLM,  
    "RefinedWebModel": FalconAWQForCausalLM,  
    "falcon": FalconAWQForCausalLM,  
    "bloom": BloomAWQForCausalLM,  
    "gptj": GPTJAWQForCausalLM,  
    "gpt_bigcode": GptBigCodeAWQForCausalLM,  
    "mistral": MistralAWQForCausalLM,  
    "mixtral": MixtralAWQForCausalLM,  
    "gpt_neox": GPTNeoXAWQForCausalLM,  
    "aquila": AquilaAWQForCausalLM,  
    "Yi": YiAWQForCausalLM,  
    "qwen": QwenAWQForCausalLM,  
    "baichuan": BaichuanAWQForCausalLM,  
    "llava": LlavaAWQForCausalLM,  
    "qwen2": Qwen2AWQForCausalLM,  
    "gemma": GemmaAWQForCausalLM,  
}
```

As of writing, AutoAWQ supports 19 different model structures. Without loss of generalization, we choose *Qwen* as our example.

The 19 classes all inherit from the *BaseAWQForCausalLM* class, and differ only in trivial things like parameter names. For example, *QwenAWQForCausalLM* has a *get\_model\_layers* method that looks like this,

```
@staticmethod  
def get_model_layers(model):  
    return model.transformer.h
```

while in *LlamaAWQForCausalLM* it is this.

```
@staticmethod  
def get_model_layers(model):  
    return model.model.layers
```

This method is just an interface to get the main body of the model, the transformer layers.

After creating a *QwenAWQForCausalLM* instance, it loads the original model by calling the *from\_pretrained* method from the *transformers* library. Next, it creates a default quantization configuration as below.

```
AwqConfig(  
    quant_method="awq",  
    zero_point=True,  
    q_group_size=128,  
    w_bit=4,  
    version="GEMM",  
    modules_to_not_convert=None,  
)
```

Now that the floating point model is duly loaded, it is time to call the *quantize*method.

First, it creates a *AwqQuantizer* instance from the configuration we provide or the default one. In its initialization it does two things. Get some input data from the calibration dataset and process them for the transformer layers.

If we do not provide our own dataset, AutoAWQ will use the *pile-val* dataset. It reads *n\_samples* (which is 512 by default) samples from the dataset, concatenates them to a single long sequence, and then chunk it to equally long subsequences each of which has *block\_size* (which is also 512 by default) tokens. Note that in this case it is fairly possible that one input sentence is split into two samples, and that one sample is made of many independent, semantically irrelevant sentences.

These samples cannot be used as input for the transformer layers at this point. They need to be embedded and may also undergo other processing steps. AutoAWQ creates a catcher to catch the input after the embedding layer and other preprocessing in a test run. It does so by patching the first transformer layer and remembering its input.

```
class Catcher(nn.Module):  
    def __init__(self, module):  
        super().__init__()  
        self.module = module  
  
    def forward(self, *args, **kwargs):  
        # assume first input to forward is hidden states  
        if len(args) > 0:  
            hidden_states = args[0]  
            del args  
        else:  
            first_key = list(kwargs.keys())[0]  
            hidden_states = kwargs.pop(first_key)  
  
        inps.append(hidden_states)  
        layer_kwargs.update(kwargs)  
        raise ValueError  # early exit to break later inference
```

It runs the model using the calibration data. The catcher will catch the input for the first transformer layer.

Once the data is ready, the quantizer has completed its initialization. Next, AutoAWQ calls the *quantize* method of the quantizer. The method is essentially a giant *for loop* that iterates over all the transformer layers of the model, handling each layer independently. That is a great feature because only one layer stays in the GPU memory throughout the process. It means you can run the quantization in a tiny GPU. Next we examine what the loop does to each layer.

### The Loop

#### Find linear modules and extract their input features

```
named_linears = get_named_linears(self.modules[i])  
  
# Filter out the linear layers we don't want to exclude  
named_linears = exclude_layers_to_not_quantize(  
    named_linears, self.modules_to_not_convert  
)  
  
input_feat = self._get_input_feat(self.modules[i], named_linears)
```

First step, it finds all the linear modules in a layer. Typically, there are four. Two in the attention block, and two in the MLP block. In our example, the *Qwen* model, there are five linear modules because of the use of SwiGLU activations. They are listed below.

```
{'attn.c_attn': Linear(in_features=5120, out_features=15360, bias=True),  
  
'attn.c_proj': Linear(in_features=5120, out_features=5120, bias=False),  
  
'mlp.c_proj': Linear(in_features=13696, out_features=5120, bias=False),  
  
'mlp.w1': Linear(in_features=5120, out_features=13696, bias=False),  
  
'mlp.w2': Linear(in_features=5120, out_features=13696, bias=False)}
```

Then, some layers are excluded from being quantized if the user specifies, perhaps to preserve better accuracy.

Next, it inserts a hook into every linear module to collect input features for each of them. In our case, we obtain five input features corresponding to the five linear modules.

#### Get the best scales

The rationale behind AWQ is simple. We scale up some important weights so that they can preserve better precision during the quantization. To correct for the weight upscaling, the weights of the previous module, such as LayerNorm, are also modified.

We determine the importance of each weight channel based on the magnitude of its corresponding input channel. Every channel of weights is assigned a “factor of importance”, or scale. According to the paper, the scale is solely determined by the input. However, AutoAWQ adds a second option to consider both the magnitudes of the input and the weight itself. We will follow the paper with the original scaling mechanism.

The shape of the input for a linear module is (B, S, D) where B is the batch size, S is the sequence length, and D is the number of channels. It first gets the average of the absolute value of a channel over all B\*S tokens.

```
x_mean = inp.abs().view(-1, inp.shape[-1]).mean(0)
```

Then *x\_mean* is raised to the power of *ratio* to get *scales*. AutoAWQ will try 20 different *ratio*s and decide which one suits best. More on that later.

```
scales = x_mean.pow(ratio).clamp(min=1e-4).view(-1)
```

Next, it normalizes the scales by dividing them with a computation-friendly approximation of a geometric mean. It is the product of the largest scale and the smallest scale and then square rooted.

```
scales = scales / (scales.max() * scales.min()).sqrt()
```

Then it applies the scales to the weights, and pseudo-quantizes the weights. “pseudo quantize” means to convert the floating point weights to integers and then convert them back. The resulting weights are still in floating point but lose some precision. For example, 4.3 can be converted to 4, and then back to 4.0, with some information lost.

```
# Q(W * s)  
fc.weight.mul_(scales_view)  
fc.weight.data = (  
    self.pseudo_quantize_tensor(fc.weight.data)[0] / scales_view  
)
```

It then multiplies the pseudo-quantized weights with the input and compares the output with the original output. It calculates the L2 distance between the two to measure how much the quantization has affected the accuracy. Recall that we mentioned earlier that AutoAWQ tries 20 *ratio*s when it raises *x\_mean* to the power of *ratio*. It uses one *ratio* each time and calculates the loss. After repeating 20 times it chooses the *ratio* with the smallest loss. That’s very brutal force actually.

```
# W * X  
int_w_output = module2inspect(x, **kwargs)  
if isinstance(int_w_output, tuple):  
    int_w_output = int_w_output[0]  
  
# compute mean squared error (L2 norm)  
loss = (  
    (fp16_output - int_w_output).float().pow(2).mean().item()  
)  # NOTE: float prevents overflow
```

Once the *ratio* is set, we have our scales.

#### Find the best clipping value

If you are familiar with model quantization, you know there’s always a tradeoff between the clipping error and the rounding error. Imagine you have a weight matrix whose elements range from -1000 to 1000. You normally want to cover the whole value range of the weight matrix so that every weight can be mapped linearly. However, in many cases most of the weights are rather small, concentrated in a narrow range like (-100, 100). Only one in a thousand weights is large, like 900. They are called outliers. These outliers, which occur infrequently, can be ignored. Instead, we focus on covering the range of the remaining weights. Smaller the range, higher the precision.

The threshold to decide whether a weight is an outlier is called the clipping value.

AutoAWQ finds the best clipping values in exactly the same way it finds the best scales. It simply tries 20 different shrinkage value and selects the one yielding the smallest L2 loss. The shrinkage value is a number between 0 and 1. The clipping value equals the largest weight multiplied by the shrinkage value.

#### Apply the quantization

Once it gets the scales and the clipping values it needs, it will multiply the weights by the best scales, update corresponding previous modules, and clip the outlier weights to the best clipping values. The final step is to really quantize them.

AutoAWQ creates a quantized linear module to replace every floating point linear module. There are four types of quantized linear module with different CUDA implementation, listed below.

```
if self.version == "gemm":  
    scales = scales.t().contiguous()  
    zeros = zeros.t().contiguous()  
    q_linear_module = WQLinear_GEMM  
  
elif self.version == "gemv":  
    q_linear_module = WQLinear_GEMV  
  
elif self.version == "marlin":  
    q_linear_module = WQLinear_Marlin  
  
elif self.version == "gemv_fast":  
    q_linear_module = WQLinear_GEMVFast
```

By default, AutoAWQ uses *WQLinear\_GEMM*. The quantized linear module creates a *qweight* tensor which is of only one fourth the memory size of the original weight tensor, if we choose 4-bit quantization.

```
self.register_buffer(  
    "qweight",  
    torch.zeros(  
        (in_features, out_features // (32 // self.w_bit)),  
        dtype=torch.int32,  
        device=dev,  
    ),  
)
```

The *qweight* tensor is in the *int32* data type. Because there’s no native support for 4-bit integers, AutoAWQ packs eight 4-bit integers into one 32-bit integer. For example, you have 0x1, 0x2, 0x3, 0x4, 0x5, 0x6, 0x7, 0x8, you can pack them as 0x12345678, or 0x86427531. AutoAWQ uses the second pattern, the interleaved one, for better performance in unpacking.

After AutoAWQ replaces every linear module in the model with a quantized linear module, the model is now a legitimate quantized model.

### Saving the Quantized Model

We surely will need to save the quantized model. AutoAWQ saves the quantized model in basically the same way as the Hugging Face *transformers* library does. In fact, it calls the *save\_pretrained* method from the *transformers* library. The only trick is that it first saves the model structure with no parameters and saves the quantized parameters in a second step.

```
class EmptyModule(nn.Module):  
    def __init__(self):  
        super(EmptyModule, self).__init__()  
  
    def forward(self, x):  
        return x  
  
# Save model and config files with empty state dict  
self.model.config.quantization_config = self.quant_config.to_transformers_dict()  
self.model.generation_config.do_sample = True  
self.model.save_pretrained(save_dir, state_dict=EmptyModule().state_dict())
```

```
# shard checkpoint into chunks (10GB default)  
shards, index = shard_checkpoint(  
    self.model.state_dict(), max_shard_size=shard_size, weights_name=model_name  
)  
  
for shard_file, shard in shards.items():  
    if safetensors:  
        # safetensors must be in the same memory, so we duplicate and use contiguous memory  
        shard = {k: v.clone().contiguous() for k, v in shard.items()}  
        save_file(  
            shard, os.path.join(save_dir, shard_file), metadata={"format": "pt"}  
        )  
    else:  
        torch.save(shard, os.path.join(save_dir, shard_file))
```

Alright, that marks the end of the quantization process. You can run the quantized model using AutoAWQ, Hugging Face *transformers*, vLLM, or any other libraries that support loading and running AWQ models. By quantizing a model, you make it faster, lighter, and cheaper.

### Conclusion

There are some details deliberately omitted in this article like the packing of *qcsales* and *qzeros*, why interleaved quantized weights is better for unpacking, the duo scaling, and the difference between quantized linear module backends. Anyways, I hope you have a big picture of how AutoAWQ works after reading this article. You can dig into it by yourself and get the answer to questions above. I would like to write new articles to talk about these topics, too.
