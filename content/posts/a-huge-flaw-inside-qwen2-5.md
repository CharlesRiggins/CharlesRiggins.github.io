---
title: "A Huge Flaw in Qwen2.5"
date: 2025-02-17
draft: false
math: false
---

### The Flagship Model

Alibaba has recently unveiled its latest flagship Large Language Model, **Qwen2.5 Max**. They claim it outperforms **DeepSeek**, the Chinese model that rivals OpenAI’s GPT and had wiped out billions of market value from U.S. tech stocks.

**Qwen2.5** follows Qwen2's architecture, and closely resembles that of LLaMA. It comes in multiple sizes, ranging from 0.5B to 72B.

**Qwen2.5-7B** is a great choice for consumer GPUs like the RTX 40 series, delivering reasonable performance for its size. However, we’ve discovered **a major flaw** inside the model.

### KV Cache Quantization

We ran into the issue while attempting to quantize the KV cache.

When hosting an LLM service, we can’t just load its weights. We also need **KV cache** in GPU memory. In short terms, the KV cache stores previously processed tokens so we don’t need to recompute them again and again.

The KV cache is typically stored in FP16, but it can be compressed to 8-bit numbers to further reduce memory usage. Thanks to the inherent noise robustness of the inner product operation, this usually does little harm to the model’s capability. This process of converting data from a higher-precision format to a lower-precision one is known as quantization.

Quantizing a real number to float8 introduces roughly **5% noise**. This level of noise is acceptable because positive and negative noise tend to **cancel out** during the inner product computation, leading to a much lower noise ratio in the final output.

This method works for almost every model — except Qwen2.5–7B. When KV cache quantization is applied, Qwen2–7B’s performance degrades significantly, producing **repetitive**, **nonsensical** responses, as shown below.

![](/images/a-huge-flaw-inside-qwen2-5/img-01.png)

The Qwen’s response deteriorate badly with KV cache quantization

### Softmax as an Noise Amplifier

We identified the root of the problem in the **extremely large-valued logits** in its attention heads.

Qwen2.5–7B features Grouped Query Attention (GQA) with 28 attention heads. Among these heads, two of them are malformed. The figure below shows a sample of logit values for all the 28 heads in the first transformer layer in Qwen2.5–7B. You can see an extremely large value in the 2nd column of the 3rd row.

![](/images/a-huge-flaw-inside-qwen2-5/img-02.png)

Sampled value of attention logits

You might wonder why large-valued logits are a problem.

The reason is that the **softmax** operator in the attention mechanism acts as a **noise amplifier**. It significantly amplifies the noise when logits are large.

Let’s illustrate that with an example.

Recall that the softmax function is essentially normalized exponentials, as shown by the equation below.

![](/images/a-huge-flaw-inside-qwen2-5/img-03.png)

The softmax function

Suppose we have two logits, A and B, that are initially equal. Let’s consider two different cases.

#### Small Logits

Let A=1 and B=1. After applying softmax, we get 0.5 and 0.5. Now, let’s introduce a 1% noise to A, making it 1.01. We then get this.

![](/images/a-huge-flaw-inside-qwen2-5/img-04.png)

After softmax, the outputs become approximately 0.503 and 0.497. It remains close to 50–50, indicating that the noise has a negligible impact.

#### Large Logits

Now consider if A=1000 and B=1000. Without noise, the softmax output is still 0.5 and 0.5. But if we add the same 1% noise to A, it becomes 1010, and we have

![](/images/a-huge-flaw-inside-qwen2-5/img-05.png)

Since exp(10) ≈ 22,000, the ratio between the two logits after softmax now becomes approximately 22,000 : 1. As a result, the softmax output moves drastically to roughly 1 and 0, completely drifting away from the original value.

The same thing happens in Qwen2.5–7B.

In a word, the two attention heads with extremely large logits make themselves highly sensitive to noise. Noise is inevitable. The model is flawed in its lack of robustness.

At this point, we have demonstrated the issue, but now you might have a question.

**Among all the 28 attention heads, what makes these two heads different?**

We have the same question and move on to dig into it.

### The Waves

We find out two weird things.

The first one is the presence of outlier channels in the key vectors. You can see in the image below that a few channels have consistently large values across different tokens. You may recall that logits are inner products of Key and Query vectors. If the Key is large in an attention head, the logit will be large.

![](/images/a-huge-flaw-inside-qwen2-5/img-06.png)

Wait! Don’t forget that Qwen2.5 is a GQA model. Each Key head is shared by several Query heads. If it’s the key vector that causes the problem, why does only one head in its group go off?

We take a closer look into the channels of the Key head.

![](/images/a-huge-flaw-inside-qwen2-5/img-07.png)

Then check the channel waves of three Query heads in this group. Pay special attention to the third graph.

![](/images/a-huge-flaw-inside-qwen2-5/img-08.png)

The wave of the third Query head highly resembles the wave of the Key head. The two waves synchronize and result in a huge inner product (i.e., logit) in the corresponding attention head.

### Smooth Matrix K

Now that we know the cause of the problem, how do we solve it?

Since we know that the large logit is caused by a large Key head and a synchronous Query head, one way to solve the problem is to try to reduce the magnitude of the key and hence the logits.

A cool feature of softmax is shift-invariance. If we add a constant to every input of the function, the result stays the same.

![](/images/a-huge-flaw-inside-qwen2-5/img-09.png)

Shift-invariance of Softmax

With this characteristic, we can first decrease the value of the Key and then quantize it. Remember that converting from float16 to float8 introduces a noise of around 5% no matter how large the original number is.

![](/images/a-huge-flaw-inside-qwen2-5/img-10.png)

By subtracting the mean value of the channel from the key vector, we make the magnitude of the outlier channels in the key vector much smaller. The mean value, which is pre-determined by methods like calibration, is constant during runtime inference.

![](/images/a-huge-flaw-inside-qwen2-5/img-11.png)

This technique is called **Smooth Matrix K** and was first introduced in the paper of **SageAttention**. By adding this processing in the code, the performance of the Qwen2.5–7B model is back to the level it should be.
