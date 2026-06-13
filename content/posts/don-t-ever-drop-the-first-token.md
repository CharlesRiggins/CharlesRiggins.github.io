---
title: "Don’t Ever Drop the First Token."
date: 2024-10-14
draft: false
math: false
---

#### A gentle introduction to LLM streaming inference

### The Limited Memory

Large Language models (LLMs) nowadays are all transformers. Transformers by design have limited memory, meaning that they can only remember a specific amount of tokens at any given time. The size of the memory is also called **context length**.

But what if we have an input that arrives and grows incrementally? Like, what if new input data keeps coming and never stops? Say if we have the data of stock prices. Each day we get some new data from the market. As the data accumulates, it will certainly at some time surpass the maximum memory size of our LLM.

A very straight forward solution would be to just drop the oldest data. In a lot of cases, it roughly works. For example, if our LLM can only remember the stock price of 100 days, then perhaps just let it remember 100 days of data from now and forget the price data before 100 days ago.

#### Streaming Inference

This solution of processing a gradually incrasing data with a limited window size has a name. It’s called straming inference.

For example, consider a input:

```
X = “abcdefg”
```

If the window size is 3 characters, the streaming inference will only use the latest 3 characters each time it process the input. The results will be:

```
Y1 = f("abc")    
Y2 = f("bcd")    
Y3 = f("cde")    
...
```

Another example. Here is a sequence of 10 numbers,

```
[1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
```

Calculating the sum of the 4 most recent numbers from left to right:

```
Y1 = 1 + 3 + 5 + 7  
Y2 = 3 + 5 + 7 + 9   
Y3 = 5 + 7 + 9 + 11   
...
```

This is also streaming inference, with a windows size of 4.

#### LLM’s inability to perform streaming inference

But LLMs can’t do streaming inference.

Well, it can technically, but cannot practically.

For example, given an article of 1 million words, we ask an LLM to count the occurrences of the word “*the*” in the most recent 1,000 words. That’s a typical task of streaming inference.

Suppose the memory size of the LLM is exactly 1,000 words. The process will look like this:

```
# Processing 1,000 words at a time    
Y0 = LLM(X[0:1000])  
Y1 = LLM(X[1:1001])  
Y2 = LLM(X[2:1002])  
...
```

If each inference is conducted independently, it is 1 million inferences, each processing 1,000 tokens. Therefore, the time complexity of this is O(NM), where N is the total input length (1 million) and M is the context length (1,000).

But as you may noticed, there could be some potential repeated calculations. We have processed the word 0 to 1000 at the first iteration, and we will process word 1 to 1001 at the second iteration, and 999 words of them are the same words. Can we make some use of that? If we can somehow use the previously processed result of the 999 tokens, then we justneed to process the new token.

Sadly, the answer is no.

Previously processed tokens’ activations can be stored and reused during subsequent inferences, eliminating the need for repeated computations. This technique is known as the **Key-Value Cache**, allowing for the efficient generation by only process the new token. However, this technique does not apply to our case here.

When processing word 1 to 1001 at the second iteration, word 0 is no longer included in the memory of the LLM.

Modern LLMs are almost all **causal models**.

> *Causal Model: Each token’s neural activation depends on the tokens preceding it.*

Since word 0 is dropped from the memory, the neural activations for word 1 to 1000 change, because they depend on tokens preceding them.

In other words, when calculating Y2, the activations for the repeated 999 tokens acutally differ from those saved during Y1 computation.

But what if we insist in using the previously saved activataion cache for the streaming inference? After all, we just drop 1 word, not a huge change.

Unfortunately, it turns out that if you discard the oldest token while still use the previously cached activations for the rest, the output will **collapse**, meaning that the output is completely unreasonable garbage.

Why does dropping just one token lead to such a huge impact? Researchers explored this phenomenon and named it **Attention Sinks**.

### Hello, Attention Sink

#### What is an Attention Sink?

We know that in the Self-Attention mechanism, the Key and Query are multiplied, then a softmax is applied to get the attention weights. These weights are then multiplied with the Value, essentially performing a weighted average over all the previous tokens’ Values.

> *Attention Sinks: A phenomenon where the first token’s attention weight is significantly higher than other tokens.*

The following heatmap from the Llama-2 model illustrates this.

![](/images/don-t-ever-drop-the-first-token/img-01.jpeg)

Visualization of the average attention logits in Llama-2–7B over 256 sentences, each with a length of 16. The model heavily attends to the initial token across all layers and heads.

#### Why Does Removing the First Token Cause a Breakdown?

Look at the softmax formula below. Due to Attention Sinks, exp(x1) is very large.

![](/images/don-t-ever-drop-the-first-token/img-02.png)

If exp(x1) is removed, the denominator shrinks significantly, causing originally minor attention weights to suddenly become large, which results in all other tokens’ Values being multiplied by a large factor and accumulated into the attention output. This greatly distorts the outcome of the Attention operator.

#### Why the First Token?

Why is it the 1st token, rather than any other token, that has this high attention weight?

This can be broken down into two sub-questions:

1. Why is it the 1st token and not the 2nd or any other token?
2. Why does an Attention Sink need to exist at all?

To answer the first, the paper’s authors propose a plausible hypothesis:

If the existence of an Attention Sink significantly impacts the attention results, it must:

> *Either always exist or never exist.*

The only way it can always exist is by placing it at the 1st token. If it is at the nth token, there would be no sink before n, but there would be one afterward, causing significant variations in the attention output.

This leads to the second question: why must there be an Attention Sink?

The authors of StreamingLLM suggest that this might be because the Attention operator wants to learn a no-op operation. Refer to the following image.

![](/images/don-t-ever-drop-the-first-token/img-03.png)

Due to residual links, the result of the Attention is added to the original x. If x already contains sufficient information, the Attention needs to output a near-zero value to leave x unchanged.

Since the Attention output is a weighted average of all Values, it achieves the no-op operation by assigning a high exp value to one token, causing the softmax denominator to become large, thus minimizing the weights assigned to the other tokens.

You might have other questions about this topic, but they are beyond the scope of this article. I recommend interested readers check out the following resources:

- Quantizable Transformers: Removing Outliers by Helping Attention Heads Do Nothing.
- Attention is off by one.
- Massive Activations in Large Language Models

### StreamingLLM

To address the inefficiency of streaming inference, the authors of StreamingLLM proposed the following approach, which can be summarized as:

> *Always include the 1st token in subsequent windows.*

If the 1st token has a high attention weight and its absence leads to a drastic reduction in the softmax denominator, then we should always include it, as illustrated below.

![](/images/don-t-ever-drop-the-first-token/img-04.png)

The 1st token always participates in subsequent inferences, while the 2nd and following tokens can be discarded.

Testing showed that this method produces results close to the O(NM) re-computation approach, as seen in the PPL (perplexity) values in the following figure.

![](/images/don-t-ever-drop-the-first-token/img-05.png)

#### Can StreamingLLM Solve the Long-Text Inference Problem?

No, because it **discards** intermediate tokens.

However, it does solve the inefficiency of streaming inference, preventing the need for repeated re-computation.

As for whether LLM streaming inference has practical applications, I don’t know.

### Conclusion

This paper’s primary contribution is the discovery and explanation of the Attention Sinks phenomenon, which has inspired many subsequent works, including those in quantization. Please support, clap, and comment, and I will share more related developments in the future.

### References

Efficient Streaming Language Models with Attention Sinks  
<https://arxiv.org/abs/2309.17453>
