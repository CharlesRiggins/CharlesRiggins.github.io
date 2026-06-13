---
title: "Supercharge Your LLM with Speculative Decoding"
date: 2026-01-17
draft: false
math: false
---

#### Guess it before you know it

#### Read fast but write slow

Large language models are autoregressive, meaning they read thousands of tokens at once but output one token at a time. A fancy term for AI’s output operation is decode. Decoding is slow.

Yet there’s a simple but surprisingly effective trick. It’s called **speculative decoding**. The concept is easy. You guess what the LLM is about to say. You feed your guess to the LLM and let it read them at once. Remember that AI can read many words at a time. If you guess it right, you essentially help the LLM output many tokens at once.

One way to guess the tokens is to find the word patterns that appeared previously. For example, if you mentioned Trump in your questions and the AI said “the president”, it makes sense to guess that it is going to say “of the United States” next.

We’ll cover:

1. What speculative sampling actually does
2. How N-gram prediction works in practice
3. What the results look like
4. How vLLM implements this optimization (and its limitations)

#### What is Speculative Sampling

In standard autoregressive decoding, if the current sequence has **N tokens**, the model produces exactly **one new token** — token **N+1** — and then the next. No matter how much parallelized compute resources you have in your hardware, you can only compute for one token at a time. No matter how strong the model is, you still pay for a full forward pass per token.

Speculative sampling breaks this one-token-at-a-time constraint.

Instead of waiting for the model to decide every step, we **guess ahead**. A lightweight mechanism predicts the next **K tokens**, appending them temporarily to the sequence. The full model then runs once on this extended input and verifies how many of those guesses were correct. The verification process contains almost the same procedure as the generation process, except for a minimal difference at the very end of the pipeline.

The outcome looks like this:

- **All K guesses are correct**  
  → You get **K + 1 tokens** from a single decode step (best case).
- **None of the guesses are correct**  
  → You fall back to normal decoding and get **1 token** (worst case).
- **The first i guesses are correct**  
  → You get **i + 1 tokens** in that step.

Visually, you can think of the sequence as:

![](/images/supercharge-your-llm-with-speculative-decoding/img-01.png)

- **Verified tokens (blue)**: already prefetched and trusted
- **Guaranteed tokens (red)**: produced by the model and always valid
- **Speculative tokens (green)**: only valid if they match what the model would have generated anyway

The key insight is simple: **if the output text is predictable, we should exploit that predictability**.

#### Using N-grams to Guess the Future

So how do we guess these tokens to be generated?

One surprisingly strong yet simple baseline is **N-grams**.

An N-gram is just a sequence of **N consecutive tokens**. In vLLM, **N is configurable**. Let’s use **N = 3** (i.e., 3-grams) for illustration.

Assume the current input + output looks like this:

![](/images/supercharge-your-llm-with-speculative-decoding/img-02.png)

We take the **last 3 tokens** — EFG—and search for this sequence earlier in the text. If we find it, we simply reuse whatever tokens followed it last time.

If **K = 2** (i.e., guess 2 tokens ahead), then in this case we would predict **AB**.

If the 3-gram EFG does not exist, we back off:

- Try the **2-gram** FG
- Then the **1-gram** G
- If even that fails, we skip speculative sampling entirely

This backoff strategy makes N-gram speculation conservative: we only speculate when the text itself provides evidence that repetition is likely.

This works especially well in structured generation tasks — outlines, summaries, boilerplate text — where repetition is common.

Here’s another example, in Chinese.

![](/images/supercharge-your-llm-with-speculative-decoding/img-03.png)

You don’t need to understand Chinese to notice that the three-character phrase “特斯拉” has appeared in the input text, and the latest output token is “特”, with n-grams we will predict “斯拉”, shown as the greyed suggestion.

#### What the Results Look Like

With N=4 and K = 3, we can observe different decoding behaviors in practice:

![](/images/supercharge-your-llm-with-speculative-decoding/img-04.png)

Text is shown in **raw** **tokens**

- **Black tokens**: speculation failed → 1 token decoded
- **Green tokens**: 1 guess correct → 2 tokens decoded
- **Blue tokens**: 2 guesses correct → 3 tokens decoded
- **Purple tokens**: all guesses correct → 4 tokens decoded

In real outputs from the outline model, successful speculation occurs frequently enough to deliver a **meaningful latency reduction**.

#### How vLLM Implements This

Speculative sampling requires a rule for deciding whether to **accept** a predicted token.

The canonical approach uses a **rejection sampler** (see: <https://arxiv.org/abs/2302.01318>), and vLLM implements that. This sampler is intentionally **probabilistic**: even if the predicted token matches the Top-1 choice of the main model, it may still be rejected.

Why? **Diversity**.

If Top-1 predictions were always accepted, decoding would collapse into greedy search, starving lower-probability tokens and reducing output diversity.

The current N-gram implementation in vLLM is very simple.

At **every decode step**, it scans the **entire input sequence** to search for matching N-grams. There is **no caching**.

This results in a time complexity of:

```
O(N × L)
```

where:

- **N** = N-gram size
- **L** = sequence length

When an N-gram appears multiple times, vLLM always selects the **first occurrence**.

This heuristic is cheap and easy to reason about, but clearly suboptimal in some cases. Smarter selection strategies — or caching — could further improve both accuracy and performance.

#### Closing Thoughts

N-gram–based speculative sampling is not glamorous. It doesn’t require another model, fancy distillation, or heavy tuning. It simply asks:

> *“Has the model already seen something like this before?”*

In structured generation workloads, the answer is often yes — and when it is, decoding becomes noticeably faster.

The current implementation leaves performance on the table, but even in its simplest form, N-gram speculation proves that **small, well-placed heuristics can still matter in modern LLM systems**.
