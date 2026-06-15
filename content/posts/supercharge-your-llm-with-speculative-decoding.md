---
title: "Supercharge Your LLM with Speculative Decoding"
date: 2026-01-17
draft: false
math: false
---

> *"Fake it before you make it."*

### Fast to read, slow to write

Large language models are autoregressive: they can process thousands of tokens in a single forward pass, but they still generate output one token at a time. The technical term for that generation step is **decoding**—and decoding is slow.

There is a simple but surprisingly effective trick called **speculative decoding**. The idea is straightforward: guess what the model is about to say next, append those words to the context, and run the model once to read and verify them. A guess counts as valid output only if the model confirms it would have produced that token anyway—then you effectively get multiple tokens from a single decode step.

One way to guess is to look for patterns that already appeared in the text. For example, if your text is about USA and the model just said *“United”*, a reasonable next guess is *“States of America”*.

We’ll cover:

1. What speculative sampling does
2. How N-gram prediction works in practice
3. What the results look like
4. Why verification uses rejection sampling

### What speculative sampling is

At each decode step, a lightweight predictor proposes the next **K** tokens and temporarily appends them to the sequence. The full model runs one forward pass on that extended input and **verifies** those guesses in order—accepting each token only if it matches what the model would have emitted under standard decoding.

Verification is essentially a normal forward pass plus an accept/reject check at each speculative position. Wrong guesses are discarded; verification stops at the first rejection.

The outcome looks like this:

- **All K guesses are correct**  
  → **K + 1 tokens** from a single decode step (best case).
- **The first guess is wrong**  
  → **1 token**—the same as vanilla decoding (worst case).
- **The first *i* guesses are correct** (*i* < K)  
  → **i + 1 tokens** in that step.

Visually, you can think of the sequence as:

![](/images/supercharge-your-llm-with-speculative-decoding/img-01.png)

- **Verified tokens (blue)**: already accepted in a previous decode step
- **Guaranteed tokens (red)**: the token the model emits during verification—always kept
- **Speculative tokens (green)**: proposed guesses; kept only if verification accepts them

The key insight: **when output is predictable, guess ahead and verify in one pass instead of decoding token by token**.

### Using N-grams to guess the future

That leaves one practical question: **how do we guess?**

A surprisingly strong baseline is **N-grams**—sequences of **N consecutive tokens**. In vLLM, both **N** and **K** are configurable: **N** is the pattern length, and **K** is how many tokens to guess ahead. Below we'll use **N = 3** (trigrams) and **K = 2**.

Suppose the current prompt and output look like this:

![](/images/supercharge-your-llm-with-speculative-decoding/img-02.png)

Take the **last 3 tokens**—**EFG**—and search for that sequence earlier in the text. If it appears, reuse whatever tokens followed that occurrence.

With **K = 2**, we predict **AB**. If **K = 1**, we predict **A**.

If **EFG** never appears, we back off:

- Try the **2-gram** **FG**
- Then the **1-gram** **G**
- If all fail, skip speculative sampling for this step

This backoff keeps N-gram speculation conservative: we only guess when the text itself suggests repetition.

That makes it especially effective for structured generation—outlines, summaries, boilerplate—where the same phrases recur.

It also works in other languages, Chinese for example:

![](/images/supercharge-your-llm-with-speculative-decoding/img-03.png)

You do not need to read Chinese to see the pattern: **特斯拉** already appears in the input. The latest output token is **特**; trigram matching predicts **斯拉** next—the greyed suggestion in the figure. And it turns out to be a correct prediction.

### What the results look like

With **N = 4** and **K = 3**, each decode step can yield one to four tokens. The image below colors tokens by how many tokens were decoded **in one step**:

![](/images/supercharge-your-llm-with-speculative-decoding/img-04.png)

Each row shows **raw tokens** (not detokenized text):

- **Black**: no guesses accepted → 1 token decoded
- **Green**: 1 guess accepted → 2 tokens decoded
- **Blue**: 2 guesses accepted → 3 tokens decoded
- **Purple**: all 3 guesses accepted → 4 tokens decoded

On tasks that generate repetitive outputs like text summaries, speculation hits often enough that decoding gets noticeably faster.

### Why verification uses rejection sampling

Speculative sampling requires a rule to decide whether to **accept** a predicted token. That raises a natural question: **what criterion should validation use?** Could we simply accept a speculative token whenever it matches the main model’s Top-1 prediction?

The short answer is: **not if we want to preserve normal sampling behavior**.

Under standard sampling, the model does not always emit its single most likely token. It samples from a probability distribution over possible next tokens. If validation accepted every Top-1 match, speculative decoding would drift toward greedy decoding: the highest-probability token would always be chosen, and lower-probability but still valid tokens would be suppressed.

That is why the canonical algorithm uses **rejection sampling** (see: <https://arxiv.org/abs/2302.01318>). The draft mechanism proposes tokens, and the full model decides whether each proposal should be accepted **in a way that preserves the same output distribution as standard sampling**.

Concretely, suppose the draft model proposes token **x**. Let **q(x)** be the probability assigned by the draft model, and **p(x)** the probability assigned by the main model at the same position.

- If **p(x) < q(x)**, the draft model has **overly favored** this token relative to the main model, so it is accepted only with probability **p(x) / q(x)**. In effect, rejection sampling pulls that excess probability back toward the target distribution.
- If the token is rejected in this way, speculative verification stops at that position, and the next token is drawn from a **residual distribution** proportional to **max(0, p(x) - q(x))** over the vocabulary.
- If **p(x) >= q(x)**, the token is accepted immediately.

The key question is not simply whether the draft token matches the Top-1 choice. The real question is whether the draft model assigns that token **too much** probability relative to the target model. If it does, rejection sampling trims that excess; if it does not, the token can pass through unchanged and recovers its missing probability mass when other tokens are rejected.

This matters because speculative decoding is not just about speed. It also needs to stay faithful to the behavior of the original sampler and the diversity of the output distribution. Rejection sampling is the piece that lets us **guess ahead without changing what distribution we are sampling from**.

### Closing thoughts

N-gram–based speculative sampling is not glamorous. It doesn’t require training another model, or fancy distillation. It simply asks:

> *“Has the model already seen something like this before?”*

In many real-world workloads, the answer is often yes — and when it is, it accelerates decoding almost for free.

