---
title: "Hello World: My First Post"
date: 2026-06-13
draft: false
math: true
tags: ["meta"]
---

This is my first post on my new Hugo + PaperMod blog, styled after
[Keller Jordan's blog](https://kellerjordan.github.io/posts/muon/).

## Math works

Inline math like $e^{i\pi} + 1 = 0$ renders with MathJax. Display math too:

$$
\nabla_\theta \mathcal{L}(\theta) = \frac{1}{N} \sum_{i=1}^{N} \nabla_\theta \ell(f_\theta(x_i), y_i)
$$

## Code works

```python
def hello(name: str) -> str:
    return f"Hello, {name}!"

print(hello("world"))
```

## Everything else

- Light/dark mode toggle in the top-right
- Reading time estimates
- Clean single-column reading layout

Write new posts by adding markdown files under `content/posts/`.
