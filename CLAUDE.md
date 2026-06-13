# CLAUDE.md

Guidance for Claude Code working in this repository.

## What this is

A personal blog built with **Hugo** (static site generator) + the **PaperMod** theme,
deployed to **GitHub Pages**. Styled to match https://kellerjordan.github.io/posts/muon/
(which uses the same Hugo + PaperMod stack).

- **Live site:** https://charlesriggins.github.io/
- **Repo:** https://github.com/CharlesRiggins/CharlesRiggins.github.io
- **GitHub user:** CharlesRiggins (repo MUST stay named `CharlesRiggins.github.io` — it's a user site)
- **Local path:** `~/Documents/github-blog`

## Environment / toolchain

- **Hugo:** extended **v0.163.1**, installed at `~/bin/hugo` (NOT via Homebrew — Homebrew is not installed).
  `~/bin` was added to PATH in `~/.zprofile`, so `hugo` works in new terminals.
  In this session's shell it may not be on PATH yet — use the full path `~/bin/hugo` to be safe.
- **Theme:** PaperMod, vendored as a git submodule at `themes/PaperMod`.
  After a fresh clone, run `git submodule update --init --recursive`.
- Hugo extended is required (PaperMod uses SCSS).

## Common commands

```bash
# Local preview (drafts included) — http://localhost:1313/
~/bin/hugo server --buildDrafts

# Production build into ./public (gitignored)
~/bin/hugo --gc --minify

# New post (then edit front matter: set draft: false)
~/bin/hugo new content posts/my-post.md
```

## Deploy

Push to `main` → GitHub Actions (`.github/workflows/deploy.yml`) builds with Hugo and
publishes to Pages. Live in ~1 minute. No manual build/commit of `public/` needed.

```bash
git add -A && git commit -m "..." && git push
```

- Pages **build type is `workflow`** (serves the Actions artifact, NOT a branch).
  If the site 404s after changing Pages settings, re-run the deploy workflow:
  `gh workflow run deploy.yml`
- `HUGO_VERSION` is pinned in the workflow — keep it in sync with the local Hugo version.

## Writing posts

- Posts live in `content/posts/`. The About page is `content/about.md`.
- Front matter that matters:
  - `draft: false` — required to publish.
  - `math: true` — opt-in **per post** to load MathJax (`$...$` inline, `$$...$$` display).
    Math is intentionally NOT enabled site-wide, so non-math pages stay lightweight.
- Code blocks use fenced ```` ```lang ```` syntax (highlight.js / Hugo Chroma).

## Key files

- `hugo.toml` — site config: title, menus, `params` (theme toggle, reading time, social icons).
- `layouts/_partials/extend_head.html` — MathJax loader (PaperMod has no built-in math).
  Note: PaperMod uses the `_partials/` dir (newer layout), not `partials/`.
- `.github/workflows/deploy.yml` — build + deploy pipeline.
- `themes/PaperMod/` — submodule; do NOT edit directly. Override by copying files into the
  project's own `layouts/` instead.

## Gotchas

- `languageCode` in `hugo.toml` emits a deprecation warning on this Hugo version — harmless.
- `public/` and `resources/_gen/` are gitignored build artifacts — never commit them.
