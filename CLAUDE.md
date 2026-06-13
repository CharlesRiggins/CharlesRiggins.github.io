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
- **Post images** live in `static/images/<post-slug>/` and are referenced as
  `/images/<post-slug>/img-NN.ext`. Keep images local (no hotlinking to external CDNs)
  so the site is self-contained. The slug must match the post's filename.

## Migrating posts from Medium

The blog content was originally migrated from Medium (`@crclq2018`). To import or
re-import posts:

- The Medium RSS feed `https://medium.com/feed/@crclq2018` embeds the **full** article
  HTML in `<content:encoded>` (not summaries) plus all image URLs — use it as the source
  rather than scraping rendered pages. Note: the feed only carries the ~10 most recent posts.
- Convert HTML → Markdown with `markdownify` + `beautifulsoup4` (installed via
  `pip install --user`). Strip Medium tracking pixels (`<img>` with `/stat?` in the src).
- Download images with **`curl`**, not Python's `urllib` — see the SSL gotcha below.
- Per-post front matter is generated with `draft: false` and `math: false` by default
  (flip `math` to `true` manually if a post needs it). Auto-tags and `canonicalURL` are
  NOT used — the github.io copy is treated as canonical.

## Key files

- `hugo.toml` — site config: title, menus, `params` (theme toggle, reading time, social icons).
  Homepage avatar: `params.homeInfoParams.avatar = "/images/avatar.jpeg"`, rendered by the
  custom `layouts/_partials/home_info.html` override (title + intro left, circular avatar right).
  Styling in `assets/css/extended/home.css` (PaperMod auto-loads `assets/css/extended/*.css`).
  The avatar is the owner's own photo — no attribution needed.
- `layouts/_partials/extend_head.html` — MathJax loader (PaperMod has no built-in math).
  Note: PaperMod uses the `_partials/` dir (newer layout), not `partials/`.
- `.github/workflows/deploy.yml` — build + deploy pipeline.
- `themes/PaperMod/` — submodule; do NOT edit directly. Override by copying files into the
  project's own `layouts/` instead.

## Gotchas

- `languageCode` in `hugo.toml` emits a deprecation warning on this Hugo version — harmless.
- `public/` and `resources/_gen/` are gitignored build artifacts — never commit them.
- **macOS python.org Python has no CA certificates by default** — `urllib`/`requests`
  downloads fail with `CERTIFICATE_VERIFY_FAILED`. Use `curl` for HTTP(S) downloads in
  scripts, or run `Install Certificates.command` from the Python install.
