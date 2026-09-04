# Antigravity (`agy`)

Gemini CLI's own ACP support is deprecated, and Google's `agy` (Antigravity CLI) has no
official ACP mode - see [antigravity-cli#31](https://github.com/google-antigravity/antigravity-cli/issues/31).
[antigravity-acp](https://github.com/shubzkothekar/antigravity-acp) is a third-party bridge that
speaks real ACP and drives `agy` as a subprocess.
This repo confirmed it end to end: `initialize`, `session/new`, and a `session/prompt` round trip
that reached Google's backend for real.

`acp/agy` is not a built-in default adapter (see the README's "Adding your own adapter").
Unlike the other adapters it needs [Bun](https://bun.com) installed, not just `npm`/`node`,
since `antigravity-acp`'s postinstall script shells out to `bun`:

```bash
curl -fsSL https://bun.sh/install | bash   # or: npm install -g bun
npm install -g antigravity-acp             # postinstall needs bun on PATH; also fetches agy if missing
```

Then register it, e.g. via the TOML config:

```toml
[adapters.agy]
binary = "agy-acp"
```

## Caveats

- `antigravity-acp`'s own precompiled release binaries (the ones that don't need Bun installed)
  are Bun-compiled and require an AVX2-capable CPU - they crash with `Illegal instruction` on
  older hardware (e.g. Celeron J-series). The `npm install -g antigravity-acp` + Bun path above
  doesn't have this problem.
- `agy` authenticates via Google OAuth (`agy` login flow), separate from any Gemini API key.
- The `antigravity-acp` README flags a real Antigravity Terms of Service risk: driving `agy`
  through a third-party ACP client is the pattern Google's ToS names as a violation, and may be
  grounds for account suspension. Read that README's "Terms of Service risk" section before use.
