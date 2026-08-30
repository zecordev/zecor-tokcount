# zecor-tokcount

Exact BPE token counting and byte-precise context trimming for LLM cost control.

Part of [Zecor](https://zecor.dev) -- an autonomous software construction engine.
Apache-2.0. Prebuilt binaries for Linux / macOS / Windows are attached to each
[release](https://github.com/zecordev/zecor-tokcount/releases); or `cargo install zecor-tokcount`.

## 3. `zecor-tokcount` — token accounting and context control

**Incumbents.** `tiktoken` (OpenAI encodings, Python), HF `tokenizers` (exact per-model,
Rust core but usually driven from Python), `llama.cpp`'s tokenizer (C, model-coupled),
a hundred `chars/4` guesses in every agent framework.

**Gaps.** Nobody ships a single fast binary that: counts against *both* the vendor
billing encoding and a local model's exact tokenizer; trims to a hard budget on a token
boundary and decodes back; **packs** a set of documents into a budget by priority; and
prices the result. Agent frameworks over-truncate (lose the answer) or over-send (pay
for nothing) because the estimate is a guess.

**Shipped.** `count` (tiktoken `cl100k`/`o200k`/`p50k`/`r50k` embedded, or an HF
`tokenizer.json` path) and `trim` (token-boundary truncate + decode). **`cost`** — USD
from a built-in per-model price table (longest-prefix match, so `gpt-4o-2024-08-06`
resolves), `--in-price` / `--out-price` override, `--in-tokens` or stdin. **`pack`** —
greedily assemble files in priority order into a token budget, whole files only, later
smaller files still get a slot; prints the blob, a JSON-ish summary to stderr.
**`diff-trim`** — trim a unified diff to a budget by dropping *whole hunks* from the end
(file headers always kept, a trailing marker records the count). Mirrored in
`zecor.tokens`; the review path (`orchestrator._land_review`) now `diff_trim`s to 60k
tokens before the model sees the diff, and `zecor cost` prices a file/stdin.

**Still to world-class.**
- **Per-doc trim in `pack`** — currently whole-file only; a required-first pass plus a
  fractional trim of the last doc would pack tighter.
- **Streaming count** — read stdin incrementally, stop early past `--budget`.
- **Special-token / chat-template overhead** counting per named template.
- **Offline model registry** — bundle common `tokenizer.json` hashes so
  `--model qwen3-coder` resolves without a path.

## Build

```
cargo build --release      # -> target/release/zecor-tokcount
cargo test --all-targets
```
