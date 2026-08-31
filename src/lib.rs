// SPDX-License-Identifier: Apache-2.0
//! Exact token counting and byte-precise context trimming.
//!
//! Two backends:
//!   * `tiktoken-rs` embedded encodings (`cl100k_base`, `o200k_base`) -- the encodings
//!     the frontier vendors bill against, self-contained, no network.
//!   * a Hugging Face `tokenizer.json` on disk -- the *exact* tokenizer a local model
//!     ships, for T1 cost/context math.
//!
//! Used by the Python engine to replace the `bytes/4` heuristic in `_estimate_tokens`
//! and to trim a prompt down to a hard token budget without a wasted round trip.

use anyhow::{anyhow, Result};
use serde::Serialize;
use tiktoken_rs::CoreBPE;

pub enum Encoder {
    Tik(CoreBPE),
    Hf(Box<tokenizers::Tokenizer>),
}

impl Encoder {
    /// `spec` is a built-in name (`cl100k_base`, `o200k_base`, `p50k_base`, `r50k_base`)
    /// or a path to a `tokenizer.json`.
    pub fn load(spec: &str) -> Result<Self> {
        match spec {
            "cl100k_base" => Ok(Encoder::Tik(tiktoken_rs::cl100k_base()?)),
            "o200k_base" => Ok(Encoder::Tik(tiktoken_rs::o200k_base()?)),
            "p50k_base" => Ok(Encoder::Tik(tiktoken_rs::p50k_base()?)),
            "r50k_base" | "gpt2" => Ok(Encoder::Tik(tiktoken_rs::r50k_base()?)),
            path if std::path::Path::new(path).is_file() => {
                let t = tokenizers::Tokenizer::from_file(path)
                    .map_err(|e| anyhow!("load {path}: {e}"))?;
                Ok(Encoder::Hf(Box::new(t)))
            }
            other => Err(anyhow!(
                "unknown encoding {other:?} (want cl100k_base | o200k_base | p50k_base | \
                 r50k_base | a path to tokenizer.json)"
            )),
        }
    }

    pub fn count(&self, text: &str) -> Result<usize> {
        Ok(match self {
            Encoder::Tik(bpe) => bpe.encode_with_special_tokens(text).len(),
            Encoder::Hf(t) => t
                .encode(text, false)
                .map_err(|e| anyhow!("encode: {e}"))?
                .len(),
        })
    }

    /// Return `text` truncated so it encodes to at most `max_tokens` tokens, on a token
    /// boundary, decoded back to a valid string. Byte-level BPE can split a multibyte
    /// character across tokens, so the cut may back off a few tokens to land on a valid
    /// UTF-8 boundary (still `<= max_tokens`).
    pub fn trim(&self, text: &str, max_tokens: usize) -> Result<String> {
        match self {
            Encoder::Tik(bpe) => {
                let ids = bpe.encode_with_special_tokens(text);
                if ids.len() <= max_tokens {
                    return Ok(text.to_string());
                }
                for take in (0..=max_tokens).rev().take(6) {
                    if let Ok(s) = bpe.decode(&ids[..take]) {
                        return Ok(s);
                    }
                }
                Ok(String::new()) // <=6 tokens all straddle a char boundary: give up cleanly
            }
            Encoder::Hf(t) => {
                let enc = t.encode(text, false).map_err(|e| anyhow!("encode: {e}"))?;
                let ids = enc.get_ids();
                if ids.len() <= max_tokens {
                    return Ok(text.to_string());
                }
                for take in (0..=max_tokens).rev().take(6) {
                    if let Ok(s) = t.decode(&ids[..take], true) {
                        return Ok(s);
                    }
                }
                Ok(String::new())
            }
        }
    }
}

// ---------------------------------------------------------------- cost --------

/// USD per 1M tokens, (input, output). Public list prices; override with `--in-price`
/// / `--out-price` on the CLI when a contract differs.
pub fn price_per_mtok(model: &str) -> Option<(f64, f64)> {
    let m = model.to_ascii_lowercase();
    let table: &[(&str, f64, f64)] = &[
        ("gpt-4o", 2.50, 10.00),
        ("gpt-4o-mini", 0.15, 0.60),
        ("gpt-4.1", 2.00, 8.00),
        ("gpt-4.1-mini", 0.40, 1.60),
        ("gpt-4.1-nano", 0.10, 0.40),
        ("o3", 2.00, 8.00),
        ("o4-mini", 1.10, 4.40),
        ("claude-opus-4", 15.00, 75.00),
        ("claude-sonnet-4", 3.00, 15.00),
        ("claude-3-5-haiku", 0.80, 4.00),
        ("claude-3-haiku", 0.25, 1.25),
        ("gemini-1.5-pro", 1.25, 5.00),
        ("gemini-1.5-flash", 0.075, 0.30),
        ("gemini-2.0-flash", 0.10, 0.40),
        ("deepseek-chat", 0.27, 1.10),
        ("local", 0.0, 0.0),
    ];
    // longest-prefix match, so "gpt-4o-2024-08-06" resolves to "gpt-4o"
    table
        .iter()
        .filter(|(k, _, _)| m.starts_with(k) || m.contains(k))
        .max_by_key(|(k, _, _)| k.len())
        .map(|(_, i, o)| (*i, *o))
}

#[derive(Debug, Serialize)]
pub struct Cost {
    pub model: String,
    pub in_tokens: usize,
    pub out_tokens: usize,
    pub in_usd: f64,
    pub out_usd: f64,
    pub total_usd: f64,
}

pub fn cost(
    model: &str,
    in_tokens: usize,
    out_tokens: usize,
    price: Option<(f64, f64)>,
) -> Result<Cost> {
    let (pin, pout) = price
        .or_else(|| price_per_mtok(model))
        .ok_or_else(|| anyhow!("no price for model {model:?}; pass --in-price / --out-price"))?;
    let in_usd = in_tokens as f64 / 1e6 * pin;
    let out_usd = out_tokens as f64 / 1e6 * pout;
    Ok(Cost {
        model: model.to_string(),
        in_tokens,
        out_tokens,
        in_usd,
        out_usd,
        total_usd: in_usd + out_usd,
    })
}

// ---------------------------------------------------------------- pack --------

#[derive(Debug, Serialize)]
pub struct PackResult {
    pub text: String,
    pub tokens: usize,
    pub included: Vec<String>,
    pub dropped: Vec<String>,
}

/// Greedily assemble `files` (in the given priority order) into a single blob that fits
/// `budget` tokens. Each file gets a `===== path =====` header. A file that would not
/// fit whole is skipped (not truncated) so every included file is complete; later,
/// smaller files still get a chance.
pub fn pack(enc: &Encoder, files: &[(String, String)], budget: usize) -> Result<PackResult> {
    let mut out = String::new();
    let mut used = 0usize;
    let mut included = Vec::new();
    let mut dropped = Vec::new();
    for (path, body) in files {
        let chunk = format!("===== {path} =====\n{body}\n\n");
        let cost = enc.count(&chunk)?;
        if used + cost <= budget {
            out.push_str(&chunk);
            used += cost;
            included.push(path.clone());
        } else {
            dropped.push(path.clone());
        }
    }
    Ok(PackResult {
        text: out,
        tokens: used,
        included,
        dropped,
    })
}

// ------------------------------------------------------------ diff-trim -------

/// Trim a unified diff to `budget` tokens by dropping whole hunks from the end. File
/// headers (`diff --git`, `---`, `+++`, `index`, `rename`, `new file`, …) are always
/// kept; a trailing marker records how many hunks were removed.
pub fn diff_trim(enc: &Encoder, diff: &str, budget: usize) -> Result<String> {
    if enc.count(diff)? <= budget {
        return Ok(diff.to_string());
    }
    // Split into a preamble + a sequence of hunks (each starts at an `@@` line).
    let mut blocks: Vec<String> = Vec::new();
    let mut cur = String::new();
    let mut in_hunk = false;
    for line in diff.split_inclusive('\n') {
        let is_file_hdr = line.starts_with("diff --git ")
            || line.starts_with("--- ")
            || line.starts_with("+++ ")
            || line.starts_with("index ")
            || line.starts_with("new file ")
            || line.starts_with("deleted file ")
            || line.starts_with("rename ")
            || line.starts_with("similarity ")
            || line.starts_with("Binary files ");
        if line.starts_with("@@") {
            blocks.push(std::mem::take(&mut cur));
            cur.push_str(line);
            in_hunk = true;
        } else if is_file_hdr && in_hunk {
            blocks.push(std::mem::take(&mut cur));
            cur.push_str(line);
            in_hunk = false;
        } else {
            cur.push_str(line);
        }
    }
    if !cur.is_empty() {
        blocks.push(cur);
    }

    // We already know the whole diff overflows, so a marker will likely be needed --
    // reserve for it up front so a late drop can't push the result back over budget.
    const MARKER_TOKENS: usize = 48;
    let inner_budget = budget.saturating_sub(MARKER_TOKENS);

    // Count each block once (O(n) encodes), keep while the running sum fits. BPE is not
    // additive across a join, but the drift is a token or two per boundary -- far under
    // the marker reserve -- so a single accumulator is accurate enough here.
    let mut kept = String::new();
    let mut used = 0usize;
    let mut dropped = 0usize;
    for b in &blocks {
        if b.is_empty() {
            continue;
        }
        let bt = enc.count(b)?;
        if !b.starts_with("@@") || used + bt <= inner_budget {
            kept.push_str(b);
            used += bt;
        } else {
            dropped += 1;
        }
    }
    if dropped > 0 {
        kept.push_str(&format!(
            "@@ ... @@ [zecor-tokcount: {dropped} hunk(s) trimmed to fit {budget} tokens]\n"
        ));
    }
    Ok(kept)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cl100k_counts_are_stable() {
        let e = Encoder::load("cl100k_base").unwrap();
        assert_eq!(e.count("hello world").unwrap(), 2);
        assert!(e.count("").unwrap() == 0);
        let long = "the quick brown fox ".repeat(50);
        assert!(e.count(&long).unwrap() > 100);
    }

    #[test]
    fn trim_hits_the_budget() {
        let e = Encoder::load("cl100k_base").unwrap();
        let text = "one two three four five six seven eight nine ten ".repeat(20);
        let trimmed = e.trim(&text, 10).unwrap();
        assert!(e.count(&trimmed).unwrap() <= 10);
        assert!(e.trim("short", 100).unwrap() == "short");
    }

    #[test]
    fn unknown_encoding_errors() {
        assert!(Encoder::load("not-a-real-encoding").is_err());
    }

    #[test]
    fn every_builtin_encoding_loads_and_counts() {
        for spec in [
            "cl100k_base",
            "o200k_base",
            "p50k_base",
            "r50k_base",
            "gpt2",
        ] {
            let e = Encoder::load(spec).unwrap_or_else(|_| panic!("load {spec}"));
            assert!(
                e.count("def add(a, b):\n    return a + b\n").unwrap() > 4,
                "{spec}"
            );
        }
    }

    #[test]
    fn trim_is_a_noop_at_or_under_budget() {
        let e = Encoder::load("cl100k_base").unwrap();
        let text = "exactly some words here";
        let n = e.count(text).unwrap();
        assert_eq!(e.trim(text, n).unwrap(), text);
        assert_eq!(e.trim(text, n + 100).unwrap(), text);
        assert_eq!(e.trim("", 0).unwrap(), "");
    }

    #[test]
    fn trim_never_returns_invalid_utf8_on_a_multibyte_boundary() {
        let e = Encoder::load("cl100k_base").unwrap();
        // emoji + CJK: token boundaries do not line up with char boundaries
        let text = "🚀 контейнер 日本語 サンドボックス ".repeat(30);
        for budget in [1usize, 2, 3, 5, 8, 13, 21] {
            let out = e.trim(&text, budget).unwrap(); // must not panic
            assert!(e.count(&out).unwrap() <= budget);
            assert!(std::str::from_utf8(out.as_bytes()).is_ok());
        }
    }

    #[test]
    fn pack_with_a_zero_budget_drops_everything() {
        let e = Encoder::load("cl100k_base").unwrap();
        let files = vec![("a".into(), "x".into()), ("b".into(), "y".into())];
        let r = pack(&e, &files, 0).unwrap();
        assert!(r.text.is_empty() && r.included.is_empty());
        assert_eq!(r.dropped.len(), 2);
    }

    #[test]
    fn pack_keeps_priority_order_of_included_files() {
        let e = Encoder::load("cl100k_base").unwrap();
        let files = vec![
            ("first.txt".into(), "aaa ".repeat(10)),
            ("second.txt".into(), "bbb ".repeat(10)),
            ("third.txt".into(), "ccc ".repeat(10)),
        ];
        let r = pack(&e, &files, 10_000).unwrap();
        assert_eq!(r.included, ["first.txt", "second.txt", "third.txt"]);
        let fp = r.text.find("first.txt").unwrap();
        let sp = r.text.find("second.txt").unwrap();
        assert!(fp < sp);
    }

    #[test]
    fn diff_trim_headers_only_is_unchanged() {
        let e = Encoder::load("cl100k_base").unwrap();
        let d = "diff --git a/x b/x\nindex 111..222 100644\n--- a/x\n+++ b/x\n";
        assert_eq!(diff_trim(&e, d, 1).unwrap(), d); // nothing droppable
    }

    #[test]
    fn cost_local_is_free_and_override_beats_the_table() {
        let c = cost("local", 5_000_000, 5_000_000, None).unwrap();
        assert_eq!(c.total_usd, 0.0);
        let o = cost("gpt-4o", 1_000_000, 0, Some((99.0, 0.0))).unwrap();
        assert!((o.in_usd - 99.0).abs() < 1e-9); // override, not the 2.50 table price
    }

    #[test]
    fn price_lookup_prefers_the_longer_prefix() {
        assert_eq!(price_per_mtok("gpt-4o-mini"), Some((0.15, 0.60)));
        assert_eq!(price_per_mtok("gpt-4o"), Some((2.50, 10.00)));
        assert_eq!(
            price_per_mtok("claude-3-5-haiku-latest"),
            Some((0.80, 4.00))
        );
        assert_eq!(price_per_mtok("totally-unknown"), None);
    }

    #[test]
    fn cost_resolves_by_prefix_and_math_is_right() {
        let c = cost("gpt-4o-2024-08-06", 1_000_000, 500_000, None).unwrap();
        assert!((c.in_usd - 2.50).abs() < 1e-9);
        assert!((c.out_usd - 5.00).abs() < 1e-9);
        assert!((c.total_usd - 7.50).abs() < 1e-9);
        assert!(cost("no-such-model", 10, 10, None).is_err());
        let o = cost("whatever", 1_000_000, 0, Some((1.0, 2.0))).unwrap();
        assert!((o.in_usd - 1.0).abs() < 1e-9);
    }

    #[test]
    fn pack_fits_budget_and_skips_whole_files() {
        let e = Encoder::load("cl100k_base").unwrap();
        let files = vec![
            ("a.txt".to_string(), "alpha ".repeat(20)),
            ("big.txt".to_string(), "huge ".repeat(400)),
            ("b.txt".to_string(), "beta ".repeat(10)),
        ];
        let r = pack(&e, &files, 120).unwrap();
        assert!(r.tokens <= 120);
        assert!(r.included.contains(&"a.txt".to_string()));
        assert!(r.dropped.contains(&"big.txt".to_string()));
        assert!(r.included.contains(&"b.txt".to_string())); // smaller file still fits
    }

    #[test]
    fn diff_trim_keeps_headers_and_marks_drops() {
        let e = Encoder::load("cl100k_base").unwrap();
        let mut d = String::from("diff --git a/x.py b/x.py\n--- a/x.py\n+++ b/x.py\n");
        for i in 0..8 {
            d.push_str(&format!(
                "@@ -{0},3 +{0},4 @@\n context\n-old{1}\n+new{1}\n+extra{1}\n",
                i * 10 + 1,
                i
            ));
        }
        let full = e.count(&d).unwrap();
        let trimmed = diff_trim(&e, &d, full / 2).unwrap();
        assert!(e.count(&trimmed).unwrap() <= full / 2);
        assert!(trimmed.contains("diff --git a/x.py b/x.py"));
        assert!(trimmed.contains("trimmed to fit"));
        // unchanged when already under budget
        assert_eq!(diff_trim(&e, &d, full + 10).unwrap(), d);
    }
}
