// SPDX-License-Identifier: Apache-2.0
//! `zecor-tokcount` -- exact token math for context and cost control.
//!
//!   zecor-tokcount count [--enc E]                     stdin -> {"tokens","chars","encoding"}
//!   zecor-tokcount trim  --max N [--enc E]             stdin -> text truncated to N tokens
//!   zecor-tokcount cost  --model M [--in-tokens N]     price a request (stdin = input text
//!                        [--out-tokens N] [--in-price X] [--out-price Y]   if --in-tokens omitted)
//!   zecor-tokcount pack  --budget N [--enc E] FILE...  concat files that fit N tokens
//!   zecor-tokcount diff-trim --max N [--enc E]         stdin diff -> diff trimmed to N tokens
//!
//! `--enc` is a built-in (cl100k_base | o200k_base | p50k_base | r50k_base) or a path to
//! a tokenizer.json. `count`/`cost`/`pack` print JSON; `trim`/`diff-trim` print text.

use std::io::{self, Read};
use zecor_tokcount::{cost, diff_trim, pack, Encoder};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mode = args.first().map(String::as_str);
    let enc = flag(&args, "--enc").unwrap_or_else(|| "cl100k_base".to_string());

    match mode {
        Some("count") => {
            let e = load(&enc);
            let text = read_stdin();
            let n = e.count(&text).unwrap_or_else(|err| die(&err.to_string()));
            println!(
                "{}",
                serde_json::json!({ "tokens": n, "chars": text.chars().count(), "encoding": enc })
            );
        }
        Some("trim") => {
            let e = load(&enc);
            let max = req_usize(&args, "--max");
            match e.trim(&read_stdin(), max) {
                Ok(s) => print!("{s}"),
                Err(err) => die(&err.to_string()),
            }
        }
        Some("cost") => {
            let model =
                flag(&args, "--model").unwrap_or_else(|| die("cost: --model M is required"));
            let e = load(&enc);
            let in_tokens = match opt_usize(&args, "--in-tokens") {
                Some(n) => n,
                None => e
                    .count(&read_stdin())
                    .unwrap_or_else(|err| die(&err.to_string())),
            };
            let out_tokens = opt_usize(&args, "--out-tokens").unwrap_or(0);
            let price = match (opt_f64(&args, "--in-price"), opt_f64(&args, "--out-price")) {
                (Some(i), Some(o)) => Some((i, o)),
                _ => None,
            };
            match cost(&model, in_tokens, out_tokens, price) {
                Ok(c) => println!("{}", serde_json::to_string(&c).expect("cost serializes")),
                Err(err) => die(&err.to_string()),
            }
        }
        Some("pack") => {
            let e = load(&enc);
            let budget = req_usize(&args, "--budget");
            let paths: Vec<String> = args
                .iter()
                .enumerate()
                .skip(1)
                .filter(|(i, a)| !a.starts_with("--") && !value_of_flag(&args, *i))
                .map(|(_, a)| a.clone())
                .collect();
            let files: Vec<(String, String)> = paths
                .iter()
                .map(|p| {
                    let body = std::fs::read_to_string(p)
                        .unwrap_or_else(|err| die(&format!("{p}: {err}")));
                    (p.clone(), body)
                })
                .collect();
            let r = pack(&e, &files, budget).unwrap_or_else(|err| die(&err.to_string()));
            eprintln!(
                "[zecor-tokcount pack: {} tokens, {} file(s) in, {} dropped: {}]",
                r.tokens,
                r.included.len(),
                r.dropped.len(),
                r.dropped.join(", ")
            );
            print!("{}", r.text);
        }
        Some("diff-trim") => {
            let e = load(&enc);
            let max = req_usize(&args, "--max");
            match diff_trim(&e, &read_stdin(), max) {
                Ok(s) => print!("{s}"),
                Err(err) => die(&err.to_string()),
            }
        }
        _ => {
            eprintln!(
                "usage: zecor-tokcount <count | trim --max N | cost --model M | \
                 pack --budget N FILE... | diff-trim --max N> [--enc <encoding|path>]"
            );
            std::process::exit(2);
        }
    }
}

fn load(enc: &str) -> Encoder {
    Encoder::load(enc).unwrap_or_else(|e| die(&e.to_string()))
}

fn read_stdin() -> String {
    let mut s = String::new();
    io::stdin().read_to_string(&mut s).ok();
    s
}

const VALUE_FLAGS: &[&str] = &[
    "--enc",
    "--max",
    "--budget",
    "--model",
    "--in-tokens",
    "--out-tokens",
    "--in-price",
    "--out-price",
];

fn flag(args: &[String], name: &str) -> Option<String> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .cloned()
}

fn value_of_flag(args: &[String], idx: usize) -> bool {
    idx > 0 && VALUE_FLAGS.contains(&args[idx - 1].as_str())
}

fn opt_usize(args: &[String], name: &str) -> Option<usize> {
    flag(args, name).and_then(|v| v.parse().ok())
}

fn opt_f64(args: &[String], name: &str) -> Option<f64> {
    flag(args, name).and_then(|v| v.parse().ok())
}

fn req_usize(args: &[String], name: &str) -> usize {
    opt_usize(args, name).unwrap_or_else(|| die(&format!("{name} N is required")))
}

fn die(msg: &str) -> ! {
    eprintln!("zecor-tokcount: {msg}");
    std::process::exit(2);
}
