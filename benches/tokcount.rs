// SPDX-License-Identifier: Apache-2.0
//! `cargo bench` -- count / trim / pack / diff-trim on realistic inputs.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use zecor_tokcount::{diff_trim, pack, Encoder};

fn source_blob(kb: usize) -> String {
    let unit =
        "def handler(req, ctx):\n    # validate, then dispatch\n    if not req.ok:\n        \
                return Err(422)\n    return route(req.path, ctx.session)\n\n";
    unit.repeat((kb * 1024) / unit.len() + 1)
}

fn synthetic_diff(hunks: usize) -> String {
    let mut d =
        String::from("diff --git a/service.py b/service.py\n--- a/service.py\n+++ b/service.py\n");
    for i in 0..hunks {
        d.push_str(&format!(
            "@@ -{0},4 +{0},6 @@\n context line\n-  old_{1}()\n+  new_{1}()\n+  extra_{1}()\n+  more_{1}()\n",
            i * 12 + 1, i
        ));
    }
    d
}

fn bench_count(c: &mut Criterion) {
    let e = Encoder::load("cl100k_base").unwrap();
    let text = source_blob(64);
    c.bench_function("count/cl100k/64KiB", |b| {
        b.iter(|| e.count(black_box(&text)).unwrap())
    });
}

fn bench_trim(c: &mut Criterion) {
    let e = Encoder::load("cl100k_base").unwrap();
    let text = source_blob(64);
    c.bench_function("trim/cl100k/64KiB->2k", |b| {
        b.iter(|| e.trim(black_box(&text), 2_000).unwrap())
    });
}

fn bench_pack(c: &mut Criterion) {
    let e = Encoder::load("cl100k_base").unwrap();
    let files: Vec<(String, String)> = (0..50)
        .map(|i| (format!("mod_{i}.py"), source_blob(4)))
        .collect();
    c.bench_function("pack/50 files/8k budget", |b| {
        b.iter(|| pack(&e, black_box(&files), 8_000).unwrap())
    });
}

fn bench_diff_trim(c: &mut Criterion) {
    let e = Encoder::load("cl100k_base").unwrap();
    let d = synthetic_diff(200);
    let half = e.count(&d).unwrap() / 2;
    c.bench_function("diff_trim/200 hunks->half", |b| {
        b.iter(|| diff_trim(&e, black_box(&d), half).unwrap())
    });
}

criterion_group!(
    benches,
    bench_count,
    bench_trim,
    bench_pack,
    bench_diff_trim
);
criterion_main!(benches);
