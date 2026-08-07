# Real-device release audit

`scripts/release-audit.sh` is the G6 capture entry point. Run it on the physical
macOS machine whose Apple GPU is being qualified, from a clean checkout of the
release commit:

```sh
scripts/release-audit.sh
```

The default capture runs:

- `cargo test --all-features`;
- `cargo clippy --all-targets --all-features -- -D warnings`;
- the `metal_backend`, `device_scheduler`, and `f32_evaluator` integration-test
  targets plus the `metal_resident_sync` unit filter separately, so the raw
  record explicitly shows the real-Metal suites;
- bounded release builds/runs of `resident_dynamic_bench` and the small
  two-trial `ant_collective_bench` dataset.

It does **not** run `cargo fmt`, the long benchmark sweeps, or
`ant_collective_bench --full`. The complete default command can still take time;
do not run it on a laptop that is thermally constrained or on battery power.

## Artifacts and provenance

Each invocation reserves a new UTC-stamped
`docs/measurements/RELEASE-AUDIT-*.log` and never appends to or overwrites an
older capture. Beside it is a `.log.sha256` file in `shasum -a 256` format.
The raw log includes the exact Git commit, initial porcelain status, `rustc -Vv`,
`cargo -Vv`, `sw_vers`, `uname`, and the complete
`system_profiler SPDisplaysDataType` GPU identity, followed by command output
and the final status. Verify a frozen capture with:

```sh
cd docs/measurements
shasum -a 256 -c RELEASE-AUDIT-<timestamp>-full.log.sha256
```

The script refuses any tracked or untracked worktree change by default. For an
investigative capture only, `--allow-dirty` records the complete initial
porcelain status in the log. Such a capture must not be presented as a clean
release capture.

## Safe preflight modes

Inspect the command plan without running tests, Clippy, or benchmarks:

```sh
scripts/release-audit.sh --dry-run
```

Run Clippy and only the explicitly named Metal suites, skipping the aggregate
test command and benchmarks:

```sh
scripts/release-audit.sh --quick
```

Both modes are labeled `qualifying_release_evidence: no` in their raw logs.
`--dry-run` may be used off macOS to inspect the plan; every executing mode
requires macOS so cfg-gated Metal tests cannot silently become zero-test
successes. Use `--output-dir DIR` for disposable preflight artifacts.

A release audit is complete only when the default full-mode log ends in
`result: PASS`, its checksum verifies, the initial tree was clean, and a human
has reviewed the G1–G6 gate status in `DREAM-COMPLETION.md`.

## Qualifying capture

The first qualifying capture is
`measurements/RELEASE-AUDIT-20260807T090150Z-full.log` for commit
`8f48bb6cdea2eed19470c652107fd0dc15b9dc65`; its adjacent checksum verifies and
the log records `tree_clean_at_start: yes`, `qualifying_release_evidence: yes`,
and `result: PASS` on the physical 16-core Apple M4 Pro. Its resident benchmark
ordering is still unstable and its ant Metal wall is slower, so the capture
closes release provenance (G6), not the performance gate (G5).
