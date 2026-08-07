#!/usr/bin/env bash
# Capture the evidence required for a real-device SOMA release audit.
set -u
set -o pipefail

usage() {
    cat <<'USAGE'
Usage: scripts/release-audit.sh [OPTIONS]

Options:
  --allow-dirty       Record, rather than reject, a dirty worktree.
  --quick             Run only the focused Metal suites and Clippy. This is
                      a preflight and is NOT sufficient release evidence.
  --dry-run           Record commands and metadata, but execute no cargo work.
                      This is NOT sufficient release evidence.
  --output-dir DIR    Write artifacts under DIR (default: docs/measurements).
  -h, --help          Show this help.

The default mode runs the complete all-feature tests, Clippy with warnings
forbidden, focused real-Metal suites, and bounded release benchmarks. Every
invocation creates a new UTC-stamped .log and adjacent .sha256 file.
USAGE
}

allow_dirty=0
quick=0
dry_run=0
output_dir=""
while (($#)); do
    case "$1" in
        --allow-dirty) allow_dirty=1 ;;
        --quick) quick=1 ;;
        --dry-run) dry_run=1 ;;
        --output-dir)
            shift
            if (($# == 0)); then
                echo "release-audit: --output-dir requires a directory" >&2
                exit 2
            fi
            output_dir=$1
            ;;
        -h|--help) usage; exit 0 ;;
        *) echo "release-audit: unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
    shift
done

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
repo=$(git -C "$script_dir/.." rev-parse --show-toplevel 2>/dev/null) || {
    echo "release-audit: script is not inside a Git worktree" >&2
    exit 2
}
cd "$repo" || exit 2

# Check before creating an artifact in the worktree, so the audit file itself
# cannot make an initially clean checkout appear dirty.
initial_status=$(git status --porcelain=v1 --untracked-files=all)
if [[ -n "$initial_status" && $allow_dirty -ne 1 ]]; then
    echo "release-audit: refusing dirty worktree; inspect 'git status' or use --allow-dirty" >&2
    printf '%s\n' "$initial_status" >&2
    exit 2
fi

if [[ $dry_run -ne 1 && $(uname -s) != Darwin ]]; then
    echo "release-audit: a qualifying capture requires a real macOS Metal device" >&2
    echo "release-audit: use --dry-run only to inspect the plan" >&2
    exit 2
fi
for tool in git date shasum tee; do
    command -v "$tool" >/dev/null 2>&1 || {
        echo "release-audit: required tool not found: $tool" >&2
        exit 2
    }
done
if [[ $dry_run -ne 1 ]]; then
    for tool in rustc cargo sw_vers system_profiler; do
        command -v "$tool" >/dev/null 2>&1 || {
            echo "release-audit: required tool not found: $tool" >&2
            exit 2
        }
    done
    gpu_probe=$(system_profiler SPDisplaysDataType) || {
        echo "release-audit: system_profiler could not identify the GPU" >&2
        exit 2
    }
    if ! grep -q 'Metal Support:' <<<"$gpu_probe"; then
        echo "release-audit: no Metal-capable physical GPU reported by system_profiler" >&2
        exit 2
    fi
fi

if [[ -z "$output_dir" ]]; then
    output_dir="$repo/docs/measurements"
elif [[ "$output_dir" != /* ]]; then
    output_dir="$repo/$output_dir"
fi
mkdir -p "$output_dir" || exit 2
stamp=$(date -u +%Y%m%dT%H%M%SZ)
mode=full
[[ $quick -eq 1 ]] && mode=quick
[[ $dry_run -eq 1 ]] && mode="${mode}-dry-run"
log="$output_dir/RELEASE-AUDIT-${stamp}-${mode}.log"
sum="$log.sha256"
# Timestamp resolution is one second. Refuse collision rather than append or
# truncate, preserving the immutability of prior evidence.
if ! (set -o noclobber; : >"$log") 2>/dev/null; then
    echo "release-audit: artifact already exists: $log" >&2
    exit 2
fi

print_command() {
    printf '$'
    printf ' %q' "$@"
    printf '\n'
}

run() {
    print_command "$@"
    if [[ $dry_run -eq 1 ]]; then
        return 0
    fi
    "$@"
}

audit_body() {
    echo "SOMA REAL-DEVICE RELEASE AUDIT"
    echo "schema: 1"
    echo "mode: $mode"
    echo "qualifying_release_evidence: $([[ $quick -eq 0 && $dry_run -eq 0 && -z "$initial_status" ]] && echo yes || echo no)"
    echo "started_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "repository: $repo"
    echo "commit: $(git rev-parse HEAD)"
    echo "tree_clean_at_start: $([[ -z "$initial_status" ]] && echo yes || echo no)"
    if [[ -n "$initial_status" ]]; then
        echo "----- git status --porcelain=v1 --untracked-files=all -----"
        printf '%s\n' "$initial_status"
    fi

    echo "----- rustc -Vv -----"
    if command -v rustc >/dev/null 2>&1; then rustc -Vv || return; else echo "unavailable (dry-run host)"; fi
    echo "----- cargo -Vv -----"
    if command -v cargo >/dev/null 2>&1; then cargo -Vv || return; else echo "unavailable (dry-run host)"; fi
    echo "----- sw_vers -----"
    if command -v sw_vers >/dev/null 2>&1; then sw_vers || return; else echo "unavailable (dry-run host)"; fi
    echo "----- uname -a -----"
    uname -a
    echo "----- system_profiler SPDisplaysDataType (GPU identity) -----"
    if command -v system_profiler >/dev/null 2>&1; then
        system_profiler SPDisplaysDataType || return
    else
        echo "unavailable (dry-run host)"
    fi

    echo "===== VALIDATION ====="
    if [[ $quick -eq 0 ]]; then
        run cargo test --all-features || return
    else
        echo "SKIPPED in quick mode: complete cargo test --all-features"
    fi
    run cargo clippy --all-targets --all-features -- -D warnings || return

    # These overlap the full suite deliberately: their separate commands make
    # it unambiguous in the raw log that cfg-gated real Metal paths ran.
    echo "===== EXPLICIT REAL-METAL SUITES ====="
    run cargo test --all-features --test metal_backend || return
    run cargo test --all-features --test device_scheduler || return
    run cargo test --all-features --test f32_evaluator || return
    run cargo test --features metal metal_resident_sync --lib || return

    echo "===== SELECTED BOUNDED RELEASE BENCHMARKS ====="
    if [[ $quick -eq 0 ]]; then
        run cargo run --release --all-features --example resident_dynamic_bench || return
        # Two trials is the harness minimum and keeps release capture bounded;
        # never use the expensive --full dataset in this audit script.
        run cargo run --release --all-features --example ant_collective_bench -- 2 || return
    else
        echo "SKIPPED in quick mode: release benchmarks"
    fi
}

# Complete the pipe before hashing, so the digest always covers every emitted
# byte, including the final result marker.
set +e
{
    audit_body
    body_status=$?
    echo "finished_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "result: $([[ $body_status -eq 0 ]] && echo PASS || echo FAIL)"
    exit "$body_status"
} 2>&1 | tee -a "$log"
status=${PIPESTATUS[0]}
set -e

(
    cd "$(dirname "$log")" || exit 1
    shasum -a 256 "$(basename "$log")" >"$(basename "$sum")"
)
checksum_status=$?
echo "raw log: $log"
echo "sha256: $sum"
if [[ $checksum_status -ne 0 ]]; then
    echo "release-audit: failed to write SHA-256" >&2
    exit "$checksum_status"
fi
exit "$status"
