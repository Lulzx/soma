# Speculative concurrent epochs

SOMA now has two Phase-F implementations:

- `EpochExecutive::Reference` runs lanes sequentially in the configured order.
- `EpochExecutive::Speculative { max_lanes }` runs an eligible epoch on real OS
  threads, validates the resulting history, and either commits it canonically
  or discards it and invokes the reference loop.

The reference executive remains normative. Optimism is an implementation
choice and does not weaken the machine semantics.

## Execution protocol

For an epoch containing between two and `max_lanes` lanes, the optimistic path:

1. clones the same pre-Phase-F kernel snapshot for every lane;
2. assigns each worker its plan-derived lane number and allocator partition;
3. executes the handler on a scoped OS thread;
4. records every operation available through `LaneView`, its resource accesses,
   final mutated object bytes, and step result (lane-local effects remain in the
   disposable snapshot and are reproduced by replay);
5. rejects histories with read/write or write/write conflicts;
6. replays the concrete operation journals on a disposable kernel copy;
7. if validation succeeds, replays them on the real kernel in lane-number order
   and applies `StepResult`s through the ordinary commit path;
8. otherwise reruns every lane with the reference executive from the untouched
   pre-Phase-F state.

The disposable replay is important. An incomplete access declaration becomes a
fallback, not a partial mutation of the real machine.

## Recorded resources

The validator tracks objects, futures, processes/capability spaces, mailboxes,
domains, and allocator partitions. A pair conflicts when one lane writes a
resource another lane reads or writes. Process commit is included, so two
read-only continuations of one process correctly conflict even when their frame
objects differ. A supervised child also writes its supervisor resource.

All fifteen `LaneView` operations have concrete journal forms. This includes the
four cross-lane operations that §4.10 identified: message enqueue/receive and
future resolve/await. Channels are absent from the set because `LaneView` does
not expose a channel operation; adding one requires adding its resource and
replay case in the same change.

Allocation is optimistic when lanes use distinct configured partitions. A
shared allocator partition conflicts. Process creation additionally writes its
domain, so bounded and unbounded process-count accounting remains canonical.

Faulting lanes and processes already marked `CancelPending` currently replay
through the reference path. Their containment footprint is deliberately treated
as global until it has an equally closed resource declaration.

## Configuration and measurements

```rust
use soma::kernel::speculation::EpochExecutive;

kernel.configure_epoch_executive(EpochExecutive::Speculative { max_lanes: 8 });
let stats = kernel.speculation_stats();
```

`SpeculationStats` reports attempted, committed, and fallback epochs; speculative
and committed lanes; and conflict versus unsupported fallbacks. Epochs outside
the configured lane bound use the reference path without counting as failed
speculation.

`examples/speculative_epoch_report.rs` measures real wall-clock time. On the
local 12-logical-core Apple M4 Pro, release medians over nine runs were:

| lanes | arithmetic ops/lane | reference | optimistic | speedup |
| ---: | ---: | ---: | ---: | ---: |
| 2 | 100,000 | 0.207 ms | 0.156 ms | 1.32× |
| 4 | 100,000 | 0.426 ms | 0.234 ms | 1.82× |
| 8 | 100,000 | 0.838 ms | 0.267 ms | 3.14× |
| 2 | 1,000,000 | 2.098 ms | 1.115 ms | 1.88× |
| 4 | 1,000,000 | 4.270 ms | 1.177 ms | 3.63× |
| 8 | 1,000,000 | 7.613 ms | 1.317 ms | 5.78× |

At 1,000 ops/lane the optimistic path is about 0.1× the reference speed, and at
10,000 it remains slower. Snapshot cloning, validation replay, and thread
startup are real fixed costs. This implementation is therefore opt-in and most
useful for coarse independent discovery computations; it is not evidence that
all epochs should execute speculatively.

Run the complete sweep with:

```sh
cargo run --release --example speculative_epoch_report
```

## Correctness tests

`tests/speculative_epochs.rs` covers:

- disjoint compute lanes committing the reference trace;
- same-process commit conflicts;
- allocator/domain conflicts;
- independent message, future, and allocation operations committing;
- two writers of one future falling back and reproducing plan order.

Every accepted execution uses the ordinary authority checks, trace emission,
effect production, continuation commit, and epoch effect applier during
canonical replay. The optimization changes where handler arithmetic runs, not
which semantic transition code publishes its outcome.
