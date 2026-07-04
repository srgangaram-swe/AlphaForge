# Execution Engine (C++ Core)

## Why it exists

The daily-bar research pipeline does not need nanosecond matching, and this
document says so up front. The native core exists for two honest reasons:

1. **Better fills in simulation.** A flat slippage-in-bps assumption ignores
   order size. The depth-aware `simulate_fill` walks a synthetic book, so
   larger simulated orders pay more — slippage becomes an emergent property
   of consuming liquidity instead of a constant.
2. **Systems engineering demonstration.** Matching engines are the canonical
   trading-infrastructure exercise: data-structure choice under latency
   constraints, integer determinism, FFI, and cross-implementation testing.

## Design

`cpp/include/alphaforge/order_book.hpp` — header-only, C++17, no dependencies.

- **Price-time priority.** Bids and asks are `std::map<price, Level>`
  (descending / ascending). Each `Level` holds a FIFO `std::list<Order>` and
  an aggregate quantity.
- **O(1) cancel.** An `unordered_map<order_id, (side, price, list iterator)>`
  index; `std::list` iterators stay valid under erasure, so cancels never scan.
- **Integer ticks and quantities.** No floating point in the matching path:
  behaviour is exactly reproducible across platforms and bit-identical to the
  Python reference.
- **Matching semantics.** Limit orders fill at the resting (maker) price while
  crossed; remainders rest. Market orders walk the opposite side; unfilled
  remainders are dropped. Fills are recorded as
  `(maker_id, taker_id, price, qty)` in match order. No self-match prevention
  (single-strategy simulation does not need it).

`alphaforge/execution/orderbook_py.py` is a pure-Python implementation of the
same contract; `alphaforge/execution/native.py` dispatches to whichever is
available. `tests/test_orderbook.py` drives both with identical random flow
(4,000 mixed operations) and requires identical order ids, fills, depth,
volumes, and best quotes.

## Benchmarks

Two benchmarks, because they answer different questions:

- `make bench-native` — pure C++ (`cpp/benchmarks/bench_main.cpp`): true
  in-process cost per operation with sampled latency percentiles.
- `make bench` — drives both engines from Python: what a Python caller
  actually observes, pybind11 overhead included.

Representative numbers (Apple M-series, 2M ops, 55% add / 30% cancel /
15% market):

| engine | throughput | p50 | p99 |
|---|---|---|---|
| C++ (in-process) | ~6.2M ops/s | ~125 ns | ~583 ns |
| C++ via pybind11 | ~1.8M ops/s | — | — |
| pure Python | ~1.1M ops/s | — | — |

The pybind11 row is the honest caveat: per-call FFI overhead dominates when
you drive a nanosecond-scale engine one op at a time from Python. Real systems
batch at the boundary or keep the hot loop native.

## Build

```bash
make native        # one-command build via scripts/build_native.py (needs pybind11)
# or the full CMake project:
cmake -S cpp -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build
```

Everything degrades gracefully: without the extension, the Python reference
implementation serves the same API and all non-parity tests still pass.
