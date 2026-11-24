# ECTSP: embedded core test scheduling for SOCs, in Julia

Given the scan chains, terminals and test patterns of the cores of an SOC and `W` test access
mechanism (TAM) wires, partition the wires into test buses, assign every core to one bus and
schedule the tests so that the SOC test time is minimal. This is the embedded core-based SOC test
scheduling problem (ECTSP), formulated as a network transportation problem: source -> core -> TAM
-> sink, where a core's test may not be split over TAMs (single source unsplittable flow).

This repository has the benchmark reader, the wrapper design that turns scan chains into test
times, a mixed-integer baseline in JuMP, and the single source unsplittable flow solver
(Dinitz-Garg-Goemans), and the greedy, best fit and shortest-job-first heuristics, all on the same
`TestProblem` / `Schedule` types.

## Layout

| Path | Contents |
|---|---|
| `benchmarks/itc02/` | the twelve ITC'02 SOC Test Benchmarks (`.soc`), unmodified |
| `src/itc02.jl` | `parse_soc`, `load_benchmark`, `benchmark_names` |
| `src/wrapper.jl` | `wrapper_lengths`, `test_time`: test time of a core on `w` TAM wires |
| `src/problem.jl` | `TestProblem` (table `T[core, width]`), `lower_bound`, `Schedule`, `validate` |
| `src/milp.jl` | `solve_assignment` (widths fixed), `solve_baseline` (widths and assignment) |
| `src/maxflow.jl` | integer maximum flow (Dinic) |
| `src/ufp.jl` | `unsplittable_flow`: Dinitz-Garg-Goemans on any acyclic flow |
| `src/ufp_schedule.jl` | `solve_ufp`, `ufp_fractional_bound`, `width_partitions` |
| `src/heuristics.jl` | `solve_greedy`, `solve_best_fit`, `solve_sjf` (`solve_heuristic`) |
| `scripts/run_baseline.jl` | MILP sweep over benchmarks, widths and TAM counts; writes `results/` |
| `scripts/run_ufp.jl` | the same sweep with the unsplittable flow solver, compared with the MILP |
| `scripts/run_heuristics.jl` | unsplittable flow, greedy, best fit and SJF side by side, with a summary |
| `envs/cuopt/` | environment with cuOpt.jl for the GPU solver |
| `results/` | measured results (`baseline_<host>.csv`, `.md`) |

## Use

```julia
using ECTSP
soc = load_benchmark("d695")                  # or parse_soc("path/to/file.soc")
p   = TestProblem(soc; max_width=32)          # p.T[i, w]: cycles of core i on a TAM of width w
pareto_curve(p, 4)                            # Pareto-optimal (width, time) points of core 4
p2  = TestProblem("mysoc", [[(1, 900), (2, 460), (4, 240)], [(1, 300), (3, 110)]]; max_width=32)   # curves given
lower_bound(p, 32)                            # architecture-independent bound for W = 32
s   = solve_baseline(p, 32, 3)                # 32 wires in exactly 3 TAMs  -> Schedule
best, all = solve_baseline(p, 32; max_tams=4) # B = 1..4, best one first
print_schedule(best, p)                       # widths, loads, start/finish of every core
solve_assignment(p, [8, 8, 16])               # widths given: R || C_max

u   = solve_ufp(p, [8, 8, 16])                # unsplittable flow, widths given
u   = solve_ufp(p, 32, 3)                     # all partitions of 32 wires into 3 TAMs
best, all = solve_ufp(p, 32; max_tams=4)
ufp_fractional_bound(p, [8, 8, 16])           # lower bound for these widths

solve_greedy(p, [8, 8, 16]);  solve_best_fit(p, 32, 3);  solve_sjf(p, 32; max_tams=4)
solve_heuristic(p, 32, 3; method=:best_fit)   # the same three through one entry point
```

Solver options: `optimizer` (any MathOptInterface MILP optimizer, default `HiGHS.Optimizer`),
`time_limit` (seconds), `mip_gap`, `threads`, `silent`, `attributes` (pairs of raw solver options).
A `Schedule` carries the makespan, the solver's proven bound, the termination status and the time;
`gap(s)` is the relative gap. HiGHS reports `OPTIMAL` at its default relative gap of 1e-4; pass
`mip_gap=0.0` for a proof to the last cycle.

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. test/runtests.jl
julia -t auto --project=. scripts/run_baseline.jl                       # all benchmarks, W = 16, 32, 64, B = 1..4
julia -t auto --project=. scripts/run_baseline.jl --widths 24 --max-tams 6 --time-limit 300 d695 p22810
julia -t auto --project=. scripts/run_ufp.jl --compare results/baseline_spark-bc08.csv
julia -t auto --project=. scripts/run_heuristics.jl --compare results/baseline_spark-bc08.csv
```

### cuOpt on spark-bc08

cuOpt.jl 0.5 loads `libcuopt_mathopt.so` (cuOpt 26.10), which is in the miniconda site-packages:

```bash
export LD_LIBRARY_PATH=$HOME/miniconda3/lib/python3.13/site-packages/libcuopt_mathopt/lib64:$LD_LIBRARY_PATH
julia --project=envs/cuopt -e 'using Pkg; Pkg.instantiate()'
julia --project=envs/cuopt scripts/run_baseline.jl --solver cuopt --tag spark-bc08_cuopt
```

## Model

**Test time of a core.** A core on `w` TAM wires gets `w` wrapper chains, each a concatenation of
wrapper input cells, internal scan chains (not split) and wrapper output cells. With `s_i`, `s_o`
the longest scan-in and scan-out chain and `p` patterns, a test takes
`(1 + max(s_i, s_o)) * p + min(s_i, s_o)` cycles. Scan chains are partitioned by LPT, then the
inputs and outputs are levelled over the chains (`wrapper_lengths`); `T[i, w]` is the best over
`1..w` wires, so it never increases with `w`. Bidirectional terminals count as an input and an
output cell.

**Pareto curves are the input.** Every solver sees only the table `T[i, w]`, the staircase through
the Pareto-optimal `(width, time)` points of each core (`pareto_curve`). The wrapper design above
is one way to obtain the curves from an ITC'02 file; curves from any other source go in through
`TestProblem(name, curves)`, where a core on a TAM of width `w` takes the time of its widest point
not wider than `w`.

**Benchmark conventions.** The hierarchy is flattened (every module is a wrapped core). The level-0
module is left out unless `include_top=true`. A module's TAM tests run back to back; tests with
`ScanUse 0` use only the terminals. Tests with `TamUse 0` (BIST) take no TAM wires and are not
scheduled; their length is kept in `TestProblem.bist`. Power data (`h953`) is parsed and not used.

**Baseline MILP** (`solve_baseline(p, W, B)`):

```
u[j,w] in {0,1}    TAM j has width w             sum_w u[j,w] = 1,  sum_{j,w} w u[j,w] = W
x[i,j] in {0,1}    core i is on TAM j            sum_j x[i,j] = 1
a[i,j,w] in [0,1]  = x[i,j] u[j,w]               a[i,j,w] <= u[j,w],  sum_w a[i,j,w] = x[i,j]
                                                 sum_{i,w} T[i,w] a[i,j,w] <= C   for every j
min C                                            widths non-decreasing in j (symmetry)
```

Times are divided by the lower bound before they enter the model; the reported makespan is
recomputed in integers from the returned assignment and every schedule is checked by `validate`.
With the widths fixed (`solve_assignment`) only `x` remains and the problem is `R || C_max`.

**Lower bound** (`lower_bound(p, W)`): `max(max_i T[i, W], ceil(sum_i min_w w T[i, w] / W))`.
It holds for any TAM architecture, so the distance to it also contains what a test-bus
architecture with at most `B` buses cannot reach.

## Single source unsplittable flow

**The rounding** (`unsplittable_flow`, `src/ufp.jl`) takes any acyclic flow that carries the
demands from one source to the terminals and returns one path per terminal such that the demand
routed over an arc exceeds the given flow on that arc by less than the largest demand. Terminals
are moved from their vertices towards the source: a terminal of demand `d` crosses an incoming arc
that carries at least `d`. When none can move, flow is shifted around an alternating cycle
(decreased along forward paths, increased against singular arcs) until an arc empties or an
increased arc carries exactly the demand of a terminal at its head. An arc whose flow has been
increased is crossed only by a terminal of exactly that demand, after which it is gone; this is
what gives the bound (before the first increase the arc can only lose flow, afterwards one
terminal crosses). The routine counts the cases where a terminal would have to cross such an arc
with flow left over (`exceptions`); none occurred in 300 000 random graphs nor on the benchmarks,
and the test suite checks the bound on 3 000 random graphs.

**The scheduling network** for TAM widths `w_1..w_B`: `source -> TAM j -> core i`, a terminal of
demand `d_i = min_j T[i, w_j]` at every core. Core `i` is admissible on TAM `j` when
`T[i, w_j] <= rho * d_i` and `T[i, w_j] <= C`. The smallest `C` with a fractional flow of at most
`C` per TAM is found by bisection over a maximum flow, and that flow is rounded. Then:

- every TAM carries at most `C + max_i d_i` in demand units, so the test time is at most
  `rho * (C + max_i d_i)`;
- with TAMs of one width (or `rho = 1`) the demands are the true times: `C <= OPT`, `d_i <= OPT`,
  the schedule is within a factor 2 of the optimum for those widths;
- with `rho = Inf` the fractional `C` is a lower bound on the optimum for those widths
  (`ufp_fractional_bound`), because demands only under-estimate the times.

On TAMs of different widths a core is slower than `d_i` on the narrow ones, which the flow does not
see. `solve_ufp` therefore rounds once per `rho` in `UFP_SLOWDOWNS` (1 to 3, and `Inf`) and keeps
the schedule with the smallest true test time; `slowdowns=(Inf,)` gives the plain formulation.
`solve_ufp(p, W, B)` does this for every partition of `W` into `B` widths (1906 partitions for
`W = 64`, `B = 4`), in parallel over Julia threads.

## Greedy, best fit, shortest job first

For given TAM widths, with `d_i` the time of core `i` on its fastest TAM (`src/heuristics.jl`):

- **greedy**: cores by decreasing `d_i`; each goes to the TAM on which it finishes earliest.
- **shortest job first**: the same rule with the cores by increasing `d_i`.
- **best fit**: for a target test time `C`, cores by decreasing `d_i`; each goes to the TAM it
  fills best (least of `C` left after it). The smallest `C` for which every core fits is found by
  bisection.

As for the unsplittable flow, `solve_*(p, W, B)` applies the rule to every partition of `W` into
`B` widths and keeps the best. None of the three carries a guarantee on TAMs of different widths.

## Results

Full comparison of all solvers (test time, excess over the MILP, run time, findings):
[results/REPORT.md](results/REPORT.md), regenerated by `julia --project=. scripts/make_report.jl`.

All runs on spark-bc08, `W = 16, 32, 64`, `B = 1..4`, all twelve benchmarks (144 cases).

| Solver | file | proven optimal (gap 1e-4) | time |
|---|---|---:|---|
| MILP, HiGHS | `results/baseline_spark-bc08.md` | 142 of 144 | 120 s limit; hit on p93791 `W=64`, `B=3,4` (gaps 3.5%, 2.2%) |
| MILP, cuOpt | `results/baseline_spark-bc08_cuopt.md` | 142 of 144 | 120 s limit; hit on p93791 `W=32`, `B=4` and `W=64`, `B=4` |
| unsplittable flow | `results/ufp_spark-bc08.md` | heuristic | under 1 s per `(SOC, W)` for all `B` together, 16 threads |

HiGHS and cuOpt agree to within 0.1% on every case. Best `B <= 4`, test time in cycles:

| SOC | W | MILP | unsplittable flow | over MILP |
|---|---:|---:|---:|---:|
| d695 | 16 | 42268 | 47320 | 11.95% |
| d695 | 32 | 21437 | 23010 | 7.34% |
| d695 | 64 | 10985 | 11564 | 5.27% |
| p22810 | 16 | 434957 | 467506 | 7.48% |
| p22810 | 32 | 229612 | 256012 | 11.50% |
| p22810 | 64 | 140941 | 155611 | 10.41% |
| p34392 | 16 | 972352 | 1158654 | 19.16% |
| p34392 | 32 | 544579 | 584524 | 7.34% |
| p34392 | 64 | 544579 | 544579 | 0.00% |
| p93791 | 16 | 1763978 | 1852513 | 5.02% |
| p93791 | 32 | 886518 | 954083 | 7.62% |
| p93791 | 64 | 447144 | 485185 | 8.51% |

The unsplittable flow schedule equals the MILP one on u226, h953, f2126, q12710 and (to 0.01%)
t512505, and is within 3.2% on d281, 8.3% on g1023 and 1.5% on a586710. The `.csv` files have one
row per `(SOC, W, B)` with the TAM widths and the core assignment.

All combinatorial solvers against the MILP (`results/heuristics_spark-bc08.md` has every case),
36 `(SOC, W)` cases, best `B <= 4`, excess of the test time over the MILP:

| method | mean | median | worst | best or tied among the four | seconds, all cases |
|---|---:|---:|---:|---:|---:|
| unsplittable flow | 3.36% | 0.00% | 19.16% | 19 | 5.17 |
| greedy | 0.79% | 0.00% | 12.66% | 28 | 0.14 |
| best fit | 0.47% | 0.00% | 4.02% | 32 | 0.02 |
| shortest job first | 14.36% | 12.53% | 42.51% | 6 | 0.01 |

Best fit over all width partitions is within 4.1% of the MILP everywhere and takes milliseconds;
the unsplittable flow is the only one of the four with a proven factor (2, on equal widths) and
with a lower bound of its own.
