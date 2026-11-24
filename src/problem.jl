# The embedded core test scheduling problem (ECTSP).
#
# An SOC has n cores and W TAM wires. The wires are partitioned into B test buses (TAMs) of widths
# w_1 + ... + w_B = W. Every core is assigned to exactly one TAM; the cores of one TAM are tested
# one after the other, different TAMs run in parallel. Core i on a TAM of width w takes T[i, w]
# cycles. Minimise the SOC test time (the makespan), max_j sum_{i on j} T[i, w_j].
#
# As a network: source -> core i (demand: its test) -> TAM j -> sink, where a core's demand may not
# be split over TAMs (single source unsplittable flow). With the widths fixed this is scheduling on
# unrelated parallel machines, R || C_max.

"""
    TestProblem

`T[i, w]` is the test time of core `i` on a TAM of width `w`, non-increasing in `w`, for
`w = 1:size(T, 2)`. `ids[i]` is the module number in the benchmark. `bist[i]` is the time of the
tests of that module that do not use the TAM.
"""
struct TestProblem
    name::String
    ids::Vector{Int}
    T::Matrix{Int}
    bist::Vector{Int}
end

ncores(p::TestProblem) = length(p.ids)
max_width(p::TestProblem) = size(p.T, 2)

"""
    TestProblem(soc; max_width=64, include_top=false)

Test time table of the cores of `soc` that have at least one TAM-applied test. The hierarchy is
flattened (every module is a core with its own wrapper); the level-0 module is left out unless
`include_top` is set, as in the published results on these benchmarks.
"""
function TestProblem(soc::SOC; max_width::Integer=64, include_top::Bool=false)
    max_width >= 1 || throw(ArgumentError("max_width must be at least 1"))
    cores = [c for c in soc.cores if (include_top || c.level > 0) && any(t -> t.tam_use, c.tests)]
    T = Matrix{Int}(undef, length(cores), max_width)
    for (i, c) in enumerate(cores), w in 1:max_width
        T[i, w] = test_time(c, w)
    end
    return TestProblem(soc.name, [c.id for c in cores], T, [bist_time(c) for c in cores])
end

"""
    lower_bound(p, W) -> Int

Architecture-independent lower bound on the test time with `W` TAM wires: no core can be faster
than on all `W` wires, and core `i` occupies at least `min_w w * T[i, w]` wire-cycles of the
`W * makespan` available.
"""
function lower_bound(p::TestProblem, W::Integer)
    1 <= W <= max_width(p) || throw(ArgumentError("W = $W outside 1:$(max_width(p))"))
    ncores(p) == 0 && return 0
    longest = maximum(@view p.T[:, W])
    area = sum(minimum(w * p.T[i, w] for w in 1:W) for i in 1:ncores(p))
    return max(longest, cld(area, W))
end

"""
    Schedule

`widths[j]` is the width of TAM `j`, `assignment[i]` the TAM of core `i`. `slots[j]` lists
`(core, start, finish)` for TAM `j` in test order. `bound` is the best proven lower bound for the
problem that was solved (same `W`, same number of TAMs or same widths); `status` is the solver's
termination status.
"""
struct Schedule
    problem::String
    method::Symbol
    total_width::Int
    widths::Vector{Int}
    assignment::Vector{Int}
    slots::Vector{Vector{NTuple{3,Int}}}
    makespan::Int
    bound::Int
    status::Symbol
    solve_seconds::Float64
end

"Relative optimality gap of a schedule, 0.0 when proven optimal."
gap(s::Schedule) = s.makespan == 0 ? 0.0 : max(0.0, (s.makespan - s.bound) / s.makespan)

"""
    build_schedule(p, widths, assignment; ...) -> Schedule

Lay out the cores of every TAM back to back (longest test first) and compute the makespan.
"""
function build_schedule(p::TestProblem, widths::Vector{Int}, assignment::Vector{Int};
                        method::Symbol=:manual, bound::Integer=0, status::Symbol=:FEASIBLE,
                        solve_seconds::Real=0.0)
    slots = [NTuple{3,Int}[] for _ in widths]
    for j in eachindex(widths)
        on = [i for i in 1:ncores(p) if assignment[i] == j]
        sort!(on; by = i -> -p.T[i, widths[j]])
        t = 0
        for i in on
            push!(slots[j], (i, t, t + p.T[i, widths[j]]))
            t += p.T[i, widths[j]]
        end
    end
    makespan = maximum((isempty(s) ? 0 : s[end][3] for s in slots); init=0)
    s = Schedule(p.name, method, sum(widths), widths, assignment, slots, makespan,
                 min(Int(bound), makespan), status, Float64(solve_seconds))
    validate(s, p)
    return s
end

"""
    validate(schedule, p) -> true

Check a schedule against the problem: widths positive, every core on exactly one TAM for exactly
its test time at that width, no two tests overlapping on a TAM, makespan as stated. Throws on a violation.
"""
function validate(s::Schedule, p::TestProblem)
    all(>=(1), s.widths) || error("TAM of width < 1")
    sum(s.widths) == s.total_width || error("TAM widths do not add up to $(s.total_width)")
    s.total_width <= max_width(p) || error("total width beyond the test time table")
    length(s.assignment) == ncores(p) || error("assignment length differs from the number of cores")
    seen = zeros(Int, ncores(p))
    finish = 0
    for (j, tam) in enumerate(s.slots)
        t = 0
        for (i, a, b) in tam
            s.assignment[i] == j || error("core $(p.ids[i]) is in the slots of TAM $j but assigned to TAM $(s.assignment[i])")
            a >= t || error("tests overlap on TAM $j")
            b - a == p.T[i, s.widths[j]] || error("core $(p.ids[i]) has the wrong duration on TAM $j")
            seen[i] += 1
            t = b
        end
        finish = max(finish, t)
    end
    all(==(1), seen) || error("cores not scheduled exactly once: $(p.ids[findall(!=(1), seen)])")
    finish == s.makespan || error("makespan is $finish, schedule says $(s.makespan)")
    s.bound <= s.makespan || error("lower bound above the makespan")
    return true
end

function Base.show(io::IO, ::MIME"text/plain", s::Schedule)
    @printf(io, "%s  W=%d  TAMs=%d  test time %d  bound %d  gap %.2f%%  [%s, %s, %.2fs]\n",
            s.problem, s.total_width, length(s.widths), s.makespan, s.bound, 100 * gap(s),
            s.method, s.status, s.solve_seconds)
    for (j, tam) in enumerate(s.slots)
        load = isempty(tam) ? 0 : tam[end][3]
        @printf(io, "  TAM %d  width %2d  load %d\n", j, s.widths[j], load)
    end
end

"Print the schedule with the start and finish time of every core; `p` supplies the module numbers."
function print_schedule(io::IO, s::Schedule, p::TestProblem)
    show(io, MIME"text/plain"(), s)
    for (j, tam) in enumerate(s.slots), (i, a, b) in tam
        @printf(io, "    TAM %d  module %3d  %12d .. %-12d\n", j, p.ids[i], a, b)
    end
end
print_schedule(s::Schedule, p::TestProblem) = print_schedule(stdout, s, p)
