# List-scheduling heuristics for the ECTSP.
#
# For given TAM widths, with Tw[i, j] the time of core i on TAM j and d_i = min_j Tw[i, j]:
#
#   greedy     cores by decreasing d_i; each goes to the TAM on which it finishes earliest
#   sjf        shortest job first: the same rule with the cores by increasing d_i
#   best fit   for a target test time C, cores by decreasing d_i; each goes to the TAM that it
#              fills best (least capacity C left after it); the smallest C for which every core
#              fits is found by bisection (best fit decreasing inside MULTIFIT)
#
# `solve_heuristic(p, W, B; method)` applies one of them to every partition of W into B widths.

const HEURISTICS = (:greedy, :best_fit, :sjf)

# earliest-completion list scheduling in the given core order
function _list_schedule(Tw::Matrix{Int}, order::Vector{Int})
    n, B = size(Tw)
    load = zeros(Int, B)
    assignment = zeros(Int, n)
    for i in order
        best = 1
        for j in 2:B
            fj, fb = load[j] + Tw[i, j], load[best] + Tw[i, best]
            (fj < fb || (fj == fb && Tw[i, j] < Tw[i, best])) && (best = j)
        end
        assignment[i] = best
        load[best] += Tw[i, best]
    end
    return assignment, maximum(load)
end

# best fit decreasing into TAMs of capacity C; nothing when a core does not fit
function _best_fit(Tw::Matrix{Int}, order::Vector{Int}, C::Int)
    n, B = size(Tw)
    load = zeros(Int, B)
    assignment = zeros(Int, n)
    for i in order
        best = 0
        for j in 1:B
            fj = load[j] + Tw[i, j]
            fj <= C || continue
            if best == 0 || fj > load[best] + Tw[i, best] || (fj == load[best] + Tw[i, best] && Tw[i, j] < Tw[i, best])
                best = j
            end
        end
        best == 0 && return nothing
        assignment[i] = best
        load[best] += Tw[i, best]
    end
    return assignment, maximum(load)
end

function _best_fit_search(Tw::Matrix{Int}, order::Vector{Int}, d::Vector{Int})
    B = size(Tw, 2)
    lo = max(maximum(d), cld(sum(d), B)) - 1          # treated as not fitting
    hi = lo + 1
    res = _best_fit(Tw, order, hi)
    while res === nothing                             # everything fits once C covers all the work
        hi = 2 * hi
        res = _best_fit(Tw, order, hi)
    end
    best = res
    hi = min(hi, best[2])
    while hi - lo > 1
        mid = lo + (hi - lo) ÷ 2
        r = _best_fit(Tw, order, mid)
        if r === nothing
            lo = mid
        else
            r[2] < best[2] && (best = r)
            hi = min(mid, r[2])
        end
    end
    return best
end

function _heuristic(p::TestProblem, widths::Vector{Int}, method::Symbol)
    method in HEURISTICS || throw(ArgumentError("method must be one of $HEURISTICS"))
    isempty(widths) && throw(ArgumentError("at least one TAM is needed"))
    all(w -> 1 <= w <= max_width(p), widths) || throw(ArgumentError("TAM widths must lie in 1:$(max_width(p))"))
    ncores(p) == 0 && return (widths=widths, assignment=Int[], makespan=0)
    Tw = p.T[:, widths]
    d = vec(minimum(Tw; dims=2))
    order = sortperm(d; rev = method != :sjf, alg=MergeSort)      # stable: ties keep the module order
    assignment, makespan = method == :best_fit ? _best_fit_search(Tw, order, d) : _list_schedule(Tw, order)
    return (widths=widths, assignment=assignment, makespan=makespan)
end

"""
    solve_heuristic(p, widths; method=:greedy) -> Schedule

Assign the cores to TAMs of the given widths with `:greedy`, `:best_fit` or `:sjf`.
`Schedule.bound` is `ufp_fractional_bound(p, widths)`.
"""
function solve_heuristic(p::TestProblem, widths::AbstractVector{<:Integer}; method::Symbol=:greedy)
    t0 = time()
    w = collect(Int, widths)
    r = _heuristic(p, w, method)
    bound = ncores(p) == 0 ? 0 : ufp_fractional_bound(p, w)
    return build_schedule(p, r.widths, r.assignment; method=method, bound=bound, status=:HEURISTIC,
                          solve_seconds=time() - t0)
end

"""
    solve_heuristic(p, W, B; method=:greedy) -> Schedule

Best schedule of the heuristic over every partition of `W` wires into exactly `B` TAMs
(in parallel over Julia threads). `Schedule.bound` is `lower_bound(p, W)`.
"""
function solve_heuristic(p::TestProblem, W::Integer, B::Integer; method::Symbol=:greedy)
    1 <= W <= max_width(p) || throw(ArgumentError("W = $W outside 1:$(max_width(p))"))
    1 <= B <= W || throw(ArgumentError("the number of TAMs must lie in 1:W"))
    t0 = time()
    parts = width_partitions(W, B)
    res = Vector{Any}(undef, length(parts))
    Threads.@threads for k in eachindex(parts)
        res[k] = _heuristic(p, parts[k], method)
    end
    best = res[argmin([r.makespan for r in res])]
    return build_schedule(p, best.widths, best.assignment; method=method, bound=lower_bound(p, W),
                          status=:HEURISTIC, solve_seconds=time() - t0)
end

"`solve_heuristic(p, W; max_tams=4, method=:greedy)`: best over `B = 1:max_tams`, and all of them."
function solve_heuristic(p::TestProblem, W::Integer; max_tams::Integer=min(W, max(ncores(p), 1), 4), method::Symbol=:greedy)
    all = [solve_heuristic(p, W, B; method=method) for B in 1:max_tams]
    return all[argmin([s.makespan for s in all])], all
end

"Greedy (longest test first, earliest finish): see `solve_heuristic`."
solve_greedy(p::TestProblem, args...; kwargs...) = solve_heuristic(p, args...; method=:greedy, kwargs...)
"Best fit decreasing with bisection on the test time: see `solve_heuristic`."
solve_best_fit(p::TestProblem, args...; kwargs...) = solve_heuristic(p, args...; method=:best_fit, kwargs...)
"Shortest job first, earliest finish: see `solve_heuristic`."
solve_sjf(p::TestProblem, args...; kwargs...) = solve_heuristic(p, args...; method=:sjf, kwargs...)
