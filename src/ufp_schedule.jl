# ECTSP through single source unsplittable flow.
#
# For given TAM widths w_1..w_B the network is
#
#     source --(load of TAM j)--> TAM j --(admissible)--> core i   [terminal, demand d_i]
#
# with d_i the test time of core i on its fastest TAM. Core i is admissible on TAM j when its time
# there, T[i, w_j], is at most rho * d_i and at most the trial makespan C. The smallest C for which
# a fractional flow of all demands with at most C per TAM exists is found by bisection over a
# maximum flow; that flow is then made unsplittable. In demand units every TAM carries at most
# C + max_i d_i, so the test time is at most rho * (C + max_i d_i).
#
# With rho = Inf the fractional C is a lower bound on the optimum for these widths (demands only
# under-estimate the times), and with TAMs of one width (or rho = 1) demands are exact and the
# schedule is within a factor 2 of the optimum. Narrow TAMs make a core slower than d_i, so
# several values of rho are tried and the schedule with the smallest true test time is kept.

const UFP_SLOWDOWNS = (1.0, 1.05, 1.1, 1.2, 1.35, 1.5, 2.0, 3.0, Inf)

# Maximum-flow test: can all demands be routed with at most C per TAM? Returns the flow matrix or nothing.
function _fractional_flow(d::Vector{Int}, Tw::Matrix{Int}, C::Int, rho::Float64)
    n, B = size(Tw)
    g = FlowNet(n + B + 2)
    s, t = 1, n + B + 2
    for j in 1:B
        add_arc!(g, s, 1 + j, C)
    end
    ids = zeros(Int, n, B)
    for i in 1:n
        add_arc!(g, 1 + B + i, t, d[i])
        for j in 1:B
            if Tw[i, j] <= C && Tw[i, j] <= rho * d[i]
                ids[i, j] = add_arc!(g, 1 + j, 1 + B + i, d[i])
            end
        end
    end
    maxflow!(g, s, t) == sum(d) || return nothing
    return [ids[i, j] == 0 ? 0 : arc_flow(g, ids[i, j]) for i in 1:n, j in 1:B]
end

# Smallest C with a fractional flow, and that flow.
function _min_fractional(d::Vector{Int}, Tw::Matrix{Int}, rho::Float64)
    B = size(Tw, 2)
    lo = max(maximum(d), cld(sum(d), B))
    hi = sum(d)                                  # every core on its fastest TAM
    F = _fractional_flow(d, Tw, lo, rho)
    F === nothing || return lo, F
    best = nothing
    while hi - lo > 1                            # lo infeasible, hi feasible
        mid = lo + (hi - lo) ÷ 2
        Fm = _fractional_flow(d, Tw, mid, rho)
        if Fm === nothing
            lo = mid
        else
            hi, best = mid, Fm
        end
    end
    best === nothing && (best = _fractional_flow(d, Tw, hi, rho))
    return hi, best
end

# One rounding: (assignment, C, largest excess of a TAM over its fractional load in units of max demand).
function _ufp_round(d::Vector{Int}, Tw::Matrix{Int}, rho::Float64)
    n, B = size(Tw)
    C, F = _min_fractional(d, Tw, rho)
    tails, heads, flow = Int[], Int[], Int[]
    for j in 1:B                                  # arcs 1..B: source -> TAM j
        push!(tails, 1); push!(heads, 1 + j); push!(flow, sum(@view F[:, j]))
    end
    for i in 1:n, j in 1:B
        F[i, j] > 0 || continue
        push!(tails, 1 + j); push!(heads, 1 + B + i); push!(flow, F[i, j])
    end
    paths, load, exceptions = unsplittable_flow(1 + B + n, tails, heads, flow, 1, collect(2+B:1+B+n), d)
    assignment = [paths[i][1] for i in 1:n]
    excess = maximum((load[j] - flow[j]) / maximum(d) for j in 1:B)
    return assignment, C, excess, exceptions
end

"""
    ufp_fractional_bound(p, widths) -> Int

Lower bound on the test time for the given TAM widths: the smallest makespan of a fractional
(splittable) routing in which every core counts with its time on its fastest TAM.
"""
function ufp_fractional_bound(p::TestProblem, widths::AbstractVector{<:Integer})
    Tw = p.T[:, collect(Int, widths)]
    d = vec(minimum(Tw; dims=2))
    return first(_min_fractional(d, Tw, Inf))
end

"""
    solve_ufp(p, widths; slowdowns=UFP_SLOWDOWNS) -> Schedule

Assign the cores to TAMs of the given widths by single source unsplittable flow
(Dinitz-Garg-Goemans rounding of a maximum flow). `slowdowns` are the admissibility factors rho
that are tried; `(Inf,)` is the plain formulation, `(1.0,)` keeps every core on its fastest TAMs.
`Schedule.bound` is `ufp_fractional_bound(p, widths)`.
"""
function solve_ufp(p::TestProblem, widths::AbstractVector{<:Integer}; slowdowns=UFP_SLOWDOWNS)
    t0 = time()
    s = _solve_ufp(p, collect(Int, widths), Float64.(collect(slowdowns)))
    return build_schedule(p, s.widths, s.assignment; method=:ufp, bound=s.bound, status=:HEURISTIC,
                          solve_seconds=time() - t0)
end

function _solve_ufp(p::TestProblem, widths::Vector{Int}, slowdowns::Vector{Float64})
    isempty(widths) && throw(ArgumentError("at least one TAM is needed"))
    all(w -> 1 <= w <= max_width(p), widths) || throw(ArgumentError("TAM widths must lie in 1:$(max_width(p))"))
    isempty(slowdowns) && throw(ArgumentError("at least one slowdown factor is needed"))
    n = ncores(p)
    n == 0 && return (widths=widths, assignment=Int[], makespan=0, bound=0)
    Tw = p.T[:, widths]
    d = vec(minimum(Tw; dims=2))
    best_assignment, best_makespan = Int[], typemax(Int)
    bound = 0
    for rho in slowdowns
        assignment, C, _, _ = _ufp_round(d, Tw, rho)
        isinf(rho) && (bound = C)
        makespan = maximum(sum(Tw[i, j] for i in 1:n if assignment[i] == j; init=0) for j in eachindex(widths))
        if makespan < best_makespan
            best_assignment, best_makespan = assignment, makespan
        end
    end
    bound == 0 && (bound = first(_min_fractional(d, Tw, Inf)))
    return (widths=widths, assignment=best_assignment, makespan=best_makespan, bound=bound)
end

"All ways to write `W` as a sum of exactly `B` positive widths in non-decreasing order."
function width_partitions(W::Integer, B::Integer)
    out = Vector{Int}[]
    cur = zeros(Int, B)
    function rec(k, left, least)
        if k == B
            left >= least && (cur[k] = left; push!(out, copy(cur)))
            return
        end
        for w in least:left ÷ (B - k + 1)
            cur[k] = w
            rec(k + 1, left - w, w)
        end
    end
    1 <= B <= W && rec(1, Int(W), 1)
    return out
end

"""
    solve_ufp(p, W, B; slowdowns=UFP_SLOWDOWNS) -> Schedule

Try every partition of `W` wires into exactly `B` TAMs (in parallel over Julia threads), solve
each by unsplittable flow and keep the best. `Schedule.bound` is the larger of `lower_bound(p, W)`
and the smallest fractional bound over the partitions, a lower bound for `B` TAMs.
"""
function solve_ufp(p::TestProblem, W::Integer, B::Integer; slowdowns=UFP_SLOWDOWNS)
    1 <= W <= max_width(p) || throw(ArgumentError("W = $W outside 1:$(max_width(p))"))
    1 <= B <= W || throw(ArgumentError("the number of TAMs must lie in 1:W"))
    t0 = time()
    parts = width_partitions(W, B)
    rhos = Float64.(collect(slowdowns))
    res = Vector{Any}(undef, length(parts))
    Threads.@threads for k in eachindex(parts)
        res[k] = _solve_ufp(p, parts[k], rhos)
    end
    best = res[argmin([r.makespan for r in res])]
    bound = max(lower_bound(p, W), minimum(r.bound for r in res))
    return build_schedule(p, best.widths, best.assignment; method=:ufp, bound=bound, status=:HEURISTIC,
                          solve_seconds=time() - t0)
end

"`solve_ufp(p, W; max_tams=4)`: best over `B = 1:max_tams`, and all of them."
function solve_ufp(p::TestProblem, W::Integer; max_tams::Integer=min(W, max(ncores(p), 1), 4), kwargs...)
    all = [solve_ufp(p, W, B; kwargs...) for B in 1:max_tams]
    return all[argmin([s.makespan for s in all])], all
end
