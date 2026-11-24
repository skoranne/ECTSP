# ECTSP through single source unsplittable flow.
#
# For given TAM widths w_1..w_B the network is
#
#     source --(capacity of TAM j)--> TAM j --(admissible)--> core i   [terminal, demand d_i]
#
# Two demand models:
#
#   :bits   the formulation of Koranne and Choudhary (DATE 2002): the commodity is test data. A TAM of
#           width w_j carries w_j bits per cycle, so its capacity for a trial test time C is C * w_j;
#           core i on TAM j uses w_j * T[i, w_j] wire-cycles and its demand is the least of these,
#           d_i = min_j w_j * T[i, w_j].
#   :time   the commodity is test time: every TAM has capacity C and d_i = min_j T[i, w_j], the time
#           of the core on its fastest TAM.
#
# In both, core i is admissible on TAM j when it fits, T[i, w_j] <= C, and when it costs there at
# most rho * d_i. The smallest C for which a fractional flow of all demands exists is found by
# bisection over a maximum flow; that flow is then made unsplittable (Dinitz, Garg, Goemans), after
# which every TAM carries at most its capacity plus the largest demand.
#
# With rho = Inf the fractional C is a lower bound on the optimum for these widths in either model
# (a demand never exceeds what the core really uses). With TAMs of one width the two models
# coincide, demands are exact and the schedule is within a factor 2 of the optimum. With unequal
# widths a core may cost more than d_i on the TAM it gets, so several values of rho are tried and
# the schedule with the smallest true test time is kept.

const UFP_SLOWDOWNS = (1.0, 1.05, 1.1, 1.2, 1.35, 1.5, 2.0, 3.0, Inf)
const UFP_MODELS = (:bits, :time)

# capacity units per cycle of every TAM, and the demands, for a model
function _ufp_model(Tw::Matrix{Int}, widths::Vector{Int}, model::Symbol)
    model in UFP_MODELS || throw(ArgumentError("model must be one of $UFP_MODELS or :both"))
    unit = model == :bits ? copy(widths) : ones(Int, length(widths))
    d = [minimum(unit[j] * Tw[i, j] for j in eachindex(unit)) for i in 1:size(Tw, 1)]
    return d, unit
end

# Maximum-flow test: can all demands be routed with at most C * unit[j] on TAM j? Returns the flow matrix or nothing.
function _fractional_flow(d::Vector{Int}, Tw::Matrix{Int}, unit::Vector{Int}, C::Int, rho::Float64)
    n, B = size(Tw)
    g = FlowNet(n + B + 2)
    s, t = 1, n + B + 2
    for j in 1:B
        add_arc!(g, s, 1 + j, C * unit[j])
    end
    ids = zeros(Int, n, B)
    for i in 1:n
        add_arc!(g, 1 + B + i, t, d[i])
        for j in 1:B
            if Tw[i, j] <= C && unit[j] * Tw[i, j] <= rho * d[i]
                ids[i, j] = add_arc!(g, 1 + j, 1 + B + i, d[i])
            end
        end
    end
    maxflow!(g, s, t) == sum(d) || return nothing
    return [ids[i, j] == 0 ? 0 : arc_flow(g, ids[i, j]) for i in 1:n, j in 1:B]
end

# Smallest C with a fractional flow, and that flow.
function _min_fractional(d::Vector{Int}, Tw::Matrix{Int}, unit::Vector{Int}, rho::Float64)
    n, B = size(Tw)
    lo = max(maximum(minimum(Tw; dims=2)), cld(sum(d), sum(unit)))
    # every core on the TAM where it costs least: feasible for every rho
    hi = max(lo, sum(Tw[i, argmin([unit[j] * Tw[i, j] for j in 1:B])] for i in 1:n))
    F = _fractional_flow(d, Tw, unit, lo, rho)
    F === nothing || return lo, F
    best = nothing
    while hi - lo > 1                            # lo infeasible, hi feasible
        mid = lo + (hi - lo) ÷ 2
        Fm = _fractional_flow(d, Tw, unit, mid, rho)
        if Fm === nothing
            lo = mid
        else
            hi, best = mid, Fm
        end
    end
    best === nothing && (best = _fractional_flow(d, Tw, unit, hi, rho))
    return hi, best
end

# One rounding: (assignment, C, largest excess of a TAM over its fractional load in units of max demand).
function _ufp_round(d::Vector{Int}, Tw::Matrix{Int}, unit::Vector{Int}, rho::Float64)
    n, B = size(Tw)
    C, F = _min_fractional(d, Tw, unit, rho)
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
_ufp_round(d::Vector{Int}, Tw::Matrix{Int}, rho::Float64) = _ufp_round(d, Tw, ones(Int, size(Tw, 2)), rho)

# Descent on the test time: move a core off the longest TAM, or exchange it with a core of another
# TAM, while that shortens the longest TAM. Not part of the flow formulation (`improve=true`).
function _improve_assignment(Tw::Matrix{Int}, assignment::Vector{Int})
    a = copy(assignment)
    n, B = size(Tw)
    load = [sum(Tw[i, j] for i in 1:n if a[i] == j; init=0) for j in 1:B]
    while true
        c = argmax(load)
        best, bi, bj, bk = load[c], 0, 0, 0
        for i in 1:n, j in 1:B
            (a[i] == c && j != c) || continue
            m = max(load[c] - Tw[i, c], load[j] + Tw[i, j])
            m < best && ((best, bi, bj, bk) = (m, i, j, 0))
            for k in 1:n
                a[k] == j || continue
                m = max(load[c] - Tw[i, c] + Tw[k, c], load[j] + Tw[i, j] - Tw[k, j])
                m < best && ((best, bi, bj, bk) = (m, i, j, k))
            end
        end
        bi == 0 && break
        load[c] -= Tw[bi, c]; load[bj] += Tw[bi, bj]; a[bi] = bj
        if bk > 0
            load[bj] -= Tw[bk, bj]; load[c] += Tw[bk, c]; a[bk] = c
        end
    end
    return a, maximum(load)
end

"""
    ufp_fractional_bound(p, widths; model=:both) -> Int

Lower bound on the test time for the given TAM widths: the smallest test time of a fractional
(splittable) routing, in which a core counts with its least wire-cycles (`:bits`) or with its time
on its fastest TAM (`:time`); `:both` is the larger of the two.
"""
function ufp_fractional_bound(p::TestProblem, widths::AbstractVector{<:Integer}; model::Symbol=:both)
    w = collect(Int, widths)
    Tw = p.T[:, w]
    size(Tw, 1) == 0 && return 0
    return maximum((model == :both ? UFP_MODELS : (model,))) do m
        d, unit = _ufp_model(Tw, w, m)
        first(_min_fractional(d, Tw, unit, Inf))
    end
end

"""
    solve_ufp(p, widths; model=:both, slowdowns=UFP_SLOWDOWNS, improve=false) -> Schedule

Assign the cores to TAMs of the given widths by single source unsplittable flow
(Dinitz-Garg-Goemans rounding of a maximum flow). `model` is `:bits` (test data as commodity, TAM
capacity proportional to its width: the formulation of Koranne and Choudhary), `:time` (test time
as commodity) or `:both` (the better schedule of the two). `slowdowns` are the admissibility
factors rho that are tried; `(Inf,)` is the plain formulation, `(1.0,)` keeps every core on the
TAMs where it costs least. `improve=true` adds a descent after the rounding (single moves and
exchanges that shorten the longest TAM); it is not part of the flow method. `Schedule.bound` is `ufp_fractional_bound(p, widths; model)`.
"""
function solve_ufp(p::TestProblem, widths::AbstractVector{<:Integer}; model::Symbol=:both, slowdowns=UFP_SLOWDOWNS,
                   improve::Bool=false)
    t0 = time()
    s = _solve_ufp(p, collect(Int, widths), Float64.(collect(slowdowns)), model, improve)
    return build_schedule(p, s.widths, s.assignment; method=:ufp, bound=s.bound, status=:HEURISTIC,
                          solve_seconds=time() - t0)
end

function _solve_ufp(p::TestProblem, widths::Vector{Int}, slowdowns::Vector{Float64}, model::Symbol=:both, improve::Bool=false)
    isempty(widths) && throw(ArgumentError("at least one TAM is needed"))
    all(w -> 1 <= w <= max_width(p), widths) || throw(ArgumentError("TAM widths must lie in 1:$(max_width(p))"))
    isempty(slowdowns) && throw(ArgumentError("at least one slowdown factor is needed"))
    models = model == :both ? UFP_MODELS : (model,)
    all(in(UFP_MODELS), models) || throw(ArgumentError("model must be one of $UFP_MODELS or :both"))
    n = ncores(p)
    n == 0 && return (widths=widths, assignment=Int[], makespan=0, bound=0)
    Tw = p.T[:, widths]
    best_assignment, best_makespan = Int[], typemax(Int)
    bound = 0
    for m in (allequal(widths) ? models[1:1] : models)      # one width: the models coincide
        d, unit = _ufp_model(Tw, widths, m)
        plain = false
        for rho in slowdowns
            assignment, C, _, _ = _ufp_round(d, Tw, unit, rho)
            isinf(rho) && (bound = max(bound, C); plain = true)
            makespan = maximum(sum(Tw[i, j] for i in 1:n if assignment[i] == j; init=0) for j in eachindex(widths))
            if makespan < best_makespan
                best_assignment, best_makespan = assignment, makespan
            end
        end
        plain || (bound = max(bound, first(_min_fractional(d, Tw, unit, Inf))))
    end
    improve && ((best_assignment, best_makespan) = _improve_assignment(Tw, best_assignment))
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
    solve_ufp(p, W, B; model=:both, slowdowns=UFP_SLOWDOWNS, improve=false) -> Schedule

Try every partition of `W` wires into exactly `B` TAMs (in parallel over Julia threads), solve
each by unsplittable flow and keep the best. `Schedule.bound` is the larger of `lower_bound(p, W)`
and the smallest fractional bound over the partitions, a lower bound for `B` TAMs.
"""
function solve_ufp(p::TestProblem, W::Integer, B::Integer; model::Symbol=:both, slowdowns=UFP_SLOWDOWNS, improve::Bool=false)
    1 <= W <= max_width(p) || throw(ArgumentError("W = $W outside 1:$(max_width(p))"))
    1 <= B <= W || throw(ArgumentError("the number of TAMs must lie in 1:W"))
    t0 = time()
    parts = width_partitions(W, B)
    rhos = Float64.(collect(slowdowns))
    res = Vector{Any}(undef, length(parts))
    Threads.@threads for k in eachindex(parts)
        res[k] = _solve_ufp(p, parts[k], rhos, model, improve)
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
