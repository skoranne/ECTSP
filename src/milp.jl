# Baseline solver: the ECTSP as a mixed-integer program in JuMP.
#
# Any MathOptInterface MILP optimizer can be passed (`HiGHS.Optimizer` by default,
# `cuOpt.Optimizer` for NVIDIA cuOpt). Times are divided by a lower bound before they enter the
# model, so the makespan variable is O(1); the reported makespan is recomputed in integers from the
# assignment the solver returns.

function _configure!(model, time_limit, mip_gap, threads, silent, attributes)
    silent && set_silent(model)
    time_limit === nothing || set_time_limit_sec(model, Float64(time_limit))
    mip_gap === nothing || set_attribute(model, MOI.RelativeGapTolerance(), Float64(mip_gap))
    threads === nothing || set_attribute(model, MOI.NumberOfThreads(), Int(threads))
    for (k, v) in attributes
        set_attribute(model, k, v)
    end
    return model
end

function _status(model)
    st = termination_status(model)
    has = primal_status(model) == MOI.FEASIBLE_POINT
    has || error("the solver returned no schedule (termination status $st)")
    return Symbol(st)
end

# The proven bound in cycles: the solver's dual bound, never below the combinatorial bound `lb`.
function _bound(model, scale, lb)
    b = try
        objective_bound(model)
    catch
        -Inf
    end
    isfinite(b) || return lb
    return max(lb, ceil(Int, b * scale - 1e-6 * max(1.0, abs(b * scale))))
end

"""
    solve_assignment(p, widths; optimizer=HiGHS.Optimizer, time_limit=60, mip_gap=nothing,
                     threads=nothing, silent=true, attributes=()) -> Schedule

The TAM widths are given; assign the cores to the TAMs for minimum test time (R || C_max):

    min C   s.t.  sum_j x[i,j] = 1,   sum_i T[i, w_j] x[i,j] <= C,   x binary.
"""
function solve_assignment(p::TestProblem, widths::AbstractVector{<:Integer}; optimizer=HiGHS.Optimizer,
                          time_limit=60, mip_gap=nothing, threads=nothing, silent::Bool=true, attributes=())
    widths = collect(Int, widths)
    isempty(widths) && throw(ArgumentError("at least one TAM is needed"))
    all(w -> 1 <= w <= max_width(p), widths) || throw(ArgumentError("TAM widths must lie in 1:$(max_width(p))"))
    n, B = ncores(p), length(widths)
    n == 0 && return build_schedule(p, widths, Int[]; method=:milp, status=:OPTIMAL)
    # bound for these widths: every core at least its best TAM; total work over B TAMs
    best = [minimum(p.T[i, w] for w in widths) for i in 1:n]
    lb = max(maximum(best), cld(sum(best), B))
    scale = Float64(lb)
    model = _configure!(Model(optimizer), time_limit, mip_gap, threads, silent, attributes)
    @variable(model, x[1:n, 1:B], Bin)
    @variable(model, C >= lb / scale)
    @constraint(model, [i in 1:n], sum(x[i, :]) == 1)
    @constraint(model, [j in 1:B], sum(p.T[i, widths[j]] / scale * x[i, j] for i in 1:n) <= C)
    @objective(model, Min, C)
    t0 = time()
    optimize!(model)
    secs = time() - t0
    status = _status(model)
    assignment = [argmax(value.(x[i, :])) for i in 1:n]
    return build_schedule(p, widths, assignment; method=:milp, bound=_bound(model, scale, lb),
                          status=status, solve_seconds=secs)
end

"""
    solve_baseline(p, W, B; optimizer=HiGHS.Optimizer, time_limit=60, mip_gap=nothing,
                   threads=nothing, silent=true, attributes=()) -> Schedule

Partition `W` TAM wires into exactly `B` TAMs and assign the cores, for minimum test time:

    u[j,w] = 1  iff TAM j has width w            (binary, one w per j, widths add up to W)
    x[i,j] = 1  iff core i is on TAM j           (binary, one j per i)
    a[i,j,w] = x[i,j] * u[j,w]                   (continuous; a <= u, sum_w a[i,j,w] = x[i,j])
    sum_{i,w} T[i,w] a[i,j,w] <= C  for every j;  min C.

TAMs are ordered by non-decreasing width to remove the permutation symmetry.
"""
function solve_baseline(p::TestProblem, W::Integer, B::Integer; optimizer=HiGHS.Optimizer, time_limit=60,
                        mip_gap=nothing, threads=nothing, silent::Bool=true, attributes=())
    1 <= W <= max_width(p) || throw(ArgumentError("W = $W outside 1:$(max_width(p))"))
    1 <= B <= W || throw(ArgumentError("the number of TAMs must lie in 1:W"))
    B == 1 && return solve_assignment(p, [W]; optimizer, time_limit, mip_gap, threads, silent, attributes)
    n = ncores(p)
    n == 0 && return build_schedule(p, vcat(fill(1, B - 1), W - B + 1), Int[]; method=:milp, status=:OPTIMAL)
    lb = lower_bound(p, W)
    scale = Float64(lb)
    Wmax = W - (B - 1)                       # every other TAM has at least one wire
    model = _configure!(Model(optimizer), time_limit, mip_gap, threads, silent, attributes)
    @variable(model, u[1:B, 1:Wmax], Bin)
    @variable(model, x[1:n, 1:B], Bin)
    @variable(model, 0 <= a[1:n, 1:B, 1:Wmax] <= 1)
    @variable(model, C >= lb / scale)
    @constraint(model, [j in 1:B], sum(u[j, :]) == 1)
    @constraint(model, sum(w * u[j, w] for j in 1:B, w in 1:Wmax) == W)
    @constraint(model, [j in 1:B-1], sum(w * u[j, w] for w in 1:Wmax) <= sum(w * u[j+1, w] for w in 1:Wmax))
    @constraint(model, [i in 1:n], sum(x[i, :]) == 1)
    @constraint(model, [i in 1:n, j in 1:B], sum(a[i, j, :]) == x[i, j])
    @constraint(model, [i in 1:n, j in 1:B, w in 1:Wmax], a[i, j, w] <= u[j, w])
    @constraint(model, [j in 1:B], sum(p.T[i, w] / scale * a[i, j, w] for i in 1:n, w in 1:Wmax) <= C)
    @objective(model, Min, C)
    t0 = time()
    optimize!(model)
    secs = time() - t0
    status = _status(model)
    widths = [argmax(value.(u[j, :])) for j in 1:B]
    sum(widths) == W || error("solver returned TAM widths $widths that do not add up to $W")
    assignment = [argmax(value.(x[i, :])) for i in 1:n]
    return build_schedule(p, widths, assignment; method=:milp, bound=_bound(model, scale, lb),
                          status=status, solve_seconds=secs)
end

"""
    solve_baseline(p, W; max_tams=min(W, ncores(p), 4), kwargs...) -> (best, all)

Run `solve_baseline(p, W, B)` for `B = 1:max_tams` and return the schedule with the smallest test
time together with all of them. `time_limit` applies to each `B`.
"""
function solve_baseline(p::TestProblem, W::Integer; max_tams::Integer=min(W, max(ncores(p), 1), 4), kwargs...)
    all = [solve_baseline(p, W, B; kwargs...) for B in 1:max_tams]
    return all[argmin([s.makespan for s in all])], all
end
