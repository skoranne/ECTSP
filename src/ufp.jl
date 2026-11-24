# Single source unsplittable flow: the algorithm of Dinitz, Garg and Goemans (1999).
#
# Input: a directed graph, a source, terminals with demands, and a (fractional) flow that carries
# every demand from the source to its terminal. Output: one source-terminal path per terminal such
# that the demand routed over any arc exceeds the given flow on that arc by at most the largest
# demand. All quantities are integers.
#
# The algorithm works backwards from the terminals. A terminal with demand d standing at vertex v
# is moved over an incoming arc (u, v) whenever that arc carries at least d; the arc loses d units
# and joins the terminal's path. When no terminal can move, flow is shifted around an alternating
# cycle: forward paths (followed along the arcs, flow decreases) alternate with backward paths
# (followed against singular arcs, flow increases) until an arc empties or a backward arc carries
# exactly the demand of a terminal at its head. An arc (u, v) is singular when v and everything
# reachable from v has out-degree at most one.

"""
    unsplittable_flow(nv, tails, heads, flow, source, terminals, demands) -> (paths, load, exceptions)

`flow[e] >= 0` on arc `tails[e] -> heads[e]` must be acyclic and satisfy conservation: at every
vertex other than `source`, inflow - outflow equals the demand of the terminals at that vertex.
`paths[k]` lists the arcs from `source` to `terminals[k]`; `load[e]` is the demand routed over arc
`e`. When `exceptions == 0` (no terminal had to cross a raised arc with flow left over; this has
not been observed), `load[e] <= flow[e] + maximum(demands)` holds for every arc by construction.
"""
function unsplittable_flow(nv::Integer, tails::AbstractVector{<:Integer}, heads::AbstractVector{<:Integer},
                           flow::AbstractVector{<:Integer}, source::Integer,
                           terminals::AbstractVector{<:Integer}, demands::AbstractVector{<:Integer})
    m, K = length(tails), length(terminals)
    length(heads) == m && length(flow) == m || throw(ArgumentError("tails, heads and flow differ in length"))
    length(demands) == K || throw(ArgumentError("terminals and demands differ in length"))
    all(>(0), demands) || throw(ArgumentError("demands must be positive"))
    all(>=(0), flow) || throw(ArgumentError("flow must be non-negative"))
    f = collect(Int, flow)
    out = [Int[] for _ in 1:nv]
    inn = [Int[] for _ in 1:nv]
    for e in 1:m
        push!(out[tails[e]], e)
        push!(inn[heads[e]], e)
    end
    _check_flow(nv, tails, heads, f, out, inn, source, terminals, demands)

    pos = collect(Int, terminals)
    paths = [Int[] for _ in 1:K]
    load = zeros(Int, m)

    # `raised[e]`: the flow on e has been increased by an augmentation. Such an arc is crossed only by
    # a terminal whose demand equals its flow (the arc then disappears), so that the demand routed
    # over an arc is at most its initial flow plus one demand. The exception is a vertex without
    # outgoing flow and a single incoming arc: that arc carries exactly the terminals standing there.
    raised = falses(m)
    nfallback = Ref(0)
    function move_terminals!()
        for k in 1:K
            v = pos[k]
            while v != source
                pick = 0
                nin = 0
                for e in inn[v]
                    f[e] > 0 || continue
                    nin += 1
                    if f[e] == demands[k]
                        pick = e
                        break
                    elseif f[e] > demands[k] && !raised[e]
                        pick = e
                    end
                end
                if pick == 0 && nin == 1 && !any(e -> f[e] > 0, out[v])
                    pick = inn[v][findfirst(e -> f[e] > 0, inn[v])]
                    nfallback[] += 1
                end
                pick == 0 && break
                f[pick] -= demands[k]
                load[pick] += demands[k]
                push!(paths[k], pick)
                v = tails[pick]
            end
            pos[k] = v
        end
    end

    function alive_out(v, except)
        for e in out[v]
            f[e] > 0 && e != except && return e
        end
        return 0
    end
    function alive_in(v, except)
        for e in inn[v]
            f[e] > 0 && e != except && return e
        end
        return 0
    end
    outdeg(v) = count(e -> f[e] > 0, out[v])

    move_terminals!()
    seen = zeros(Int, nv)                   # position of a vertex in the current walk
    walk_arc = Int[]
    walk_fwd = Bool[]
    budget = m + K * nv + 8                 # every round empties an arc or moves a terminal
    while any(!=(source), pos)
        (budget -= 1) >= 0 || error("unsplittable_flow: no progress (internal error)")
        # walk from the source until a vertex repeats; the closed part is the alternating cycle
        fill!(seen, 0); empty!(walk_arc); empty!(walk_fwd)
        v = source
        seen[v] = 1
        forward = true
        last = 0
        first_arc = 0
        while true
            e = 0
            if forward
                e = alive_out(v, last)
                if e == 0                   # a terminal vertex: turn around over another incoming arc
                    forward = false
                    e = alive_in(v, last)
                end
            else
                # arrived backwards over `last`; other outgoing arcs end the backward path
                if outdeg(v) >= 2
                    forward = true
                    e = alive_out(v, last)
                else
                    e = alive_in(v, 0)
                end
            end
            e == 0 && error("unsplittable_flow: walk is stuck at vertex $v (internal error)")
            push!(walk_arc, e); push!(walk_fwd, forward)
            v = forward ? heads[e] : tails[e]
            last = e
            if seen[v] != 0
                first_arc = seen[v]
                break
            end
            seen[v] = length(walk_arc) + 1
        end
        eps = typemax(Int)
        for i in first_arc:length(walk_arc)
            e = walk_arc[i]
            if walk_fwd[i]
                eps = min(eps, f[e])
            else
                h = heads[e]
                for k in 1:K
                    pos[k] == h && demands[k] > f[e] && (eps = min(eps, demands[k] - f[e]))
                end
            end
        end
        0 < eps < typemax(Int) || error("unsplittable_flow: bad augmentation (internal error)")
        for i in first_arc:length(walk_arc)
            f[walk_arc[i]] += walk_fwd[i] ? -eps : eps
            walk_fwd[i] || (raised[walk_arc[i]] = true)
        end
        move_terminals!()
    end
    foreach(reverse!, paths)
    return paths, load, nfallback[]
end

function _check_flow(nv, tails, heads, f, out, inn, source, terminals, demands)
    excess = zeros(Int, nv)
    for e in eachindex(f)
        excess[heads[e]] += f[e]
        excess[tails[e]] -= f[e]
    end
    for k in eachindex(terminals)
        excess[terminals[k]] -= demands[k]
    end
    any(e -> f[e] > 0, inn[source]) && throw(ArgumentError("the flow enters the source"))
    for v in 1:nv
        v == source && continue
        excess[v] == 0 || throw(ArgumentError("flow conservation is violated at vertex $v (excess $(excess[v]))"))
    end
    # acyclic: peel vertices without incoming flow
    indeg = [count(e -> f[e] > 0, inn[v]) for v in 1:nv]
    stack = [v for v in 1:nv if indeg[v] == 0]
    peeled = 0
    while !isempty(stack)
        v = pop!(stack)
        peeled += 1
        for e in out[v]
            f[e] > 0 || continue
            (indeg[heads[e]] -= 1) == 0 && push!(stack, heads[e])
        end
    end
    peeled == nv || throw(ArgumentError("the flow has a cycle"))
    return nothing
end
