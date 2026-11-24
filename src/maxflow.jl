# Integer maximum flow (Dinic). Arcs are stored in pairs: arc e (odd) and its reverse e + 1.

struct FlowNet
    head::Vector{Int}
    cap::Vector{Int}                # residual capacity
    adj::Vector{Vector{Int}}
end

FlowNet(nv::Integer) = FlowNet(Int[], Int[], [Int[] for _ in 1:nv])

_rev(e::Int) = isodd(e) ? e + 1 : e - 1

"Add arc u -> v of capacity `c`; returns its id. The flow it carries after `maxflow!` is `arc_flow(g, id)`."
function add_arc!(g::FlowNet, u::Integer, v::Integer, c::Integer)
    push!(g.head, v); push!(g.cap, c); push!(g.adj[u], length(g.head))
    push!(g.head, u); push!(g.cap, 0); push!(g.adj[v], length(g.head))
    return length(g.head) - 1
end

arc_flow(g::FlowNet, e::Int) = g.cap[e+1]

function _augment!(g::FlowNet, level, it, v, t, pushed)
    v == t && return pushed
    while it[v] <= length(g.adj[v])
        e = g.adj[v][it[v]]
        w = g.head[e]
        if g.cap[e] > 0 && level[w] == level[v] + 1
            d = _augment!(g, level, it, w, t, min(pushed, g.cap[e]))
            if d > 0
                g.cap[e] -= d
                g.cap[_rev(e)] += d
                return d
            end
        end
        it[v] += 1
    end
    return 0
end

"Maximum s-t flow; the network is left in its residual state."
function maxflow!(g::FlowNet, s::Integer, t::Integer)
    nv = length(g.adj)
    level = zeros(Int, nv)
    it = ones(Int, nv)
    queue = Int[]
    total = 0
    while true
        fill!(level, 0)
        level[s] = 1
        empty!(queue); push!(queue, s)
        qi = 1
        while qi <= length(queue)
            v = queue[qi]; qi += 1
            for e in g.adj[v]
                w = g.head[e]
                if g.cap[e] > 0 && level[w] == 0
                    level[w] = level[v] + 1
                    push!(queue, w)
                end
            end
        end
        level[t] == 0 && return total
        fill!(it, 1)
        while true
            d = _augment!(g, level, it, s, t, typemax(Int))
            d == 0 && break
            total += d
        end
    end
end
