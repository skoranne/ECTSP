using ECTSP
using Test
using Random

# exhaustive optimum of R||Cmax for small instances
function brute_assignment(p, widths)
    n, B = ncores(p), length(widths)
    best = typemax(Int)
    for code in 0:B^n-1
        load = zeros(Int, B)
        c = code
        for i in 1:n
            j = c % B + 1
            c ÷= B
            load[j] += p.T[i, widths[j]]
        end
        best = min(best, maximum(load))
    end
    return best
end

function compositions(W, B)
    B == 1 && return [[W]]
    out = Vector{Int}[]
    for w in 1:W-B+1, rest in compositions(W - w, B - 1)
        w <= rest[1] && push!(out, vcat(w, rest))
    end
    return out
end

@testset "ECTSP" begin
    @testset "ITC'02 reader" begin
        names = benchmark_names()
        @test length(names) == 12
        for n in names
            soc = load_benchmark(n)
            @test soc.name == n
            @test all(c -> c.id == soc.cores[1].id || c.level >= 0, soc.cores)
        end
        d = load_benchmark("d695")
        @test length(d.cores) == 11
        @test d.cores[5].scan_chains == [54, 53, 52, 52]      # module 4
        @test d.cores[5].tests[1].patterns == 105
        h = load_benchmark("h953")
        @test h.options["Power"] == 1
        @test h.cores[2].tests[1].power == 565860000
        @test count(t -> !t.tam_use, vcat([c.tests for c in load_benchmark("d281").cores]...)) == 8
        mktemp() do path, io
            println(io, "SocName x\nTotalModules 1\nModule 0 Level 0 Inputs 1 Outputs 1 Bidirs 0 ScanChains 2 : 5")
            close(io)
            @test_throws SocFormatError parse_soc(path)
        end
    end

    @testset "wrapper design" begin
        @test wrapper_lengths(Int[], 32, 32, 1) == (32, 32)
        @test wrapper_lengths(Int[], 32, 32, 4) == (8, 8)
        @test wrapper_lengths([54, 53, 52, 52], 36, 39, 1) == (247, 250)
        @test wrapper_lengths([54, 53, 52, 52], 36, 39, 2) == (124, 125)
        @test wrapper_lengths([10, 2], 3, 0, 2) == (10, 10)   # inputs fill the short chain first
        @test wrapper_lengths([10, 2], 12, 0, 2) == (12, 10)
        @test scan_test_time(247, 250, 105) == 251 * 105 + 247
        p = TestProblem(load_benchmark("d695"); max_width=32)
        @test ncores(p) == 10
        @test p.T[1, 1] == (1 + 32) * 12 + 32                 # module 1: 32 in, 32 out, 12 patterns
        @test all(p.T[:, 1:31] .>= p.T[:, 2:32])   # monotone in the width
        @test all(p.T .> 0)
    end

    @testset "bounds and schedules" begin
        p = TestProblem(load_benchmark("d695"); max_width=16)
        @test lower_bound(p, 16) >= maximum(p.T[:, 16])
        s = build_schedule(p, [8, 8], [isodd(i) ? 1 : 2 for i in 1:10])
        @test validate(s, p)
        @test s.makespan == max(sum(p.T[1:2:9, 8]), sum(p.T[2:2:10, 8]))
        @test s.makespan >= lower_bound(p, 16)
        bad = Schedule(s.problem, s.method, 16, [8, 8], s.assignment, s.slots, s.makespan - 1, 0, :X, 0.0)
        @test_throws ErrorException validate(bad, p)
    end

    @testset "MILP baseline" begin
        u = TestProblem(load_benchmark("u226"); max_width=8)       # 5 cores
        for widths in ([4, 4], [1, 2, 5], [8])
            s = solve_assignment(u, widths)
            @test s.status == :OPTIMAL
            @test s.makespan == brute_assignment(u, widths)
        end
        for B in 1:3
            s = solve_baseline(u, 8, B; mip_gap=0.0)
            @test validate(s, u)
            @test length(s.widths) == B && sum(s.widths) == 8
            @test s.makespan == minimum(brute_assignment(u, w) for w in compositions(8, B))
            @test s.bound <= s.makespan
        end
        d = TestProblem(load_benchmark("d695"); max_width=16)
        best, all = solve_baseline(d, 16; max_tams=3, time_limit=60)
        @test length(all) == 3
        @test best.makespan == minimum(s.makespan for s in all)
        @test all[1].makespan == sum(d.T[:, 16])
        @test all[2].makespan <= all[1].makespan                 # more TAMs never hurt here
        @test lower_bound(d, 16) <= best.makespan <= 1.05 * lower_bound(d, 16)
    end

    @testset "maximum flow" begin
        g = ECTSP.FlowNet(4)
        a = ECTSP.add_arc!(g, 1, 2, 3); b = ECTSP.add_arc!(g, 1, 3, 2)
        ECTSP.add_arc!(g, 2, 3, 5); ECTSP.add_arc!(g, 2, 4, 2); ECTSP.add_arc!(g, 3, 4, 3)
        @test ECTSP.maxflow!(g, 1, 4) == 5
        @test ECTSP.arc_flow(g, a) == 3 && ECTSP.arc_flow(g, b) == 2
    end

    @testset "unsplittable flow (Dinitz-Garg-Goemans)" begin
        # two terminals of demand 2 and 3 split over two parallel routes
        tails, heads = [1, 1, 2, 3], [2, 3, 4, 4]
        paths, load, exc = unsplittable_flow(4, tails, heads, [3, 2, 3, 2], 1, [4, 4], [2, 3])
        @test exc == 0 && sort(load[1:2]) in ([0, 5], [2, 3])
        @test_throws ArgumentError unsplittable_flow(4, tails, heads, [3, 2, 3, 1], 1, [4, 4], [2, 3])   # conservation
        @test_throws ArgumentError unsplittable_flow(2, [1, 2], [2, 1], [1, 1], 1, Int[], Int[])         # cycle
        rng = MersenneTwister(2026)
        worst = 0.0
        for _ in 1:3000
            nv = rand(rng, 3:14)
            tails, heads = Int[], Int[]
            for u in 1:nv, v in u+1:nv
                (v == u + 1 || rand(rng) < 0.45) && for _ in 1:rand(rng, 1:2)
                    push!(tails, u); push!(heads, v)
                end
            end
            K = rand(rng, 1:12)
            term = rand(rng, 2:nv, K)
            dem = rand(rng, 1:rand(rng, (3, 20, 1000)), K)
            f = zeros(Int, length(tails))
            for k in 1:K                      # split every demand over random source-terminal paths
                left = dem[k]
                while left > 0
                    part = rand(rng) < 0.3 ? left : rand(rng, 1:left)
                    left -= part
                    v = term[k]
                    while v != 1
                        e = rand(rng, findall(==(v), heads))
                        f[e] += part
                        v = tails[e]
                    end
                end
            end
            paths, load, exc = unsplittable_flow(nv, tails, heads, f, 1, term, dem)
            ok = exc == 0
            routed = zeros(Int, length(tails))
            for k in 1:K
                v = 1
                for e in paths[k]
                    ok &= tails[e] == v
                    v = heads[e]
                    routed[e] += dem[k]
                end
                ok &= v == term[k]
            end
            ok &= routed == load && all(load .<= f .+ maximum(dem)) && all(load[f .== 0] .== 0)
            ok || (@test ok; break)
            worst = max(worst, maximum((load .- f) ./ maximum(dem)))
        end
        @test 0.5 < worst <= 1.0              # the bound is attained closely, never exceeded
    end

    @testset "ECTSP by unsplittable flow" begin
        @test width_partitions(8, 3) == [[1, 1, 6], [1, 2, 5], [1, 3, 4], [2, 2, 4], [2, 3, 3]]
        @test length(width_partitions(64, 4)) == 1906
        d = TestProblem(load_benchmark("d695"); max_width=16)
        for widths in ([8, 8], [4, 4, 4, 4], [2, 6, 8], [1, 3, 5, 7])
            opt = solve_assignment(d, widths; mip_gap=0.0)
            s = solve_ufp(d, widths)
            @test validate(s, d) && s.method == :ufp
            @test s.bound == ufp_fractional_bound(d, widths) <= opt.makespan <= s.makespan
            # the rounding itself: no TAM gains more than one demand over its fractional load
            Tw = d.T[:, widths]
            dem = vec(minimum(Tw; dims=2))
            for rho in (1.0, 1.5, Inf)
                a, C, excess, exc = ECTSP._ufp_round(dem, Tw, rho)
                @test exc == 0 && excess <= 1.0
                load = [sum(dem[i] for i in 1:10 if a[i] == j; init=0) for j in eachindex(widths)]
                @test maximum(load) <= C + maximum(dem)
                @test all(Tw[i, a[i]] <= rho * dem[i] for i in 1:10)
            end
            # test data as commodity: TAM j has capacity C * w_j, a core needs its least wire-cycles
            bits, unit = ECTSP._ufp_model(Tw, widths, :bits)
            @test unit == widths && bits == [minimum(widths .* Tw[i, :]) for i in 1:10]
            for rho in (1.0, 1.5, Inf)
                a, C, excess, exc = ECTSP._ufp_round(bits, Tw, unit, rho)
                @test exc == 0 && excess <= 1.0
                load = [sum(bits[i] for i in 1:10 if a[i] == j; init=0) for j in eachindex(widths)]
                @test all(load .<= C .* widths .+ maximum(bits))
                @test all(widths[a[i]] * Tw[i, a[i]] <= rho * bits[i] for i in 1:10)
            end
            for m in UFP_MODELS
                sm = solve_ufp(d, widths; model=m)
                @test validate(sm, d) && s.makespan <= sm.makespan
                @test sm.bound == ufp_fractional_bound(d, widths; model=m) <= s.bound
            end
            @test_throws ArgumentError solve_ufp(d, widths; model=:nope)
            si = solve_ufp(d, widths; improve=true)
            @test validate(si, d) && opt.makespan <= si.makespan <= s.makespan
            if allequal(widths)                # identical TAMs: factor 2 of the optimum
                @test s.makespan <= 2 * opt.makespan
            end
        end
        best, every = solve_ufp(d, 16; max_tams=3)
        milp, _ = solve_baseline(d, 16; max_tams=3)
        @test every[1].makespan == sum(d.T[:, 16])
        @test lower_bound(d, 16) <= best.bound <= milp.makespan <= best.makespan <= 1.25 * milp.makespan
    end

    @testset "greedy, best fit, shortest job first" begin
        # identical TAMs, times 5 4 3 3 3: longest first reaches the optimum 9, shortest first gives 11
        q = TestProblem("toy", collect(1:5), repeat([5, 4, 3, 3, 3], 1, 2), zeros(Int, 5))
        @test solve_greedy(q, [1, 1]).makespan == 10
        @test solve_best_fit(q, [1, 1]).makespan == 9
        @test solve_sjf(q, [1, 1]).makespan == 11
        @test solve_sjf(q, [1, 1]).assignment == [1, 2, 1, 2, 1]      # 3 3 3 4 5 in turn to the TAM that finishes first
        @test_throws ArgumentError solve_heuristic(q, [1, 1]; method=:nope)
        d = TestProblem(load_benchmark("d695"); max_width=16)
        for widths in ([16], [8, 8], [2, 6, 8], [1, 3, 5, 7]), m in HEURISTICS
            opt = solve_assignment(d, widths; mip_gap=0.0)
            s = solve_heuristic(d, widths; method=m)
            @test validate(s, d) && s.method == m
            @test s.bound <= opt.makespan <= s.makespan
        end
        milp, _ = solve_baseline(d, 16; max_tams=3)
        for (f, m) in ((solve_greedy, :greedy), (solve_best_fit, :best_fit), (solve_sjf, :sjf))
            best, every = f(d, 16; max_tams=3)
            @test best.method == m && length(every) == 3
            @test every[1].makespan == sum(d.T[:, 16])
            @test milp.makespan <= best.makespan <= 1.25 * milp.makespan
            fixed = f(d, best.widths)
            @test fixed.makespan == best.makespan && fixed.assignment == best.assignment
        end
    end

    @testset "Pareto curves as input" begin
        d = TestProblem(load_benchmark("d695"); max_width=32)
        curves = [pareto_curve(d, i) for i in 1:ncores(d)]
        @test curves[1] == [(1, 428), (2, 220), (3, 155), (4, 116), (5, 103), (6, 90), (7, 77), (8, 64), (11, 51), (16, 38), (32, 25)]
        @test all(c -> issorted(c; by=first) && issorted(c; by=last, rev=true) && allunique(last.(c)), curves)
        g = TestProblem("d695", curves; max_width=32, ids=d.ids)
        @test g.T == d.T && g.ids == d.ids
        @test solve_best_fit(g, 16, 3).makespan == solve_best_fit(d, 16, 3).makespan
        q = TestProblem("toy", [[(1, 100), (4, 30), (3, 50), (5, 40)], [(1, 7)]]; max_width=6)
        @test q.T == [100 100 50 30 30 30; 7 7 7 7 7 7]
        @test pareto_curve(q, 1) == [(1, 100), (3, 50), (4, 30)]      # (5, 40) is dominated
        @test_throws ArgumentError TestProblem("bad", [[(2, 10)]])
    end
end
