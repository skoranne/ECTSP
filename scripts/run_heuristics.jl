# Comparison of the combinatorial solvers on the ITC'02 benchmarks.
#
#   julia -t auto --project=. scripts/run_heuristics.jl [options] [benchmark ...]
#
#   --widths 16,32,64                    total TAM widths W
#   --max-tams 4                         B = 1..max-tams TAMs (every width partition is tried)
#   --methods ufp,greedy,best_fit,sjf    solvers to run
#   --out results                        output directory
#   --tag NAME                           suffix of the output files (default: host name)
#   --compare FILE                       a baseline_*.csv of run_baseline.jl; adds the MILP column
#
# Writes <out>/heuristics_<tag>.csv (one row per SOC, W, B, method) and <out>/heuristics_<tag>.md
# (best B per SOC, W and method; percentages are the excess over the MILP, or over LB(W) without --compare).

using ECTSP
using Printf

function main(args)
    opt = Dict{String,Any}("widths" => "16,32,64", "max-tams" => "4", "methods" => "ufp,greedy,best_fit,sjf",
                           "out" => "results", "tag" => String(first(split(gethostname(), '.'))), "compare" => "")
    names = String[]
    i = 1
    while i <= length(args)
        if startswith(args[i], "--")
            haskey(opt, args[i][3:end]) && i < length(args) || error("bad option $(args[i])")
            opt[args[i][3:end]] = args[i+1]
            i += 2
        else
            push!(names, args[i]); i += 1
        end
    end
    isempty(names) && (names = benchmark_names())
    widths = parse.(Int, split(opt["widths"], ','))
    maxb = parse(Int, opt["max-tams"])
    methods = Symbol.(split(opt["methods"], ','))
    all(m -> m == :ufp || m in HEURISTICS, methods) || error("unknown method in $(opt["methods"])")
    run(p, W, B, m) = m == :ufp ? solve_ufp(p, W, B) : solve_heuristic(p, W, B; method=m)

    milp = Dict{Tuple{String,Int},Int}()
    if !isempty(opt["compare"])
        for line in Iterators.drop(eachline(opt["compare"]), 1)
            c = split(line, ',')
            parse(Int, c[4]) <= maxb || continue
            key = (String(c[1]), parse(Int, c[3]))
            milp[key] = min(get(milp, key, typemax(Int)), parse(Int, c[5]))
        end
    end

    mkpath(opt["out"])
    csv = joinpath(opt["out"], "heuristics_$(opt["tag"]).csv")
    md = joinpath(opt["out"], "heuristics_$(opt["tag"]).md")
    excess = Dict(m => Float64[] for m in methods)      # per (SOC, W): relative excess over the reference
    seconds = Dict(m => 0.0 for m in methods)
    wins = Dict(m => 0 for m in methods)
    open(csv, "w") do cio
        open(md, "w") do mio
            println(cio, "soc,cores,W,tams,method,test_time,seconds,lower_bound_W,widths,assignment")
            println(mio, "# ECTSP solver comparison, host $(first(split(gethostname(), '.'))), $(Threads.nthreads()) threads\n")
            println(mio, "Best number of TAMs B <= $maxb per total width W, all width partitions tried. Test time in cycles;")
            println(mio, "in brackets the excess over ", isempty(milp) ? "LB(W)" : "the MILP", " and the B that was chosen.\n")
            println(mio, "| SOC | W | LB(W) | MILP | ", join(methods, " | "), " |")
            println(mio, "|---|---:|---:|---:|", repeat("---:|", length(methods)))
            for n in names
                p = TestProblem(load_benchmark(n); max_width=maximum(widths))
                for W in widths
                    lb = lower_bound(p, W)
                    ref = get(milp, (n, W), lb)
                    cells = String[]
                    best_time = Dict{Symbol,Int}()
                    for m in methods
                        all = [run(p, W, B, m) for B in 1:min(maxb, W, max(ncores(p), 1))]
                        for r in all
                            @printf(cio, "%s,%d,%d,%d,%s,%d,%.4f,%d,%s,%s\n", n, ncores(p), W, length(r.widths), m, r.makespan,
                                    r.solve_seconds, lb, join(r.widths, '+'),
                                    join(("$(p.ids[i]):$(r.assignment[i])" for i in 1:ncores(p)), ' '))
                        end
                        b = all[argmin([r.makespan for r in all])]
                        best_time[m] = b.makespan
                        push!(excess[m], (b.makespan - ref) / ref)
                        seconds[m] += sum(r.solve_seconds for r in all)
                        push!(cells, @sprintf("%d (%+.2f%%, B=%d)", b.makespan, 100 * (b.makespan - ref) / ref, length(b.widths)))
                    end
                    least = minimum(values(best_time))
                    for m in methods
                        best_time[m] == least && (wins[m] += 1)
                    end
                    println(mio, "| $n | $W | $lb | ", haskey(milp, (n, W)) ? milp[(n, W)] : "", " | ", join(cells, " | "), " |")
                    println(rpad(n, 9), "W=", lpad(W, 2), "  ", join(("$m $(best_time[m])" for m in methods), "  "))
                    flush(stdout)
                end
            end
            println(mio, "\n## Summary over $(length(first(values(excess)))) (SOC, W) cases\n")
            println(mio, "| method | mean excess | median | worst | best or tied | total seconds |")
            println(mio, "|---|---:|---:|---:|---:|---:|")
            for m in methods
                e = sort(excess[m])
                med = (e[(length(e)+1)÷2] + e[length(e)÷2+1]) / 2
                line = @sprintf("| %s | %.2f%% | %.2f%% | %.2f%% | %d | %.2f |", m, 100 * sum(e) / length(e), 100 * med,
                                100 * e[end], wins[m], seconds[m])
                println(mio, line)
                println(line)
            end
        end
    end
    println("wrote $csv and $md")
    return 0
end

exit(main(ARGS))
