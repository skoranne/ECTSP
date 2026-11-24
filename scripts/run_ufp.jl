# Unsplittable-flow sweep over the ITC'02 benchmarks.
#
#   julia -t auto --project=. scripts/run_ufp.jl [options] [benchmark ...]
#
#   --widths 16,32,64     total TAM widths W
#   --max-tams 4          solve for B = 1..max-tams TAMs (every width partition is tried)
#   --out results         output directory
#   --tag NAME            suffix of the output files (default: host name)
#   --compare FILE        a baseline_*.csv of run_baseline.jl; adds the MILP test time to the table
#
# Writes <out>/ufp_<tag>.csv (one row per case) and <out>/ufp_<tag>.md (best B per W).

using ECTSP
using Printf

function main(args)
    opt = Dict{String,Any}("widths" => "16,32,64", "max-tams" => "4", "out" => "results",
                           "tag" => String(first(split(gethostname(), '.'))), "compare" => "")
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

    milp = Dict{Tuple{String,Int},Int}()            # best MILP test time per (soc, W) with B <= maxb
    if !isempty(opt["compare"])
        for line in Iterators.drop(eachline(opt["compare"]), 1)
            c = split(line, ',')
            parse(Int, c[4]) <= maxb || continue
            key = (String(c[1]), parse(Int, c[3]))
            milp[key] = min(get(milp, key, typemax(Int)), parse(Int, c[5]))
        end
    end

    mkpath(opt["out"])
    csv = joinpath(opt["out"], "ufp_$(opt["tag"]).csv")
    md = joinpath(opt["out"], "ufp_$(opt["tag"]).md")
    open(csv, "w") do cio
        open(md, "w") do mio
            println(cio, "soc,cores,W,tams,test_time,bound,gap_pct,seconds,lower_bound_W,widths,assignment")
            println(mio, "# ECTSP by single source unsplittable flow, host $(first(split(gethostname(), '.'))), $(Threads.nthreads()) threads\n")
            println(mio, "Best number of TAMs B <= $maxb per total width W; all width partitions tried. `LB(W)` is the")
            println(mio, "architecture-independent bound, `MILP` the best baseline test time for the same W and B <= $maxb.\n")
            println(mio, "| SOC | cores | W | B | TAM widths | test time | LB(W) | over LB | MILP | over MILP | seconds (all B) |")
            println(mio, "|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|")
            for n in names
                p = TestProblem(load_benchmark(n); max_width=maximum(widths))
                for W in widths
                    best, all = solve_ufp(p, W; max_tams=min(maxb, W, max(ncores(p), 1)))
                    for r in all
                        @printf(cio, "%s,%d,%d,%d,%d,%d,%.4f,%.3f,%d,%s,%s\n", n, ncores(p), W, length(r.widths), r.makespan,
                                r.bound, 100 * gap(r), r.solve_seconds, lower_bound(p, W), join(r.widths, '+'),
                                join(("$(p.ids[i]):$(r.assignment[i])" for i in 1:ncores(p)), ' '))
                    end
                    lb = lower_bound(p, W)
                    secs = sum(r.solve_seconds for r in all)
                    m = get(milp, (n, W), 0)
                    @printf(mio, "| %s | %d | %d | %d | %s | %d | %d | %.2f%% | %s | %s | %.2f |\n", n, ncores(p), W,
                            length(best.widths), join(best.widths, '+'), best.makespan, lb, 100 * (best.makespan - lb) / lb,
                            m == 0 ? "" : string(m), m == 0 ? "" : @sprintf("%.2f%%", 100 * (best.makespan - m) / m), secs)
                    @printf("%-8s W=%2d  B=%d  time %10d  LB %10d  over LB %6.2f%%  %s  %6.2fs  widths %s\n", n, W,
                            length(best.widths), best.makespan, lb, 100 * (best.makespan - lb) / lb,
                            m == 0 ? "" : @sprintf("over MILP %6.2f%%", 100 * (best.makespan - m) / m), secs, best.widths)
                    flush(stdout)
                end
            end
        end
    end
    println("wrote $csv and $md")
    return 0
end

exit(main(ARGS))
