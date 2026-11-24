# Baseline sweep over the ITC'02 benchmarks.
#
#   julia -t auto --project=. scripts/run_baseline.jl [options] [benchmark ...]
#
#   --widths 16,32,64     total TAM widths W
#   --max-tams 4          solve for B = 1..max-tams TAMs
#   --time-limit 120      seconds per (benchmark, W, B)
#   --solver highs|cuopt  MILP solver (cuopt needs cuOpt.jl and libcuopt in the environment)
#   --solver-threads 2    threads per solve; (benchmark, W, B) cases run in parallel on Julia threads
#   --out results         output directory
#   --tag NAME            suffix of the output files (default: host name)
#
# Writes <out>/baseline_<tag>.csv (one row per case) and <out>/baseline_<tag>.md (best B per W).

using ECTSP
using Printf

function parse_args(args)
    opt = Dict{String,Any}("widths" => [16, 32, 64], "max-tams" => 4, "time-limit" => 120.0, "solver" => "highs",
                           "solver-threads" => 2, "out" => "results", "tag" => first(split(gethostname(), '.')))
    names = String[]
    i = 1
    while i <= length(args)
        a = args[i]
        if startswith(a, "--")
            key = a[3:end]
            haskey(opt, key) || error("unknown option $a")
            i < length(args) || error("option $a needs a value")
            v = args[i+1]
            opt[key] = key == "widths" ? parse.(Int, split(v, ',')) :
                       key in ("max-tams", "solver-threads") ? parse(Int, v) :
                       key == "time-limit" ? parse(Float64, v) : v
            i += 2
        else
            push!(names, a)
            i += 1
        end
    end
    return opt, isempty(names) ? benchmark_names() : names
end

function main(args)
    opt, names = parse_args(args)
    optimizer = if opt["solver"] == "cuopt"
        @eval Main import cuOpt
        Base.invokelatest(() -> Main.cuOpt.Optimizer)
    elseif opt["solver"] == "highs"
        ECTSP.HiGHS.Optimizer
    else
        error("unknown solver $(opt["solver"])")
    end
    threads = opt["solver"] == "highs" ? opt["solver-threads"] : nothing
    widths = opt["widths"]
    problems = Dict(n => TestProblem(load_benchmark(n); max_width=maximum(widths)) for n in names)
    cases = [(n, W, B) for n in names for W in widths for B in 1:min(opt["max-tams"], W, ncores(problems[n]))]
    results = Vector{Any}(undef, length(cases))
    lk = ReentrantLock()
    # cuOpt owns the GPU: one solve at a time
    ntasks = opt["solver"] == "cuopt" ? 1 : Threads.nthreads()
    next = Threads.Atomic{Int}(1)
    @sync for _ in 1:ntasks
        Threads.@spawn while true
            k = Threads.atomic_add!(next, 1)
            k > length(cases) && break
            n, W, B = cases[k]
            results[k] = try
                Base.invokelatest(solve_baseline, problems[n], W, B; optimizer=optimizer,
                                  time_limit=opt["time-limit"], threads=threads)
            catch err
                sprint(showerror, err)
            end
            lock(lk) do
                r = results[k]
                if r isa Schedule
                    @printf("%-8s W=%2d B=%d  time %10d  bound %10d  gap %6.2f%%  %-12s %7.1fs  widths %s\n",
                            n, W, B, r.makespan, r.bound, 100 * gap(r), r.status, r.solve_seconds, r.widths)
                else
                    @printf("%-8s W=%2d B=%d  FAILED: %s\n", n, W, B, r)
                end
                flush(stdout)
            end
        end
    end

    mkpath(opt["out"])
    csv = joinpath(opt["out"], "baseline_$(opt["tag"]).csv")
    open(csv, "w") do io
        println(io, "soc,cores,W,tams,test_time,bound,gap_pct,status,seconds,lower_bound_W,widths,assignment")
        for (k, (n, W, B)) in enumerate(cases)
            r = results[k]
            r isa Schedule || continue
            p = problems[n]
            @printf(io, "%s,%d,%d,%d,%d,%d,%.4f,%s,%.2f,%d,%s,%s\n", n, ncores(p), W, B, r.makespan, r.bound,
                    100 * gap(r), r.status, r.solve_seconds, lower_bound(p, W), join(r.widths, '+'),
                    join(("$(p.ids[i]):$(r.assignment[i])" for i in 1:ncores(p)), ' '))
        end
    end
    md = joinpath(opt["out"], "baseline_$(opt["tag"]).md")
    open(md, "w") do io
        println(io, "# ECTSP baseline (JuMP + $(opt["solver"])), host $(first(split(gethostname(), '.')))\n")
        println(io, "Best number of TAMs B <= $(opt["max-tams"]) per total width W; $(opt["time-limit"]) s per (SOC, W, B).")
        println(io, "`LB(W)` is the architecture-independent bound; `gap` is against the solver bound for that B.\n")
        println(io, "| SOC | cores | W | B | TAM widths | test time | LB(W) | over LB | gap for B | status |")
        println(io, "|---|---:|---:|---:|---|---:|---:|---:|---:|---|")
        for n in names, W in widths
            ks = [k for (k, c) in enumerate(cases) if c[1] == n && c[2] == W && results[k] isa Schedule]
            isempty(ks) && continue
            r = results[ks[argmin([results[k].makespan for k in ks])]]
            lb = lower_bound(problems[n], W)
            @printf(io, "| %s | %d | %d | %d | %s | %d | %d | %.2f%% | %.2f%% | %s |\n", n, ncores(problems[n]), W,
                    length(r.widths), join(r.widths, '+'), r.makespan, lb, 100 * (r.makespan - lb) / lb, 100 * gap(r), r.status)
        end
    end
    nfail = count(r -> !(r isa Schedule), results)
    println("wrote $csv and $md; $(length(cases) - nfail) solved, $nfail failed")
    return nfail == 0 ? 0 : 1
end

exit(main(ARGS))
