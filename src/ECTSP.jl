"""
    ECTSP

Embedded core test scheduling for core-based SOCs: ITC'02 benchmark reader, wrapper design
(test time as a function of TAM width), and a JuMP mixed-integer baseline for the TAM
partitioning and core assignment problem.
"""
module ECTSP

using JuMP
import HiGHS
using Printf

export SOC, Core, CoreTest, SocFormatError, parse_soc, load_benchmark, benchmark_names, benchmark_dir
export wrapper_lengths, scan_test_time, test_time, bist_time
export TestProblem, ncores, max_width, lower_bound, pareto_curve
export Schedule, gap, build_schedule, validate, print_schedule
export solve_assignment, solve_baseline
export unsplittable_flow, solve_ufp, ufp_fractional_bound, width_partitions, UFP_SLOWDOWNS
export solve_heuristic, solve_greedy, solve_best_fit, solve_sjf, HEURISTICS

include("itc02.jl")
include("wrapper.jl")
include("problem.jl")
include("milp.jl")
include("maxflow.jl")
include("ufp.jl")
include("ufp_schedule.jl")
include("heuristics.jl")

end # module
