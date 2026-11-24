# Test wrapper design: the test time of a core as a function of the TAM width it is given.
#
# A core connected to w TAM wires gets w wrapper chains. Each wrapper chain is a concatenation of
# wrapper input cells, internal scan chains and wrapper output cells. With s_i / s_o the longest
# wrapper scan-in / scan-out chain and p patterns, the test takes
#
#     T = (1 + max(s_i, s_o)) * p + min(s_i, s_o)        clock cycles.
#
# Internal scan chains are not split. They are partitioned over the wrapper chains by LPT (longest
# first, into the currently shortest chain); the functional inputs and outputs are then spread to
# level the scan-in and scan-out lengths. This is the Design_wrapper scheme of Iyengar, Chakrabarty
# and Marinissen (JETTA 2002) with LPT for the partitioning step.

# Smallest h >= maximum(lens) such that `cells` unit cells fit below level h on the chains `lens`.
function _fill_level(lens::Vector{Int}, cells::Int)
    top = maximum(lens; init=0)
    cells == 0 && return top
    s = sort(lens)
    acc = 0
    for k in eachindex(s)
        acc += s[k]
        h = cld(cells + acc, k)             # level when only the k shortest chains receive cells
        if k == length(s) || h <= s[k+1]
            return max(h, top)
        end
    end
    return top
end

"""
    wrapper_lengths(scan_chains, inputs, outputs, w) -> (s_i, s_o)

Longest wrapper scan-in and scan-out chain when the scan chains and the `inputs` / `outputs`
wrapper cells are distributed over exactly `w` wrapper chains.
"""
function wrapper_lengths(scan_chains::AbstractVector{<:Integer}, inputs::Integer, outputs::Integer, w::Integer)
    w >= 1 || throw(ArgumentError("TAM width must be at least 1"))
    bins = zeros(Int, w)
    for len in sort(collect(Int, scan_chains); rev=true)
        bins[argmin(bins)] += len
    end
    return _fill_level(bins, Int(inputs)), _fill_level(bins, Int(outputs))
end

scan_test_time(si::Integer, so::Integer, patterns::Integer) = (1 + max(si, so)) * patterns + min(si, so)

"""
    test_time(core, test, w) -> Int

Clock cycles of one test of `core` on a TAM of width `w` (best over using 1..w of the wires).
Bidirectional terminals need an input and an output wrapper cell.
"""
function test_time(core::Core, test::CoreTest, w::Integer)
    chains = test.scan_use ? core.scan_chains : Int[]
    nin = core.inputs + core.bidirs
    nout = core.outputs + core.bidirs
    best = typemax(Int)
    for k in 1:w
        si, so = wrapper_lengths(chains, nin, nout, k)
        best = min(best, scan_test_time(si, so, test.patterns))
    end
    return best
end

"Cycles of all TAM-applied tests of `core` on a TAM of width `w`; the tests of one core run one after the other."
test_time(core::Core, w::Integer) = sum(test_time(core, t, w) for t in core.tests if t.tam_use; init=0)

"Cycles of the tests of `core` that do not use the TAM (BIST and the like): patterns only."
bist_time(core::Core) = sum(t.patterns for t in core.tests if !t.tam_use; init=0)
