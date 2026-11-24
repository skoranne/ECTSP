# Reader for the ITC'02 SOC Test Benchmarks (.soc files).
#
#   SocName d695
#   TotalModules 11
#   Options Power 0 XY 0
#   Module 4 Level 1 Inputs 36 Outputs 39 Bidirs 0 ScanChains 4 : 54 53 52 52
#   Module 4 TotalTests 1
#   Module 4 Test 1 ScanUse 1 TamUse 1 Patterns 105 [Power p]

"One test of a module. `scan_use`: the test shifts the internal scan chains; `tam_use`: the test is applied through the TAM."
struct CoreTest
    index::Int
    scan_use::Bool
    tam_use::Bool
    patterns::Int
    power::Float64          # 0.0 when the benchmark has no power data
end

"One module (core) of an SOC. `level` 0 is the SOC top level."
struct Core
    id::Int
    level::Int
    inputs::Int
    outputs::Int
    bidirs::Int
    scan_chains::Vector{Int}
    tests::Vector{CoreTest}
end

struct SOC
    name::String
    options::Dict{String,Int}
    cores::Vector{Core}     # in module-number order, including module 0
end

struct SocFormatError <: Exception
    file::String
    line::Int
    msg::String
end
Base.showerror(io::IO, e::SocFormatError) = print(io, "SocFormatError: ", e.file, ":", e.line, ": ", e.msg)

function _int(tok, file, ln)
    v = tryparse(Int, tok)
    v === nothing && throw(SocFormatError(file, ln, "expected an integer, found '$tok'"))
    return v
end

# "Key value Key value ..." -> Dict; stops at `stop` (exclusive) and returns the index after it.
function _keyvals(toks, from, file, ln; stop=nothing)
    d = Dict{String,String}()
    i = from
    while i <= length(toks)
        if stop !== nothing && toks[i] == stop
            return d, i + 1
        end
        i == length(toks) && throw(SocFormatError(file, ln, "key '$(toks[i])' has no value"))
        d[String(toks[i])] = String(toks[i+1])
        i += 2
    end
    return d, i
end

function _need(d, key, file, ln)
    haskey(d, key) || throw(SocFormatError(file, ln, "missing field '$key'"))
    return _int(d[key], file, ln)
end

"""
    parse_soc(path) -> SOC

Read one ITC'02 benchmark file.
"""
function parse_soc(path::AbstractString)
    file = String(path)
    name = ""
    total = -1
    options = Dict{String,Int}()
    headers = Dict{Int,NamedTuple}()
    tests = Dict{Int,Vector{CoreTest}}()
    declared = Dict{Int,Int}()
    for (ln, line) in enumerate(eachline(path))
        toks = split(line)
        isempty(toks) && continue
        key = toks[1]
        if key == "SocName"
            length(toks) >= 2 || throw(SocFormatError(file, ln, "SocName without a name"))
            name = String(toks[2])
        elseif key == "TotalModules"
            length(toks) >= 2 || throw(SocFormatError(file, ln, "TotalModules without a count"))
            total = _int(toks[2], file, ln)
        elseif key == "Options"
            d, _ = _keyvals(toks, 2, file, ln)
            for (k, v) in d
                options[k] = _int(v, file, ln)
            end
        elseif key == "Module"
            length(toks) >= 3 || throw(SocFormatError(file, ln, "truncated Module line"))
            id = _int(toks[2], file, ln)
            kind = toks[3]
            if kind == "Level"
                d, next = _keyvals(toks, 3, file, ln; stop=":")
                nsc = _need(d, "ScanChains", file, ln)
                lens = [_int(t, file, ln) for t in toks[next:end]]
                length(lens) == nsc ||
                    throw(SocFormatError(file, ln, "module $id declares $nsc scan chains but lists $(length(lens))"))
                haskey(headers, id) && throw(SocFormatError(file, ln, "module $id defined twice"))
                headers[id] = (level=_need(d, "Level", file, ln), inputs=_need(d, "Inputs", file, ln),
                               outputs=_need(d, "Outputs", file, ln), bidirs=_need(d, "Bidirs", file, ln),
                               scan_chains=lens)
            elseif kind == "TotalTests"
                length(toks) >= 4 || throw(SocFormatError(file, ln, "TotalTests without a count"))
                declared[id] = _int(toks[4], file, ln)
            elseif kind == "Test"
                length(toks) >= 4 || throw(SocFormatError(file, ln, "truncated Test line"))
                d, _ = _keyvals(toks, 5, file, ln)
                pw = haskey(d, "Power") ? tryparse(Float64, d["Power"]) : 0.0
                pw === nothing && throw(SocFormatError(file, ln, "bad Power value '$(d["Power"])'"))
                t = CoreTest(_int(toks[4], file, ln), _need(d, "ScanUse", file, ln) != 0,
                             _need(d, "TamUse", file, ln) != 0, _need(d, "Patterns", file, ln), pw)
                push!(get!(tests, id, CoreTest[]), t)
            else
                throw(SocFormatError(file, ln, "unknown Module record '$kind'"))
            end
        else
            throw(SocFormatError(file, ln, "unknown record '$key'"))
        end
    end
    isempty(name) && throw(SocFormatError(file, 0, "no SocName record"))
    total >= 0 && total != length(headers) &&
        throw(SocFormatError(file, 0, "TotalModules is $total but $(length(headers)) modules are defined"))
    cores = Core[]
    for id in sort!(collect(keys(headers)))
        h = headers[id]
        ts = get(tests, id, CoreTest[])
        haskey(declared, id) && declared[id] != length(ts) &&
            throw(SocFormatError(file, 0, "module $id declares $(declared[id]) tests but lists $(length(ts))"))
        push!(cores, Core(id, h.level, h.inputs, h.outputs, h.bidirs, h.scan_chains, ts))
    end
    for id in keys(tests)
        haskey(headers, id) || throw(SocFormatError(file, 0, "tests given for undefined module $id"))
    end
    return SOC(name, options, cores)
end

"Directory of the bundled ITC'02 benchmark files."
benchmark_dir() = normpath(joinpath(@__DIR__, "..", "benchmarks", "itc02"))

"Names of the bundled benchmarks, smallest first."
function benchmark_names()
    names = [splitext(f)[1] for f in readdir(benchmark_dir()) if endswith(f, ".soc")]
    return sort!(names; by = n -> (parse(Int, filter(isdigit, n)), n))
end

"`load_benchmark(\"d695\")` reads `benchmarks/itc02/d695.soc`."
load_benchmark(name::AbstractString) = parse_soc(joinpath(benchmark_dir(), String(name) * ".soc"))
