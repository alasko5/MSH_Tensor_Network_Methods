# TDVP step benchmark, reproducing RUN_IDLE of msh_substrate_td_tn.ipynb (cell 20): one
# TB.evolve_with_tdvp_timedep(t -> ramp_mpo(tm + t, g, rot, sites; P...), ψ, 1, 0.1; maxdim, cutoff, tol=1e-12, krylovdim=30,
# reverse_step=true) at tm = T/2, median of 3 after a first one. Reads msh_substrate/ only.
#
# Usage: julia --project=gpu_bench/jlenv tdvp_bench.jl cpu|gpu [case ...]   (cases: bench9 bench10 bench11 bench11s leak128 leak256)
# `cpu`: single BLAS thread. Times (a) the reference call as is (H build inside) and (b) H build and sweep separately.
# `gpu`: H(t) built on the CPU (single thread), moved to the GPU with adapt(CuArray, ·) (ComplexF64 kept), then the same
#        TB.evolve_with_tdvp_timedep with the GPU MPO and GPU MPS. Checks the GPU step against a CPU step from the same state.
# Appends rows to timings.csv next to this file.

using LinearAlgebra, Serialization, Printf, Statistics
BLAS.set_num_threads(1)
const MSH = joinpath(homedir(), "MSH_Tensor_Network_Methods", "msh_substrate")
include(joinpath(MSH, "substrate_model.jl"))
const DEVICE = ARGS[1]
if DEVICE == "gpu"
    using CUDA
    using ITensors: adapt
end

const DT = 0.1
const CASES = Dict(   # name => (point, L, rot, maxdim, cutoff, file)
    "bench9"   => ("bench", 9, 10.0, 256, 1e-14, "mpo16/td_tn_large_bench_9_9_rot10_m256_c1.0e-14.jls"),
    "bench10"  => ("bench", 10, 10.0, 256, 1e-14, "mpo16/td_tn_large_bench_10_10_rot10_m256_c1.0e-14.jls"),
    "bench11"  => ("bench", 11, 10.0, 256, 1e-14, "mpo16/td_tn_large_bench_11_11_rot10_m256_c1.0e-14.jls"),
    "bench11s" => ("bench", 11, 100.0, 256, 1e-14, "mpo16/td_tn_large_bench_11_11_rot100_m256_c1.0e-14.jls"),
    "leak128"  => ("leak", 8, 100.0, 128, 1e-14, "td_tn_large_leak_8_8_rot100_m128_c1.0e-14.jls"),
    "leak256"  => ("leak", 8, 100.0, 256, 1e-12, "td_tn_large_leak_8_8_rot100_m256_c1.0e-12.jls"),
)
const POINTS = Dict("bench" => ((;), 12), "leak" => (PC, 64))

function write_rows(rows)
    f = joinpath(@__DIR__, "timings.csv"); new = !isfile(f)
    open(f, "a") do io
        new && println(io, "method,device,point,L,rot_T,chi,part,seconds")
        for r in rows; println(io, join(r, ",")); end
    end
end

ongpu(t) = ITensors.NDTensors.data(ITensors.storage(t)) isa CuArray
sync() = DEVICE == "gpu" ? CUDA.synchronize() : nothing
step(Hf, ψ, maxdim, cutoff) = TB.evolve_with_tdvp_timedep(Hf, ψ, 1, DT; maxdim, cutoff, tol=1e-12, krylovdim=30, reverse_step=true)[end]

for name in ARGS[2:end]
    pt, L, rot, maxdim, cutoff, file = CASES[name]; P, ℓ = POINTS[pt]
    g = Substrate(L, L, ℓ); d = deserialize(joinpath(MSH, "data", file)); tm = 0.5 * ramp_duration(ℓ, rot)
    χ = maxlinkdim(d.ψ); dev = DEVICE == "gpu" ? "gpu" : "cpu_1thr"
    build() = ramp_mpo(tm + DT / 2, g, rot, d.sites; P...)
    if DEVICE == "cpu"
        ref() = step(t -> ramp_mpo(tm + t, g, rot, d.sites; P...), d.ψ, maxdim, cutoff)
        ref(); tref = median([@elapsed(ref()) for _ in 1:3])
        tb, ts = Float64[], Float64[]
        for k in 1:4
            b = @elapsed (H = build()); s = @elapsed step(_ -> H, d.ψ, maxdim, cutoff)
            k > 1 && (push!(tb, b); push!(ts, s))
        end
        @printf("CPU %-9s χ=%3d: reference call %.3f s | H_build %.3f  sweep %.3f  sum %.3f s\n", name, χ, tref,
                median(tb), median(ts), median(tb .+ ts))
        write_rows([["tdvp", dev, pt, L, rot, χ, "reference_call", tref], ["tdvp", dev, pt, L, rot, χ, "H_build", median(tb)],
                    ["tdvp", dev, pt, L, rot, χ, "step", median(ts)], ["tdvp", dev, pt, L, rot, χ, "total", median(tb .+ ts)]])
    else
        ψg = adapt(CuArray, d.ψ); sync()
        @assert eltype(ψg[1]) == ComplexF64 && all(ongpu, ψg)
        tb, tt, ts = Float64[], Float64[], Float64[]; ψ1g = nothing; Hlast = nothing
        for k in 1:4
            b = @elapsed (H = build()); t = @elapsed (Hg = adapt(CuArray, H); sync())
            s = @elapsed (ψ1g = step(_ -> Hg, ψg, maxdim, cutoff); sync())
            k > 1 && (push!(tb, b); push!(tt, t); push!(ts, s)); Hlast = H
        end
        @assert all(ongpu, ψ1g) && eltype(ψ1g[1]) == ComplexF64
        used = try   # device memory in use (all processes), MiB
            parse(Float64, strip(read(`nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits`, String)))
        catch; NaN; end
        @printf("GPU %-9s χ=%3d: H_build %.3f  transfer %.4f  sweep %.3f  total %.3f s  (device mem in use %.0f MiB)\n", name, χ,
                median(tb), median(tt), median(ts), median(tb .+ tt .+ ts), used)
        # accuracy: the same step on the CPU from the same state with the same H
        ψ1c = step(_ -> Hlast, d.ψ, maxdim, cutoff); ψ1 = adapt(Array, ψ1g)
        ov = inner(ψ1c, ψ1); df = norm(ψ1c - ψ1)
        @printf("CHECK %-9s: |<ψ_cpu|ψ_gpu>| - 1 = %.2e   ‖ψ_cpu - ψ_gpu‖ = %.2e   χ_cpu=%d χ_gpu=%d\n", name, abs(ov) - 1, df,
                maxlinkdim(ψ1c), maxlinkdim(ψ1))
        write_rows([["tdvp", dev, pt, L, rot, χ, "H_build", median(tb)], ["tdvp", dev, pt, L, rot, χ, "transfer", median(tt)],
                    ["tdvp", dev, pt, L, rot, χ, "step", median(ts)], ["tdvp", dev, pt, L, rot, χ, "total", median(tb .+ tt .+ ts)]])
        CUDA.reclaim()
    end
    flush(stdout)
end
