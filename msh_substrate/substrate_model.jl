# Model of the MSH chain on a 2D substrate, shared by msh_substrate_tn.ipynb and the time-dependent notebooks.
#
# Quantics BdG MPO (`msh_substrate`) on sites [nambu; spin; y₁ … y_Ly; x₁ … x_Lx] (most significant bit first, n = i_x + i_y N_x),
# Nambu basis (c↑, c↓, c†↑, c†↓); geometry (`Substrate`); spin textures `S_z`, `S_tilt`; MPS/MPO helpers (`diag_mpo`, `vector_mps`,
# `row_profile_mps`, `dense`, ...); the zero-mode pipeline (`zero_mode_subspace`, `fermion_from_subspace`, `ΔQ_mode`, ...); the
# second parameter point `PC` (Δ = 0.6, μ = −3, J = 3; the defaults of `msh_substrate` are the benchmark point μ = −7, Δ = 2.4, J = 5.8).
# The notebook explains each part; the code here is the code it used to contain. The Python counterpart is substrate_model.py.
# Use: include("substrate_model.jl") (defines everything in the including module).

using TensorBinding, ITensors, ITensorMPS, LinearAlgebra, Random, Logging
const TB = TensorBinding

const σ0 = ComplexF64[1 0; 0 1]
const σx = ComplexF64[0 1; 1 0]
const σy = ComplexF64[0 -im; im 0]
const σz = ComplexF64[1 0; 0 -1]

# diagonal MPO diag(ψ) of an MPS ψ on the position qubits: each site tensor is copied onto the diagonal
function diag_mpo(ψ::MPS, pos)
    W = MPO(pos, "Id")
    for k in eachindex(pos)
        s = pos[k]
        δ = ITensor(ComplexF64, s, s', dag(s''))
        for j in 1:2; δ[s => j, s' => j, s'' => j] = 1; end
        W[k] = replaceprime(ψ[k] * δ, 2 => 0; tags=tags(s))
    end
    return W
end

# MPS of a vector v on qubits `q` (q[1] = most significant bit), exact up to `cutoff`; builds the 2^length(q) array
vector_mps(v::AbstractVector, q; cutoff=1e-14) =
    ITensors.@disable_warn_order MPS(permutedims(reshape(ComplexF64.(v), ntuple(_ -> 2, length(q))), length(q):-1:1), q; cutoff)

# exact diagonal MPO diag(v) on the position qubits (pos[1] = most significant bit); O(length(v)) work
diag_mpo_exact(v::AbstractVector, pos; cutoff=1e-14) = diag_mpo(vector_mps(v, pos; cutoff), pos)

# row_profile_mps(yvecs, w, ypos, xpos): MPS of (yvecs[1] ⊗ … ⊗ yvecs[Ly]) ⊗ w on the qubits [ypos; xpos], i.e. a product of
# single-qubit vectors on the y qubits times a profile w (length 2^Lx) on the x qubits. Costs O(2^Lx), not O(N): only w is
# compressed. With yvecs = the bits of one row it is nonzero on that row; with yvecs = (1, 1) it is the same profile on every row.
function row_profile_mps(yvecs, w, ypos, xpos; cutoff=1e-14)
    ψx = vector_mps(w, xpos; cutoff)
    isempty(ypos) && return ψx
    link = [Index(1, "Link,ly=$k") for k in eachindex(ypos)]          # dimension-1 bonds through the product part
    T = ITensor[]
    for (k, s) in enumerate(ypos)
        t = ITensor(ComplexF64.(yvecs[k]), s) * onehot(link[k] => 1)
        k > 1 && (t *= onehot(link[k-1] => 1))
        push!(T, t)
    end
    ψx[1] *= onehot(link[end] => 1)
    return MPS(vcat(T, collect(ψx)))
end

# dense matrix of an MPO (big-endian over `sites`); small systems only
dense(mpo, sites) = (T = ITensors.@disable_warn_order prod(mpo); D = prod(dim.(sites));
                     reshape(Array(T, reverse(prime.(sites))..., reverse(sites)...), D, D))
adjoint_mpo(K) = swapprime(dag(K), 0, 1)

# ── geometry (section 2 of msh_substrate_tn) ──
struct Substrate
    Lx::Int; Ly::Int; ℓ::Int
end
Nx(g::Substrate) = 2^g.Lx
Ny(g::Substrate) = 2^g.Ly
nsites(g::Substrate) = Nx(g) * Ny(g)
chain_row(g::Substrate) = Ny(g) ÷ 2
chain_x(g::Substrate) = ((Nx(g) - g.ℓ) ÷ 2):((Nx(g) - g.ℓ) ÷ 2 + g.ℓ - 1)

# spin textures of the chain: all along ẑ (topological), and a generic tilted one for the Hamiltonian check
S_z(ix) = (0.0, 0.0, 1.0)
S_tilt(ix) = (sin(0.3ix) * cos(0.7), sin(0.3ix) * sin(0.7), cos(0.3ix))

# ── Hamiltonian MPO (section 3) ──
"""
    msh_substrate(g, S; t, μ, α, Δ, J, cutoff, sites, profile_cutoff) -> (H_BdG::MPO, sites)

BdG MPO of the MSH chain (spin texture `S(ix)`) on the substrate `g`. `sites` reuses existing site indices. `profile_cutoff` is the
truncation cutoff (discarded squared weight) of the exchange profiles along x; the default 1e-14 is exact for the static textures,
while a generic texture (the ramp) needs 1e-20 (with 1e-14 its entries are off by up to ~5e-7).
"""
function msh_substrate(g::Substrate, S; t=1.0, μ=-7.0, α=0.9, Δ=2.4, J=5.8, cutoff=1e-13, sites=nothing, profile_cutoff=1e-14)
    nx = Nx(g)
    ns, ss, pos = isnothing(sites) ? (TB.nambu_index(), TB.spin_index(), siteinds("Qubit", g.Lx + g.Ly)) :
                                     (sites[1], sites[2], sites[3:end])
    ypos, xpos = pos[1:g.Ly], pos[g.Ly+1:end]
    Id = MPO(pos, "Id")
    P(op, M) = TB.prepend_spin(M, ss, op)
    every_row = [[1.0, 1.0] for _ in 1:g.Ly]
    row(iy) = [(b = (iy >> (g.Ly - k)) & 1; b == 0 ? [1.0, 0.0] : [0.0, 1.0]) for k in 1:g.Ly]

    K1 = first(TB.shift_pair_mpos(pos, 1; cyclic=false))
    mask = diag_mpo(row_profile_mps(every_row, [ix != nx - 1 ? 1.0 : 0.0 for ix in 0:nx-1], ypos, xpos), pos)
    Kx = apply(K1, mask; cutoff)
    bonds = [(Kx, -t * σ0 + im * α * σy)]
    g.Ly > 0 && push!(bonds, (first(TB.shift_pair_mpos(pos, nx; cyclic=false)), -t * σ0 - im * α * σx))

    terms = MPO[P(σ0, -μ * Id)]
    for (K, hop) in bonds                                         # c†_{r+d} hop c_r and its h.c.
        push!(terms, P(hop, K), P(Matrix(hop'), adjoint_mpo(K)))
    end
    xs, iy0 = chain_x(g), chain_row(g)
    for (k, σ) in enumerate((σx, σy, σz))
        w = zeros(nx)                                             # J S_k(ix) along the chain row
        for ix in xs
            w[ix + 1] = J * S(ix)[k]
        end
        maximum(abs, w) > 1e-14 && push!(terms, P(σ, diag_mpo(row_profile_mps(row(iy0), w, ypos, xpos; cutoff=profile_cutoff), pos)))
    end
    H0 = +(terms...; cutoff)

    Hp  = P(:iSy, Δ * Id)
    HB  = +(TB.prepend_nambu(H0, ns, :Pp), TB.prepend_nambu(-1.0 * dag(H0), ns, :Ph),
            TB.prepend_nambu(Hp, ns, :tp), TB.prepend_nambu(adjoint_mpo(Hp), ns, :tm); cutoff)
    return HB, [ns; ss; pos]
end

mpo_size(H) = sum(length(storage(T)) for T in H)              # stored elements of the MPO tensors

# ── zero-mode pipeline (section 6) ──
# KrylovKit warns ("ignoring imaginary component") at every DMRG step on these MPOs; the imaginary parts are round-off.
quietly(f) = with_logger(f, ConsoleLogger(stderr, Logging.Error))
norm_Hψ(H, ψ) = norm(orthogonalize(apply(H, ψ; alg="naive", truncate=false), 1)[1])
apply_site(op, ψ, j) = (φ = copy(ψ); φ[j] = noprime(op * φ[j]); φ)
τz_on(ψ) = (φ = copy(ψ); s = siteind(ψ, 1); φ[1] = noprime(ITensor([1.0 0.0; 0.0 -1.0], s', s) * φ[1]); φ)
ΔQ_mode(ψ) = real(inner(ψ, τz_on(ψ)))
mps_size(ψ) = sum(length(storage(T)) for T in ψ)              # stored elements of the MPS tensors

xqubits(g::Substrate) = (3 + g.Ly):(2 + g.Ly + g.Lx)        # site numbers of the x qubits, most significant first
"Particle–hole conjugate C·ψ = τ_x K ψ of an MPS (Nambu site first)."
Cconj(ψ) = (ns = siteind(ψ, 1); φ = dag(ψ); φ[1] = noprime(ITensor([0.0 1.0; 1.0 0.0], ns', ns) * φ[1]); φ)
"Mirror x → N_x − 1 − x: the bitwise complement of i_x, a NOT gate on each x qubit (site numbers xq)."
mirror_x(ψ, sites, xq) = (φ = copy(ψ); for j in xq; φ[j] = noprime(ITensor([0.0 1.0; 1.0 0.0], sites[j]', sites[j]) * φ[j]); end; φ)

"""
    zero_mode_subspace(HB, sites; nzero=2, η=1e-2, xq=nothing, ...) -> NamedTuple

Penalized DMRG on K = H² + η² for `nzero` near-zero states plus one gap-check state. The second state starts from the mirror
image of ψ₁ (x qubits `xq`), orthogonalized to ψ₁ — or, without `xq`, from C·ψ₁ if that has a part orthogonal to ψ₁ — and the
others from random MPS.
Returns the states, their ‖Hψ‖, the gap estimate, K, the times of building K and of the DMRG runs, and the start of state 2
(`start2` = "Cψ1", "mirror" or "random", with the norms of the orthogonalized C·ψ₁ and mirror image).
"""
function zero_mode_subspace(HB, sites; nzero=2, η=1e-2, seed=1234, nsweeps=20, maxdim=[10, 20, 50, 100, 200],
                            cutoff=1e-14, cutoff_K=1e-20, noise=[1e-6, 1e-7, 1e-8, 0.0], weight=10.0,
                            eigsolve_krylovdim=20, eigsolve_maxiter=3, xq=nothing)
    t_K = @elapsed (K = TB.build_K(HB, sites, 0.0, η; cutoff_K))
    Random.seed!(seed)
    ψs = MPS[]; start2 = ("random", NaN, NaN)
    orth(φ) = +(φ, -inner(ψs[1], φ) * ψs[1]; cutoff=1e-14)            # remove the ψ₁ component
    t_dmrg = @elapsed for k in 1:nzero+1
        ψ0 = random_mps(sites; linkdims=10)
        if k == 2
            c = orth(Cconj(ψs[1])); nc = norm(c)
            m = xq === nothing ? nothing : orth(mirror_x(ψs[1], sites, xq)); nm = m === nothing ? NaN : norm(m)
            if nm > 0.1                                  # preferred: the other end of the chain (localized Majorana)
                ψ0 = m / nm; start2 = ("mirror", nc, nm)
            elseif nc > 0.1
                ψ0 = c / nc; start2 = ("Cψ1", nc, nm)
            end
        end
        _, ψ = quietly() do
            isempty(ψs) ? dmrg(K, ψ0; nsweeps, maxdim, cutoff, noise, eigsolve_krylovdim, eigsolve_maxiter, outputlevel=0) :
                          dmrg(K, ψs, ψ0; weight, nsweeps, maxdim, cutoff, noise, eigsolve_krylovdim, eigsolve_maxiter, outputlevel=0)
        end
        push!(ψs, ψ / norm(ψ))
    end
    ψgap = pop!(ψs)
    return (; ψs, E_sub = [norm_Hψ(HB, ψ) for ψ in ψs], E_gap = norm_Hψ(HB, ψgap), K, t_K, t_dmrg, start2)
end

"""
    fermion_from_subspace(HB, sites, ψs, g; cutoff=1e-14) -> NamedTuple

X̂ localization (x qubits), Cγ = γ, and the +E fermion ψf = (γ₁ + iγ₂)/√2. Returns the Majorana x centres (0-indexed), γs, ψf, E_f.
"""
function fermion_from_subspace(HB, sites, ψs, g::Substrate; cutoff=1e-14)
    ns = sites[1]; xq = (3 + g.Ly):(2 + g.Ly + g.Lx)          # site numbers of the x qubits, most significant first
    n1(j) = ITensor([0.0 0.0; 0.0 1.0], sites[j]', sites[j])
    S = [inner(a, b) for a in ψs, b in ψs]
    X = sum(2.0^(g.Lx - k) * [inner(a, apply_site(n1(j), b, j)) for a in ψs, b in ψs] for (k, j) in enumerate(xq))
    FX = eigen(Hermitian((X + X') / 2), Hermitian((S + S') / 2))
    combine(c) = (γ = +(c[1] * ψs[1], c[2] * ψs[2]; cutoff); γ / norm(γ))
    γs = [combine(FX.vectors[:, i]) for i in 1:2]
    τx = ITensor([0.0 1.0; 1.0 0.0], ns', ns)
    C(ψ) = (φ = dag(ψ); φ[1] = noprime(τx * φ[1]); φ)
    γs = [exp(im * angle(inner(γ, C(γ))) / 2) * γ for γ in γs]   # Cγ = e^{iθ}γ → e^{iθ/2}γ is self-conjugate
    ψf = +(γs[1], im * γs[2]; cutoff); ψf = ψf / norm(ψf)
    E_f = real(inner(ψf', HB, ψf))
    E_f < 0 && ((ψf, E_f) = (C(ψf), -E_f))
    return (; centers = FX.values, γs, ψf, E_f, C)
end

# ── the second parameter point, where the zero modes leak into the substrate (section 8) ──
const PC = (μ = -3.0, Δ = 0.6, J = 3.0)

# ── spin ramp of the chain (time-dependent notebooks; the same as Ramp in substrate_model.py) ──
# Two-sided ramp: the chain spins rotate from ẑ to the in-plane antiferromagnet from both ends inwards, one site after the other
# (site k from the nearer end starts at t = step_T·k·rot_T and takes rot_T), until only the middle ℓ_final sites are along ẑ.
# S = (0, sin θ, cos θ), θ = (−1)^ix (π/2) f, f = smoothstep(t/rot_T − step_T·k); ix is the absolute column.
const RAMP_ELL_FINAL, RAMP_STEP_T = 4, 0.1
smoothstep(τ) = τ <= 0 ? 0.0 : τ >= 1 ? 1.0 : sin(π * τ / 2)^2
"Time at which the last rotating site has finished."
ramp_duration(ℓ, rot_T; ℓ_final=RAMP_ELL_FINAL, step_T=RAMP_STEP_T) = (((ℓ - ℓ_final) ÷ 2 - 1) * step_T + 1) * rot_T
"Rotation fraction f ∈ [0, 1] of chain site n = 0 … ℓ−1 at time t."
ramp_f(t, n, ℓ, rot_T; ℓ_final=RAMP_ELL_FINAL, step_T=RAMP_STEP_T) =
    (k = min(n, ℓ - 1 - n); k >= (ℓ - ℓ_final) ÷ 2 ? 0.0 : smoothstep(t / rot_T - step_T * k))
"Spin texture S(ix) of the chain of `g` at time t."
ramp_texture(t, g::Substrate, rot_T; kw...) =
    (x0 = first(chain_x(g)); ix -> (θ = (-1)^ix * (π / 2) * ramp_f(t, ix - x0, g.ℓ, rot_T; kw...); (0.0, sin(θ), cos(θ))))
"BdG MPO of the ramp at time t on the existing `sites` (parameters P as for msh_substrate; exchange profiles with cutoff 1e-20)."
ramp_mpo(t, g::Substrate, rot_T, sites; P...) = first(msh_substrate(g, ramp_texture(t, g, rot_T); sites, profile_cutoff=1e-20, P...))
"Number of steps between recorded snapshots: about n snapshots over the ramp (the same rule in substrate_model.py)."
snap_every(T, dt; n=130) = max(1, round(Int, T / dt / n))
