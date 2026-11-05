using Pkg
Pkg.activate(joinpath(@__DIR__))
for p in ("IJulia", "Plots")   # from the copied repo Project.toml; not needed here
    haskey(Pkg.project().dependencies, p) && Pkg.rm(p)
end
Pkg.add(["CUDA", "Serialization"])
Pkg.status()
