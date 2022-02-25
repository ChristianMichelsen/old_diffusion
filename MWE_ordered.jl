
using Turing
using Distributions
using StatsPlots

include("bayesian_utils.jl")


function get_data(; μ₁ = 1, μ₂ = 2, N = 1000, fraction = 0.5)
    dist1 = Rayleigh(μ₁)
    dist2 = Rayleigh(μ₂)
    N1 = round(Int, fraction * N)
    N2 = round(Int, (1 - fraction) * N)
    data = vcat(rand(dist1, N1), rand(dist2, N2))
    return data
end


function get_data_3D(; μ₁ = 1, μ₂ = 2, μ₃ = 3, N = 1000, fraction1 = 0.3, fraction2 = 0.3)
    dists = [Rayleigh(μᵢ) for μᵢ in [μ₁, μ₂, μ₃]]
    N1 = round(Int, fraction1 * N)
    N2 = round(Int, fraction2 * N)
    N3 = round(Int, (1 - fraction1 - fraction2) * N)
    Ns = [N1, N2, N3]
    data = vcat([rand(dist, N) for (dist, N) in zip(dists, Ns)]...)
    return data
end


@model function diffusion_ordered(Δr)

    N = length(Δr)

    ΔD ~ filldist(Exponential(), 2)
    D = cumsum(ΔD)

    θ ~ Uniform(0, 1)
    w = [θ, 1 - θ]

    dists = [Rayleigh(d) for d in D]
    Δr ~ filldist(MixtureModel(dists, w), N)
    return (; D)
end


@model function diffusion_ordered_dirichlet(Δr, K = 2)

    N = length(Δr)

    ΔD ~ filldist(Exponential(), K)
    D = cumsum(ΔD)

    w ~ Dirichlet(K, 1)

    dists = [Rayleigh(d) for d in D]
    Δr ~ filldist(MixtureModel(dists, w), N)
    return (; D)
end



@model function diffusion_ordered_Δw(Δr, K = 2)

    N = length(Δr)

    ΔD ~ filldist(Exponential(), K)
    D = cumsum(ΔD)

    Δw ~ filldist(Exponential(), K)
    w = Δw / sum(Δw)

    dists = [Rayleigh(d) for d in D]
    Δr ~ filldist(MixtureModel(dists, w), N)
    return (; D, w)
end



#%%

N_samples = 1000
N_threads = 4

data = get_data(; fraction = 0.7);

function get_chain(model::Turing.Model, N_samples = 1000, N_threads = 1)
    chain_org = sample(model, NUTS(0.65), MCMCThreads(), N_samples, N_threads)
    return merge_variables_into_chain(model, chain_org)
end


chn_2D = get_chain(diffusion_ordered(data), N_samples, N_threads)
chn_2D_dirichlet = get_chain(diffusion_ordered_dirichlet(data), N_samples, N_threads)
chn_2D_Δw = get_chain(diffusion_ordered_Δw(data), N_samples, N_threads)

plot(extract_variables(chn_2D, [:D, :w]))
savefig("MWE_plot_ordered.png")
plot(extract_variables(chn_2D_dirichlet, [:D, :w]))
savefig("MWE_plot_ordered_dirichlet.png")
plot(extract_variables(chn_2D_Δw, [:D, :w]))
savefig("MWE_plot_ordered_Δw.png")


#%%

# loglikelihoods = pointwise_loglikelihoods(diffusion_ordered(data), Turing.MCMCChains.get_sections(chn_ordered, :parameters))

# param_mod_predict = diffusion_ordered(similar(y, Missing))
# ynames = string.(keys(loglikelihoods))
# loglikelihoods_vals = getindex.(Ref(loglikelihoods), ynames)
# # Reshape into `(nchains, nsamples, size(y)...)`
# loglikelihoods_arr = permutedims(cat(loglikelihoods_vals...; dims=3), (2, 1, 3))

data_3D = get_data_3D(; μ₁ = 1, μ₂ = 4, μ₃ = 8, fraction1 = 0.6, fraction2 = 0.3);

chn_3D = get_chain(diffusion_ordered(data_3D), N_samples, N_threads)
chn_3D_dirichlet = get_chain(diffusion_ordered_dirichlet(data_3D, 3), N_samples, N_threads)
chn_3D_Δw = get_chain(diffusion_ordered_Δw(data_3D, 3), N_samples, N_threads)


plot(extract_variables(chn_3D, [:D, :w]))
plot(extract_variables(chn_3D_dirichlet, [:D, :w]))
plot(extract_variables(chn_3D_Δw, [:D, :w]))
