using DataFrames
using CSV
using NaturalSort
using CairoMakie
# using GLMakie
using Optim
using StatsFuns
using ParetoSmoothedImportanceSampling
using Turing
using Distributions
import StatsPlots
import Logging
using MCMCChainsStorage
using HDF5
# using MCMCChains

# if make_figure
#     GLMakie.activate!()
# end

include("utils.jl")

input_dir = "../RealCleandFinalData/Sir3-Halo-Sir2DSir4D_Judith_Cleaned"
dfs = get_dfs(input_dir)
Δr = df2dist(dfs)

include("bayesian_utils.jl")

#%%

N_samples = 1_000;
N_threads = 4
logger = Logging.SimpleLogger(Logging.Error)
savefig = true

#%%


@model function diffusion_1D(Δr)
    N = length(Δr)
    # prior
    D ~ truncated(Exponential(1), 0.0001, 10)
    # likelihood
    Δr ~ filldist(Rayleigh(D2σ²(D)), N)
end
model_1D = diffusion_1D(Δr);
chains_1D = get_chains(model_1D, N_samples, N_threads, "1D")
df_1D = DataFrame(chains_1D);
map_1D = optimize(model_1D, MAP())
fig_1D = plot_chains(chains_1D, (1000, 400))
if savefig
    save("figures/chains_1D.pdf", fig_1D)
end

# %%


@model function diffusion_2D(Δr)
    N = length(Δr)
    # priors
    D1 ~ truncated(Exponential(1), 0, 0.5)
    D2 ~ truncated(Exponential(1), 0.5, 10)
    D = [D1, D2]
    Θ1 ~ Uniform(0, 1)
    w = [Θ1, 1 - Θ1]
    # likelihood
    Δr ~ filldist(MixtureModel([Rayleigh(D2σ²(d)) for d in D], w), N)
end;
model_2D = diffusion_2D(Δr);
chains_2D = get_chains(model_2D, N_samples, N_threads, "2D")
df_2D = DataFrame(chains_2D);
map_2D = optimize(model_2D, MAP())
fig_2D = plot_chains(chains_2D, (1000, 800))
if savefig
    save("figures/chains_2D.pdf", fig_2D)
end

# %%


@model function diffusion_3D(Δr)
    N = length(Δr)
    # priors
    D1 ~ truncated(Exponential(0.15), 0, 0.18)
    D2 ~ truncated(Exponential(0.5), 0.18, 0.8)
    D3 ~ truncated(Exponential(1.2), 0.8, 10)
    D = [D1, D2, D3]
    Θ1 ~ Uniform(0, 0.7)
    Θ2 ~ Uniform(0, 0.7)
    w = [Θ1, Θ2, 1 - Θ1 - Θ2]
    # make sure that Θ1 + Θ2 sum to less than 1
    if Θ1 + Θ2 >= 1
        Turing.@addlogprob! -Inf
        return
    end
    # likelihood
    Δr ~ filldist(MixtureModel([Rayleigh(D2σ²(d)) for d in D], w), N)
end;
model_3D = diffusion_3D(Δr);
chains_3D = get_chains(model_3D, N_samples, N_threads, "3D")
df_3D = DataFrame(chains_3D);
map_3D = optimize(model_3D, MAP())
fig_3D = plot_chains(chains_3D, (1000, 1200))
if savefig
    save("figures/chains_3D.pdf", fig_3D)
end

#%%

llhs_1D = get_log_likelihoods_1D(Δr, df_1D);
llhs_2D = get_log_likelihoods_2D(Δr, df_2D);
llhs_3D = get_log_likelihoods_3D(Δr, df_3D);

df_comparison = model_comparison(
    [llhs_1D, llhs_2D, llhs_3D],
    ["1D Model", "2D Model", "3D Model"],
    #
)


f_model_comparison = plot_model_comparison(df_comparison)
if savefig
    save("figures/model_comparison.pdf", f_model_comparison)
end


#%%

