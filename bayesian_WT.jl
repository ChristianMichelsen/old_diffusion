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
using Random

#%%

include("utils.jl")
include("bayesian_utils.jl")

#%%

# type = "Sir2DSir4D"
type = "WT"

N_samples = 1_000;
N_threads = 4
min_L = 10
# min_L = 100
savefig = false
# savefig = true

#%%

input_dir = "../RealCleandFinalData/$(input_dirs[type])"
dfs = get_dfs(input_dir)
df_Δ = get_df_Δ_long_runs(dfs, min_L = min_L) # get_df_Δ(dfs)
N_groups = length(levels(df_Δ.idx))


#%%

f_color_groups = plot_color_groups(df_Δ)

# x = x


#%%

@model function diffusion_independent(Δr, idx)
    N_groups = length(levels(idx))

    # prior
    D ~ filldist(truncated(Exponential(1), 0.0001, 10), N_groups)

    # likelihood
    for i_group = 1:N_groups
        group_mask = i_group .== idx
        Δr[group_mask] ~ filldist(Rayleigh(D2σ²(D[i_group])), sum(group_mask))
    end

end


name_independent =
    get_unique_name(N_samples, "independent__$(min_L)__min_L", type, N_threads)
println("Running $(N_groups) coefficients, independent")

model_independent = diffusion_independent(df_Δ.Δr, df_Δ.idx);
chains_independent = get_chains(model_independent, N_samples, name_independent, N_threads)
df_independent = DataFrame(chains_independent);
# plot_chains(chains_independent[:, 1:10, :], (1000, 2000))
# optimize(model_independent, MAP())
llhs_independent = get_log_likelihoods_independent(df_Δ, df_independent, name_independent)
compute_lppd(llhs_independent)
elpd_loo_independent, loo_i_independent, pk_independent = psisloo(llhs_independent');
se_elpd_loo_independent = std(loo_i_independent) * sqrt(length(loo_i_independent))
pk_qualify(pk_independent)


#%%

@model function diffusion_continuous(Δr, idx)
    N_groups = length(levels(idx))

    D ~ arraydist([
        truncated(Exponential(0.1), 0.0001, 0.075),
        truncated(Exponential(0.1), 0.075, 1),
    ],)

    dists = [Rayleigh(D2σ²(d)) for d in D]
    θ ~ filldist(Uniform(0, 1), N_groups)
    # θ ~ filldist(Beta(0.5, 0.5), N_groups)

    for i_group = 1:N_groups
        group_mask = i_group .== idx
        Θ1 = θ[i_group]
        w = [Θ1, 1 - Θ1]
        Δr[group_mask] ~ filldist(MixtureModel(dists, w), sum(group_mask))
    end
end

name_continuous = get_unique_name(N_samples, "continuous__$(min_L)__min_L", type, N_threads)
println("2 coefficients, continuous")

model_continuous = diffusion_continuous(df_Δ.Δr, df_Δ.idx);
chains_continuous = get_chains(model_continuous, N_samples, name_continuous, N_threads)
df_continuous = DataFrame(chains_continuous);
plot_chains(chains_continuous[names(chains_continuous, :parameters)[1:12]], (1000, 2000))

llhs_continuous = get_log_likelihoods_continuous(df_Δ, df_continuous, name_continuous);
compute_lppd(llhs_continuous)
elpd_loo_continuous, loo_i_continuous, pk_continuous = psisloo(llhs_continuous');
se_elpd_loo_continuous = std(loo_i_continuous) * sqrt(length(loo_i_continuous))
pk_qualify(pk_continuous)


#%%

@model function diffusion_continuous_ordered(Δr, idx)
    N_groups = length(levels(idx))

    ΔD ~ filldist(Exponential(0.1), 2)
    D = cumsum(ΔD)

    dists = [Rayleigh(D2σ²(d)) for d in D]
    θ ~ filldist(Uniform(0, 1), N_groups)

    for i_group = 1:N_groups
        group_mask = i_group .== idx
        Θ1 = θ[i_group]
        w = [Θ1, 1 - Θ1]
        Δr[group_mask] ~ filldist(MixtureModel(dists, w), sum(group_mask))
    end
    return (; D)
end

# N_threads = 1
name_continuous_ordered =
    get_unique_name(N_samples, "continuous_ordered__$(min_L)__min_L", type, N_threads)
println("2 coefficients, continuous ordered")

model_continuous_ordered = diffusion_continuous_ordered(df_Δ.Δr, df_Δ.idx);
chains_continuous_ordered =
    get_chains(model_continuous_ordered, N_samples, name_continuous_ordered, N_threads)

chains_continuous_ordered =
    merge_variables_into_chain(model_continuous_ordered, chains_continuous_ordered)
df_continuous_ordered = DataFrame(chains_continuous_ordered);


plot_chains(
    extract_variables(chains_continuous_ordered, [:D, :θ])[:, 1:12, :],
    (1000, 2000),
)

llhs_continuous_ordered =
    get_log_likelihoods_continuous(df_Δ, df_continuous_ordered, name_continuous_ordered);
compute_lppd(llhs_continuous_ordered)
elpd_loo_continuous_ordered, loo_i_continuous_ordered, pk_continuous_ordered =
    psisloo(llhs_continuous_ordered');
se_elpd_loo_continuous_ordered =
    std(loo_i_continuous_ordered) * sqrt(length(loo_i_continuous_ordered))
pk_qualify(pk_continuous_ordered)


#%%


@model function diffusion_discrete(Δr, idx)
    N_groups = length(levels(idx))
    K = 2

    D ~ arraydist([
        truncated(Exponential(0.1), 0.04, 0.06),
        truncated(Exponential(0.1), 0.01, 0.2),
    ],)

    Z = Vector{Int}(undef, N_groups)
    for i_group = 1:N_groups
        group_mask = i_group .== idx
        Z[i_group] ~ Categorical(K)
        σ² = D2σ²(D[Z[i_group]])
        Δr[group_mask] ~ filldist(Rayleigh(σ²), sum(group_mask))
    end
end

name_discrete = get_unique_name(N_samples, "discrete__$(min_L)__min_L", type, N_threads)
println("2 coefficients, discrete")

model_discrete = diffusion_discrete(df_Δ.Δr, df_Δ.idx);

chains_discrete = get_chains(
    model_discrete,
    Gibbs(HMC(0.01, 50, :D), PG(120, :Z)),
    N_samples,
    name_discrete,
    N_threads,
)


plot_chains(chains_discrete[:, 1:12, :], (1000, 2000))
plot_chains(chains_discrete[100:end, 1:12, :], (1000, 2000))
df_discrete = DataFrame(chains_discrete[100:end, :, :]);
llhs_discrete = get_log_likelihoods_discrete(df_Δ, df_discrete, name_discrete)
compute_lppd(llhs_discrete)
elpd_loo_discrete, loo_i_discrete, pk_discrete = psisloo(llhs_discrete');
se_elpd_loo_discrete = std(loo_i_discrete) * sqrt(length(loo_i_discrete))
pk_qualify(pk_discrete)


#%%


df_comparison = model_comparison(
    [
        llhs_independent,
        llhs_continuous,
        llhs_continuous_ordered,
        llhs_discrete, #
    ],
    [
        "$(N_groups) coefficients, independent",
        "2 coefficients, continuous",
        "2 coefficients, continuous ordered",
        "2 coefficients, discrete",
    ],
    #
)


f_model_comparison = plot_model_comparison(df_comparison)
if savefig
    save("figures/$(type)__model_comparison__min_L__$(min_L).pdf", f_model_comparison)
end

#%%

Ds = extract_variables(chains_continuous_ordered, :D)
θs = extract_variables(chains_continuous_ordered, :θ)

mask = DataFrame(mean(θs)).mean .> 0.8
slow_groups = (1:N_groups)[mask]

df_Δ_slow_groups = df_Δ[df_Δ.idx.∈Ref(slow_groups), :]

#%%

loglikelihoods = pointwise_loglikelihoods(
    model_continuous_ordered,
    Turing.MCMCChains.get_sections(chains_continuous_ordered, :parameters),
)

param_mod_predict =
    diffusion_continuous_ordered(similar(df_Δ.Δr, Missing), similar(df_Δ.idx, Missing),)
ynames = string.(keys(loglikelihoods))
loglikelihoods_vals = getindex.(Ref(loglikelihoods), ynames)
# Reshape into `(nchains, nsamples, size(y)...)`
loglikelihoods_arr = permutedims(cat(loglikelihoods_vals...; dims = 3), (2, 1, 3))
