using ColorSchemes



function D2σ²(D)
    return sqrt(2 * D * TAU)
end


function get_unique_name(
    N_samples::Int,
    model_name::String,
    type::String,
    N_threads::Int = 1,
)
    return "$(type)__$(model_name)__$(N_samples)__samples__$(N_threads)__threads"
end


function get_chains(
    model::AbstractMCMC.AbstractModel,
    alg::Turing.Inference.InferenceAlgorithm,
    N_samples::Int,
    model_name::String,
    N_threads::Int = 1;
    forced::Bool = false,
)

    filename = "chains/$model_name.h5"

    if isfile(filename) && !forced
        println("Loading $filename")
        chains = h5open(filename, "r") do f
            read(f, Chains)
        end
        return sort(chains)

    else
        println("Running Bayesian Analysis, please wait.")
        Random.seed!(1)
        logger = Logging.SimpleLogger(Logging.Error)
        chains = Logging.with_logger(logger) do
            if N_threads == 1
                sample(model, alg, N_samples)
            else
                sample(model, alg, MCMCThreads(), N_samples, N_threads)
            end
        end

        println("Saving $filename")
        h5open(filename, "w") do f
            write(f, chains)
        end

        return sort(chains)

    end

end

function get_chains(
    model::AbstractMCMC.AbstractModel,
    N_samples::Int,
    model_name::String,
    N_threads::Int = 1;
    forced::Bool = false,
)
    return get_chains(model, NUTS(0.65), N_samples, model_name, N_threads; forced = forced)
end


# function get_chains(model::DynamicPPL.Model, N_samples, N_threads, model_name, type; forced = false)
# end



function get_log_likelihoods_1D(Δr, diffusion_df_1D)
    pdfs = @. Rayleigh(D2σ²(diffusion_df_1D.D))
    llhs = zeros((length(Δr), length(diffusion_df_1D.D)))
    for (j, pdf) in enumerate(pdfs)
        llhs[:, j] = logpdf.(pdf, Δr)
    end
    return llhs
end



function get_log_likelihoods_2D(Δr, diffusion_df_2D)
    N = length(Δr)
    M = size(diffusion_df_2D, 1)
    llhs = zeros((N, M))
    for j = 1:M
        row = diffusion_df_2D[1, :]
        D = [row.D1, row.D2]
        w = [row.Θ1, 1 - row.Θ1]
        pdf = MixtureModel(Rayleigh, [D2σ²(d) for d in D], w)
        llhs[:, j] = logpdf.(pdf, Δr)
    end
    return llhs
end

function get_log_likelihoods_3D(Δr, diffusion_df_3D)
    N = length(Δr)
    M = size(diffusion_df_3D, 1)
    llhs = zeros((N, M))
    for j = 1:M
        row = diffusion_df_3D[1, :]
        D = [row.D1, row.D2, row.D3]
        w = [row.Θ1, row.Θ2, 1 - row.Θ1 - row.Θ2]
        pdf = MixtureModel(Rayleigh, [D2σ²(d) for d in D], w)
        llhs[:, j] = logpdf.(pdf, Δr)
    end
    return llhs
end



function compare_models_WAIC(llhs_x, llhs_y)

    waic_x = waic(llhs_x'; pointwise = false)[1]
    waic_y = waic(llhs_y'; pointwise = false)[1]

    waic_i_x = waic(llhs_x'; pointwise = true)[1]
    waic_i_y = waic(llhs_y'; pointwise = true)[1]

    N = length(waic_i_x)
    Δ_se = sqrt(N * var(waic_i_x - waic_i_y))
    Δ_waic = waic_x - waic_y
    z = Δ_waic / Δ_se
    return (; waic_x, waic_y, Δ_waic, Δ_se, z)
end

function compute_lppd(llhs)
    return waic(llhs'; pointwise = false).lppd
end



function PSIS(llhs)
    loo, loo_i, pk = psisloo(llhs')
    elpd_loo = sum(loo_i)
    se_elpd_loo = std(loo_i) * sqrt(length(loo_i))
    # pk_qualify(pk)
    return (; elpd_loo, se_elpd_loo, loo_i)
end

function _compare_loo_is(loo_i_x, loo_i_y)
    Δloo_i = loo_i_y - loo_i_x
    Δelpd = sum(Δloo_i)
    Δse_elpd = std(Δloo_i) * sqrt(length(Δloo_i))
    z = Δelpd / Δse_elpd
    return (; Δelpd, Δse_elpd, z)
end

function model_comparison(llhs, names::Vector{String})

    elpds = Float64[]
    se_elpds = Float64[]
    loo_is = Vector[]
    lppds = Float64[]
    for (llh, name) in zip(llhs, names)
        elpd_loo, se_elpd_loo, loo_i = PSIS(llh)
        push!(elpds, elpd_loo)
        push!(se_elpds, se_elpd_loo)
        push!(loo_is, loo_i)
        lppd = compute_lppd(llh)
        push!(lppds, lppd)
    end

    ordered = sortperm(elpds, rev = true)

    Δelpds = Float64[]
    Δse_elpds = Float64[]
    zs = Float64[]
    for i in ordered
        Δelpd, Δse_elpd, z = _compare_loo_is(loo_is[i], loo_is[ordered[1]])
        push!(Δelpds, Δelpd)
        push!(Δse_elpds, Δse_elpd)
        push!(zs, z)
    end

    df_comparison = DataFrame(
        name = names[ordered],
        lppd = lppds[ordered],
        elpd = elpds[ordered],
        se_elpd = se_elpds[ordered],
        Δelpd = Δelpds,
        Δse_elpd = Δse_elpds,
        z = zs,
    )
    df_comparison[1, :Δse_elpd] = NaN

    return df_comparison
end



function plot_model_comparison(df_comparison)

    ys = size(df_comparison, 1):-1:1


    f = Figure(resolution = (1000, 300))
    ax = Axis(
        f[1, 1],
        xlabel = "Log Score",
        # ylabel = "Model",
        limits = (nothing, nothing, minimum(ys) - 0.3, maximum(ys) + 0.3),
        title = "Model Comparison",
        yticks = (ys, df_comparison.name),
    )


    errorbars!(
        ax,
        df_comparison.elpd,
        ys,
        df_comparison.se_elpd,
        color = :black,
        whiskerwidth = 10,
        direction = :x,
    ) # same low and high error


    vlines!(
        ax,
        df_comparison.elpd[1],
        # linestyle = :dash,
        color = :black,
        linewidth = 0.75,
    )


    scatter!(ax, df_comparison.elpd, ys, markersize = 15, color = :black, marker = :circle)
    if "lppd" in names(df_comparison)
        scatter!(ax, df_comparison.lppd, ys, markersize = 15, color = :black, marker = :x)
    end

    mask_Δ = 2:size(df_comparison, 1)
    errorbars!(
        ax,
        df_comparison[mask_Δ, :elpd],
        ys[mask_Δ] .+ 0.3,
        df_comparison[mask_Δ, :Δse_elpd],
        color = :grey,
        whiskerwidth = 7,
        direction = :x,
    ) # same low and high error

    scatter!(
        ax,
        df_comparison[mask_Δ, :elpd],
        ys[mask_Δ] .+ 0.3,
        marker = :utriangle,
        markersize = 15,
        color = :grey,
    )

    for i in mask_Δ
        z = round(df_comparison[i, :z], digits = 2)
        x = df_comparison[i, :elpd]
        y = ys[i] + 0.5
        text!(
            "z = $z",
            position = (x, y),
            align = (:center, :center),
            color = :grey,
            textsize = 16,
        )
    end
    return f

end


#%%


function get_colors()
    names = ["red", "blue", "green", "purple", "orange", "yellow", "brown", "pink", "grey"]
    colors = ColorSchemes.Set1_9
    # d_colors = Dict(names .=> colors)
    return colors
end

function plot_chains(chns, resolution = (1_000, 1200))

    params = names(chns, :parameters)

    colors = get_colors()
    n_chains = length(chains(chns))
    n_samples = length(chns)

    fig = Figure(; resolution = resolution)

    # left part of the plot // traces
    for (i, param) in enumerate(params)
        ax = Axis(fig[i, 1]; ylabel = string(param))
        for chain = 1:n_chains
            values = chns[:, param, chain]
            lines!(ax, 1:n_samples, values; label = string(chain))
        end

        if i == length(params)
            ax.xlabel = "Iteration"
        end

    end

    # right part of the plot // density
    for (i, param) in enumerate(params)
        ax =
            Axis(fig[i, 2]; ylabel = string(param), limits = (nothing, nothing, 0, nothing))
        for chain = 1:n_chains
            values = chns[:, param, chain]
            density!(
                ax,
                values;
                label = string(chain),
                strokewidth = 3,
                strokecolor = (colors[chain], 0.8),
                color = (colors[chain], 0),
            )
        end

        hideydecorations!(ax, grid = false)
        ax.title = string(param)
        if i == length(params)
            ax.xlabel = "Parameter estimate"
        end
    end

    return fig

end



function get_df_C_Δr(df_Δ, m13_1_ch1)

    idx = df_Δ.idx
    Δr = df_Δ.idx.Δr

    N_groups = length(levels(idx))
    i_groups = 1:N_groups
    Δr_mean = Float64[]
    Δr_std = Float64[]
    Δr_sdom = Float64[]
    D_mean = Float64[]
    D_std = Float64[]
    D_sdom = Float64[]
    for i_group in i_groups
        Δr_group = Δr[i_group.==idx]
        push!(Δr_mean, mean(Δr_group))
        push!(Δr_std, std(Δr_group))
        push!(Δr_sdom, sdom(Δr_group))

        D_group = m13_1_ch1[:, i_group, 1]
        push!(D_mean, mean(D_group))
        push!(D_std, std(D_group))
        push!(D_sdom, sdom(D_group))
    end

    df_C_Δr = DataFrame((; i_groups, Δr_mean, Δr_std, Δr_sdom, D_mean, D_std, D_sdom))
    return df_C_Δr
end



function plot_df_C_Δr(df_C_Δr)

    f = Figure()
    ax = Axis(
        f[1, 1],
        xlabel = "Δr",
        ylabel = "D",
        title = input_dirs[type],
        # limits = (0, nothing, 0, nothing),
    )

    errorbars!(ax, df_C_Δr.Δr_mean, df_C_Δr.D_mean, df_C_Δr.D_std, whiskerwidth = 10)
    # errorbars!(ax, df_C_Δr.Δr_mean, df_C_Δr.D_mean, df_C_Δr.D_sdom, whiskerwidth = 10)
    scatter!(ax, df_C_Δr.Δr_mean, df_C_Δr.D_mean, markersize = 7, color = :black)

    # errorbars!(
    #     ax,
    #     df_C_Δr.Δr_mean,
    #     df_C_Δr.D_mean,
    #     df_C_Δr.Δr_sdom,
    #     whiskerwidth = 10,
    #     direction = :x,
    # )

    return f
end


#%%


function get_log_likelihoods(df_Δ, df_model, f_loglik, model_name; forced)

    filename = "likelihoods/$model_name.h5"

    if isfile(filename) && !forced
        println("Loading $filename")
        llhs = h5open(filename, "r") do f
            read(f, "llhs")
        end
        return llhs

    else
        println("Computing likelihoods, please wait.")
        llhs = f_loglik(df_Δ, df_model)
        println("Saving $filename")
        h5open(filename, "w") do f
            write(f, "llhs", llhs)
        end

        return llhs

    end

end


function compute_log_likelihoods_independent(df_Δ, df_independent)
    N_groups = length(levels(df_Δ.idx))
    N = nrow(df_Δ)
    M = nrow(df_independent)
    llhs = zeros((N, M))
    for i = 1:N
        group = df_Δ.idx[i]
        D_str = "D[$(group)]"
        for j = 1:M
            D = df_independent[j, D_str]
            pdf = Rayleigh(D2σ²(D))
            llhs[i, j] = logpdf(pdf, df_Δ[i, :Δr])
        end
    end
    return llhs
end


function get_log_likelihoods_independent(df_Δ, df_independent, model_name; forced = false)
    return get_log_likelihoods(
        df_Δ,
        df_independent,
        compute_log_likelihoods_independent,
        model_name;
        forced = forced,
    )
end





function compute_log_likelihoods_continuous(df_Δ, df_continuous)
    N_groups = length(levels(df_Δ.idx))
    N = nrow(df_Δ)
    M = nrow(df_continuous)
    llhs = zeros((N, M))
    for i = 1:N
        group = df_Δ.idx[i]
        θ_str = "θ[$(group)]"
        for j = 1:M
            θ1 = df_continuous[j, θ_str]
            w = [θ1, 1 - θ1]
            D = df_continuous[j, ["D[1]", "D[2]"]]
            dists = [Rayleigh(D2σ²(d)) for d in D]
            pdf = MixtureModel(dists, w)
            llhs[i, j] = logpdf(pdf, df_Δ[i, :Δr])
        end
    end
    return llhs
end


function get_log_likelihoods_continuous(df_Δ, df_independent, model_name; forced = false)
    return get_log_likelihoods(
        df_Δ,
        df_independent,
        compute_log_likelihoods_continuous,
        model_name;
        forced = forced,
    )
end



function compute_log_likelihoods_discrete(df_Δ, df_discrete)
    N_groups = length(levels(df_Δ.idx))
    N = nrow(df_Δ)
    M = nrow(df_discrete)
    llhs = zeros((N, M))
    for i = 1:N
        group = df_Δ.idx[i]
        Z_str = "Z[$(group)]"
        for j = 1:M
            z = Int(df_discrete[j, Z_str])
            D = df_discrete[j, "D[$(z)]"]
            pdf = Rayleigh(D2σ²(D))
            llhs[i, j] = logpdf(pdf, df_Δ[i, :Δr])
        end
    end
    return llhs
end


function get_log_likelihoods_discrete(df_Δ, df_independent, model_name; forced = false)
    return get_log_likelihoods(
        df_Δ,
        df_independent,
        compute_log_likelihoods_discrete,
        model_name;
        forced = forced,
    )
end



#%%

function get_variable_result(gen_quants, variable, N_samples, K, N_threads)
    variable_result = zeros(N_samples, K, N_threads)
    for thread = 1:N_threads
        for (i, xi) in enumerate(gen_quants[:, thread])
            variable_result[i, :, thread] = xi[variable]
        end
    end
    return variable_result
end


function get_variable_chain(gen_quants, variable, N_samples, K, N_threads)
    variable_result = get_variable_result(gen_quants, variable, N_samples, K, N_threads)
    variable_names = [Symbol("$variable[$i]") for i = 1:K]
    variable_chn = Chains(variable_result, variable_names)
    return variable_chn
end

function get_variable_chain_merged(gen_quants, variables, N_samples, K, N_threads)
    return hcat(
        [
            get_variable_chain(gen_quants, variable, N_samples, K, N_threads) for
            variable in variables
        ]...,
    )
end


function merge_variables_into_chain(model::Turing.Model, chain_in::Chains)

    chains_params = Turing.MCMCChains.get_sections(chain_in, :parameters)
    gen_quants = generated_quantities(model, chains_params)

    first_element = first(gen_quants)
    variables = keys(first_element)
    K = length(first_element[variables[1]]) # dimension

    chns_generated =
        get_variable_chain_merged(gen_quants, variables, N_samples, K, N_threads)
    return hcat(chain_in, setrange(chns_generated, range(chain_in)))
end


function extract_variables(c::Chains, var::Union{Symbol, String})
    return sort(c[namesingroup(c, var)])
end

function extract_variables(c::Chains, vars::Union{Vector{Symbol}, Vector{String}})
    names = vcat([namesingroup(c, var) for var in vars]...)
    return sort(c[names])
end
