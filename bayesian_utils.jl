using ColorSchemes



function D2σ²(D)
    return sqrt(2 * D * TAU)
end



function get_chains(model, N_samples, N_threads, model_name, type; forced = false)

    filename = "chains/$(type)__$(model_name)__$(N_samples)__samples__$(N_threads)__threads.h5"

    if isfile(filename) && !forced
        chain = h5open(filename, "r") do f
            read(f, Chains)
        end
        return chain

    else

        chains = Logging.with_logger(logger) do
            sample(model, NUTS(0.65), MCMCThreads(), N_samples, N_threads)
        end

        h5open(filename, "w") do f
            write(f, chains)
        end

        return chains

    end

end



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
    for (llh, name) in zip(llhs, names)
        elpd_loo, se_elpd_loo, loo_i = PSIS(llh)
        push!(elpds, elpd_loo)
        push!(se_elpds, se_elpd_loo)
        push!(loo_is, loo_i)
    end

    ordered = sortperm(elpds, rev = true)

    names[ordered]
    elpds[ordered]
    se_elpds[ordered]

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

    scatter!(ax, df_comparison.elpd, ys, markersize = 10, color = :black)

    vlines!(
        ax,
        df_comparison.elpd[1],
        # linestyle = :dash,
        color = :black,
        linewidth = 0.75,
    )



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

    return f

end


#%%


function get_colors()
    names = ["red", "blue", "green", "purple", "orange", "yellow", "brown", "pink", "grey"]
    colors = ColorSchemes.Set1_9
    # d_colors = Dict(names .=> colors)
    return colors
end

function plot_chains(chns, resolution=(1_000, 1200))

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
        ax = Axis(fig[i, 2]; ylabel = string(param), limits = (nothing, nothing, 0, nothing))
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
