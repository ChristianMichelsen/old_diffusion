
using StatsBase
using LinearAlgebra: diag
using NLSolversBase


#%%

const TAU = 0.02

input_dirs = Dict(
    "WT" => "Sir3-Halo-WT_Single_Molecules_Data_Set_04_Judith_Cleaned",
    "Sir2DSir4D" => "Sir3-Halo-Sir2DSir4D_Judith_Cleaned",
)


function get_files(input_dir)
    return sort(
        filter(x -> occursin("Cell", x), readdir(input_dir, join = true)),
        lt = natural,
    )
end

function extract_cell_number(file)
    str = basename(file)
    return parse(Int, split(str, "Cell")[2])
end

function file2df(file)
    df = DataFrame(CSV.File(file; delim = "\t", header = [:id, :x, :y, :t]))
    df[!, :cell] .= extract_cell_number(file)
    return df
end

function get_dfs(input_dir)
    files = get_files(input_dir)
    dfs = DataFrame[]
    for file in files
        df = file2df(file)
        push!(dfs, df)
    end
    return vcat(dfs...)
end


function group2dist(g::SubDataFrame)
    xy = Matrix(g[!, [:x, :y]])
    return sqrt.(sum((xy[1:end-1, :] .- xy[2:end, :]) .^ 2, dims = 2))[:, 1]
end

function df2dist(df::DataFrame)
    gd = groupby(df, [:cell, :id], sort = false)
    return reduce(vcat, [group2dist(g) for g in gd])
end



# transform!(groupby(sales_df, [:year, :region]), eachindex => :store_id)

function get_df_Δ(df::DataFrame)
    gd = groupby(df, [:cell, :id], sort = false)

    Δr = reduce(vcat, [group2dist(g) for g in gd])
    idx = reduce(vcat, [fill(i, nrow(g) - 1) for (i, g) in enumerate(gd)])
    id = reduce(vcat, [g[2:end, :id] for g in gd])
    cell = reduce(vcat, [g[2:end, :cell] for g in gd])
    return DataFrame((; Δr, idx, id, cell))
end


#%%

function get_df_rows(dfs)
    gd = groupby(dfs, [:cell, :id], sort = false)
    data = [(g[1, :cell], g[1, :id], nrow(g)) for g in gd]
    df_rows = DataFrame((; cell, id, L) for (cell, id, L) in data)
    sort!(df_rows, :L, rev = true)
    return df_rows
end

function get_df_by_cell_id(df, cell, id)
    df = subset(dfs, :cell => ByRow(==(cell)), :id => ByRow(==(id)))
    return df
end



function get_df_Δ_long_runs(dfs::DataFrame, df_Δ::DataFrame; min_L::Int = 10)

    df_by_group = get_df_rows(dfs)
    df_long_runs = df_by_group[df_by_group.L.>min_L, :]

    df_Δ_long_runs = vcat(
        [
            get_df_by_cell_id(df_Δ, cell, id) for
            (cell, id) in zip(df_long_runs.cell, df_long_runs.id)
        ]...,
    )

    return get_df_Δ(df_Δ_long_runs)
end


function get_df_Δ_long_runs(dfs::DataFrame; min_L::Int = 10)
    df_Δ = get_df_Δ(dfs)
    return get_df_Δ_long_runs(dfs, df_Δ, min_L = min_L)
end


#%%

function plot!(ax, likelihood, res::Optim.OptimizationResults)
    xx = range(0, 1, length = 1000 + 1)
    yy = likelihood.(xx, Optim.minimizer(res)...)
    lines!(ax, xx, yy)
end


function plot!(ax, likelihood, res::NamedTuple)
    xx = range(0, 1, length = 1000 + 1)
    yy = likelihood.(xx, Optim.minimizer(res.res)...)
    lines!(ax, xx, yy)
end


#%%

# function likelihood1(Δr, σ)
#     return @. Δr / σ^2 * exp(-Δr^2 / (2 * σ^2))
# end

function likelihood1(Δr, D)
    return @. Δr / (2 * D * TAU) * exp(-Δr^2 / (4 * D * TAU))
end

function loglikelihood1(Δr, σ)
    return sum(log.(likelihood1(Δr, σ)))
end


#%%


function likelihood2(Δr, σ1, σ2, f)
    return f * likelihood1(Δr, σ1) + (1 - f) * likelihood1(Δr, σ2)
end


function loglikelihood2(Δr, σ1, σ2, f)
    return sum(log.(likelihood2(Δr, σ1, σ2, f)))
end


#%%


function StatsBase.cov2cor(X::Matrix)
    return cov2cor(X, sqrt.(diag(X)))
end

#%%


#%%




function fit(Δr, llh, p0; lower = nothing, upper = nothing, method = nothing)
    func = TwiceDifferentiable(p -> -llh(Δr, p...), p0; autodiff = :forward)
    if isa(lower, Nothing) && isa(upper, Nothing) && isa(method, Nothing)
        res = optimize(func, p0)
    elseif isa(lower, Nothing) && isa(upper, Nothing) && !isa(method, Nothing)
        res = optimize(func, p0, method)
    elseif !isa(lower, Nothing) && !isa(upper, Nothing) && isa(method, Nothing)
        res = optimize(func, lower, upper, p0)
    else
        @show lower, upper, method
        throw(ArgumentError("Cannot set both lower/upper and method."))
    end

    return get_fit_results(res, func)
end

function get_fit_results(res, func)
    parameters = Optim.minimizer(res)
    numerical_hessian = hessian!(func, parameters)
    var_cov_matrix = inv(numerical_hessian)
    return (
        res = res,
        μ = parameters,
        σ = sqrt.(diag(var_cov_matrix)),
        Σ = var_cov_matrix,
        ρ = cov2cor(var_cov_matrix),
    )
end



τ = 0.02
function σ2D(σ)
    return σ^2 / (2 * τ)
end

function D2σ(D)
    sqrt(2 * D * τ)
end


#%%

function likelihood3(Δr, σ1, σ2, σ3, f1, f2)
    return f1 * likelihood1(Δr, σ1) +
           f2 * likelihood1(Δr, σ2) +
           (1 - f1 - f2) * likelihood1(Δr, σ3)
end


function loglikelihood3(Δr, σ1, σ2, σ3, f1, f2)
    return sum(log.(likelihood3(Δr, σ1, σ2, σ3, f1, f2)))
end


function plot_color_groups(df_Δ)
    color = mod.(df_Δ.idx, 7)

    f = Figure(resolution = (2000, 500))
    ax = Axis(
        f[1, 1],
        xlabel = "Index",
        ylabel = "Δr",
        title = input_dirs[type],
        limits = (0, nothing, 0, nothing),
    )
    xs = 1:nrow(df_Δ)
    scatter!(ax, df_Δ.Δr, color = color, colormap = :darktest)
    return f
end



function sdom(x)
    return std(x) / sqrt(length(x))
end
