using DataFrames
using CSV
using NaturalSort
using CairoMakie
# using GLMakie
using Optim

make_figure = true
# if make_figure
#     GLMakie.activate!()
# end

include("utils.jl")

input_dir = "../RealCleandFinalData/Sir3-Halo-Sir2DSir4D_Judith_Cleaned"
dfs = get_dfs(input_dir)
Δr = df2dist(dfs)

#%%


df_rows = get_df_rows(dfs)
cell = df_rows[1, :cell]
id = df_rows[1, :id]
df = get_df_by_cell_id(df_rows, cell, id)

# df_cell_12 = subset(dfs, :cell => ByRow(==(12)))
# # df = subset(dfs, :cell => ByRow(==(cell)), :id => ByRow(==(id)))

# for id in unique(df_cell_12.id)
#     df_cell_12_id = subset(df_cell_12, :id => ByRow(==(id)))
#     f = Figure()
#     ax = Axis(
#         f[1, 1],
#         xlabel = "steps",
#         ylabel = "Δr",
#         limits = (0, nothing, 0, nothing),
#         title = "Cell $(cell), ID $(id)",
#     )
#     scatter!(ax, df2dist(df_cell_12_id))
#     display(f)
# end


if make_figure
    f = Figure()
    ax = Axis(
        f[1, 1],
        xlabel = "steps",
        ylabel = "Δr",
        limits = (0, nothing, 0, nothing),
        title = "Cell $(cell), ID $(id)",
    )
    scatter!(ax, df2dist(df))
    display(f)
    f
end
save("single_cell_id.pdf", f)



#%%

if make_figure
    f = Figure()
    ax = Axis(
        f[1, 1],
        xlabel = "steps",
        ylabel = "Δr",
        limits = (0, nothing, 0, nothing),
        title = "Cell $(cell), ID $(id)",
    )
    scatter!(ax, df2dist(df))
    display(f)
    f
end
save("single_cell_id.pdf", f)

#%%

if make_figure
    f = Figure()
    ax = Axis(f[1, 1], xlabel = "Δr", ylabel = "Counts", limits = (0, nothing, 0, nothing))
    hist!(ax, Δr, bins = 100, strokewidth = 1, normalization = :pdf)
    display(f)
    f
end

#%%

x = x


#%%


p0 = [0.4]
res1 = fit(Δr, loglikelihood1, p0)
# res1 = fit(Δr, loglikelihood1, p0; lower = [0], upper = [Inf])
# res1 = fit(Δr, loglikelihood1, p0; method = LBFGS())

#%%

if make_figure
    plot!(ax, likelihood1, res1)
    display(f)
    f
end

#%%

p0 = [0.15, 0.87, 0.6]
res2 = fit(Δr, loglikelihood2, p0)
# res2 = fit(Δr, loglikelihood2, p0; lower = [0, 0, 0], upper = [Inf, Inf, 1])
# res2 = fit(Δr, loglikelihood2, p0; method = LBFGS())


#%%

if make_figure
    plot!(ax, likelihood2, res2)
    display(f)
    f
end


#%%


p0 = [0.1, 0.4, 1.5, 0.45, 0.45]
res3 = fit(Δr, loglikelihood3, p0)
res3 = fit(Δr, loglikelihood3, p0; lower = [0, 0, 0, 0, 0], upper = [Inf, Inf, Inf, 1, 1])
# res3 = fit(Δr, loglikelihood3, p0; method = LBFGS())


if make_figure
    plot!(ax, likelihood3, res3)
    display(f)
    f
end
