"""
Murphy and Topel model with frailty and productive ageing
"""

if occursin("jashwin", pwd())
    cd("C://Users/jashwin/Documents/GitHub/international-gains-to-healthy-longevity/")
else
    cd("/Users/julianashwin/Documents/GitHub/international-gains-to-healthy-longevity/")
end

using Statistics, Parameters, DataFrames
using QuadGK, NLsolve, Roots, FiniteDifferences
using Plots, XLSX, ProgressMeter, Formatting, TableView, Latexify, LaTeXStrings, PlotlyJS
using VegaLite


# Import functions
include("src/TargetingAging.jl")

# Plotting backend
gr()

param = :none
HLE_bool = false
no_compress = false
age_eval = 0.0

# Biological parameters
bio_pars = BiologicalParameters()
# Economic model
econ_pars = EconomicParameters()

# Reset bio_pars.γ (exponent on frailty in mortality) to match starting LE
bio_pars.γ = find_zero(function (x) bio_pars.γ = x;
    bio_pars.LE_base - LE(bio_pars, (no_compress = false,age_start = 0, Wolverine = 0), aa = 0.)
	end, (1e-16, 1.))

# Specify options for scenario to run
opts = (param = param, bio_pars0 = deepcopy(bio_pars), LE_max = 40, step = 1, HLE = HLE_bool,
    Age_range = [0,20, 40, 60, 80], AgeGrad = 20, AgeRetire = 65,
	plot_shape_3d = false, zoom_pars = [0.5,0.9], redo_z0 = false, no_compress = no_compress,
	prod_age = false, age_start = 0, Wolverine = 0)

vars_df = define_vars_df(bio_pars, econ_pars, opts)
econ_pars.z0 = compute_z0(econ_pars, vars_df)

bio_pars0 = deepcopy(bio_pars)



"""
Struldbrugg elongation
"""
# Define relevant parameters and ranges
param1 = :M1
param2 = :D1
LE_range = Array{Float64,1}(60:1.0:100)
HLE_range = Array{Float64,1}(55:1.0:100)
# Define options
opts = (param = param1, bio_pars0 = deepcopy(bio_pars0), LE_max = 40, step = 1, HLE = HLE_bool,
    Age_range = [0,20, 40, 60, 80], AgeGrad = 20, AgeRetire = 65,
	plot_shape_3d = false, zoom_pars = [0.5,0.9], redo_z0 = false, no_compress = false,
	prod_age = false, age_start = 0, Wolverine = 0)
# Compute matrix
s_elg_wtp_df = struld_matrix(bio_pars, param1, param2, LE_range, HLE_range, opts, edge_adj = true)
CSV.write("figures/Olshansky_plots/s_elg_wtp_df.csv", s_elg_wtp_df)

s_elg_wtp_df = CSV.read("figures/Olshansky_plots/s_elg_wtp_df.csv", DataFrame)
s_elg_wtp_df.WTP_thsds = s_elg_wtp_df.WTP_1y./1000
s_elg_wtp_matrix = chunk(s_elg_wtp_df.WTP_thsds[1:(length(LE_range)*length(HLE_range))], length(LE_range))
s_elg_wtp_matrix = Matrix(transpose(hcat(s_elg_wtp_matrix...)))

# Reset bio_pars
bio_pars =deepcopy(opts.bio_pars0)
# Plot surface
pyplot()
plts = surface(s_elg_wtp_df.LE, s_elg_wtp_df.HLE, s_elg_wtp_df.WTP_thsds,
	camera=(45,45), xlabel= "LE", ylabel= "HLE", zlabel = "WTP",
	title="WTP for M1", c = :binary, margin=3Plots.mm)


# Heatmap as a check
heatmap(LE_range, HLE_range, s_elg_wtp_matrix, c =:coolwarm, margin=3Plots.mm,
	xlabel= "LE", ylabel= "HLE", subplot= 1,title="WTP")
savefig("figures/Olshansky_plots/WTP_M1_htmp.pdf")


plt_s = surface(s_elg_wtp_df.LE, s_elg_wtp_df.HLE, s_elg_wtp_df.WTP_thsds,
	camera=(45,25), xlabel= "LE", ylabel= "HLE", zlabel = "WTP",
	c = :tab10, margin=4Plots.mm, legend = false)


gr()
plt_s = surface(s_elg_wtp_df.LE, s_elg_wtp_df.HLE, s_elg_wtp_df.WTP_thsds,
	camera=(45,25), xlabel= "LE", ylabel= "HLE", zlabel = "WTP",
	c = :binary, margin=4Plots.mm, legend = false)
plt_s = plot!(size = (400,400))
savefig("figures/Olshansky_plots/WTP_struld.pdf")


# 2d contour plot
layout = Layout(xaxis_title="LE", yaxis_title="HLE")
p1 = PlotlyJS.plot(PlotlyJS.contour(
    x=LE_range, # horizontal axis
    y=HLE_range, # vertical axis
    z=s_elg_wtp_matrix,
	contours=attr( colorscale="Hot", showlabels = true, labelfont = attr(size = 12,color = "white")),
	colorbar=attr(title="WTP", titleside="right", titlefont=attr(size=14,family="Arial, sans-serif"))
	),layout)
PlotlyJS.savefig(p1, "figures/Olshansky_plots/WTP_struld_contour.pdf", width = 400, height = 400)





"""
Dorian Gray elongation
"""
# Define relevant parameters and ranges
param1 = :M1
param2 = :D1
LE_range = Array{Float64,1}(60:1.0:100)
HLE_range = Array{Float64,1}(55:1.0:100)
# Define options
opts = (param = param2, bio_pars0 = deepcopy(bio_pars0), LE_max = 40, step = 1, HLE = HLE_bool,
    Age_range = [0,20, 40, 60, 80], AgeGrad = 20, AgeRetire = 65,
	plot_shape_3d = false, zoom_pars = [0.5,0.9], redo_z0 = false, no_compress = false,
	prod_age = false, age_start = 0, Wolverine = 0)
# Compute matrix
d_elg_wtp_df = dorian_matrix(bio_pars, param1, param2, LE_range, HLE_range, opts, edge_adj = true)
CSV.write("figures/Olshansky_plots/d_elg_wtp_df.csv", d_elg_wtp_df)

d_elg_wtp_df = CSV.read("figures/Olshansky_plots/d_elg_wtp_df.csv", DataFrame)
d_elg_wtp_df = d_elg_wtp_df[d_elg_wtp_df.LE .<= 100,:]
d_elg_wtp_df = d_elg_wtp_df[d_elg_wtp_df.HLE .<= 100,:]
d_elg_wtp_df.WTP_thsds = d_elg_wtp_df.WTP_1y./1000
d_elg_wtp_matrix = chunk(d_elg_wtp_df.WTP_thsds[1:(length(LE_range)*length(HLE_range))], length(LE_range))
d_elg_wtp_matrix = Matrix(transpose(hcat(d_elg_wtp_matrix...)))

# Reset bio_pars
bio_pars =deepcopy(bio_pars0)
# Plot surface

plts = surface(d_elg_wtp_df.HLE, d_elg_wtp_df.LE, d_elg_wtp_df.WTP_1y,
	camera=(45,45), xlabel= "HLE", ylabel= "LE", zlabel = "WTP",
	title="WTP for D1", colorbar =:none, margin=3Plots.mm)
savefig("figures/Olshansky_plots/WTP_D1.pdf")
# Heatmap as a check
heatmap(LE_range, HLE_range, d_elg_wtp_matrix, c =:coolwarm, margin=3Plots.mm,
	xlabel= "LE", ylabel= "HLE", subplot= 1,title="WTP")
savefig("figures/Olshansky_plots/WTP_D1_htmp.pdf")

plt_d = surface(d_elg_wtp_df.HLE, d_elg_wtp_df.LE, d_elg_wtp_df.WTP_thsds,
	camera=(45,25), xlabel= "HLE", ylabel= "LE", zlabel = "WTP",
	c = :tab10, margin=4Plots.mm, legend = false)
plt_d = plot!(size = (400,400))
savefig("figures/Olshansky_plots/WTP_dorian.pdf")


# 2d contour plot
layout = Layout(xaxis_title="LE", yaxis_title="HLE")
p1 = PlotlyJS.plot(PlotlyJS.contour(
    x=LE_range, # horizontal axis
    y=HLE_range, # vertical axis
    z=d_elg_wtp_matrix,
	contours=attr( colorscale="Hot", showlabels = true, labelfont = attr(size = 12,color = "white")),
	colorbar=attr(title="WTP", titleside="right", titlefont=attr(size=14,family="Arial, sans-serif"))
	),layout)
PlotlyJS.savefig(p1, "figures/Olshansky_plots/WTP_dorian_contour.pdf", width = 400, height = 400)




"""
Struldbrugg rectangularisation
"""
# Define relevant parameters and ranges
param1 = :γ
param2 = :ψ
LE_range = Array{Float64,1}(70:1.0:97)
HLE_range = Array{Float64,1}(65:1.0:97)
# Define options
opts = (param = param1, bio_pars0 = deepcopy(bio_pars0), LE_max = 40, step = 1, HLE = HLE_bool,
    Age_range = [0,20, 40, 60, 80], AgeGrad = 20, AgeRetire = 65,
	plot_shape_3d = false, zoom_pars = [0.5,0.9], redo_z0 = false, no_compress = true,
	prod_age = false, age_start = 0, Wolverine = 0)
# Compute matrix
s_rect_wtp_df = struld_matrix(bio_pars, param1, param2, LE_range, HLE_range, opts)
CSV.write("figures/Olshansky_plots/s_rect_wtp_df.csv", s_rect_wtp_df)
# Reset bio_pars
bio_pars =deepcopy(opts.bio_pars0)
# Plot surface
plts = surface(s_rect_wtp_df.LE, s_rect_wtp_df.HLE, s_rect_wtp_df.WTP_1y,
	camera=(45,15), xlabel= "LE", ylabel= "HLE", zlabel = "WTP",
	title="WTP for γ", colorbar =:none, margin=3Plots.mm)
savefig("figures/Olshansky_plots/WTP_gamma.pdf")
# Heatmap for sense-check
s_rect_wtp_matrix = chunk(s_rect_wtp_df.WTP_1y[1:(length(LE_range)*length(HLE_range))], length(LE_range))
s_rect_wtp_matrix = Matrix(transpose(hcat(s_rect_wtp_matrix...)))
heatmap(LE_range, HLE_range, s_rect_wtp_matrix, c =:coolwarm, margin=3Plots.mm,
	xlabel= "LE", ylabel= "HLE", subplot= 1,title="WTP")
savefig("figures/Olshansky_plots/WTP_gamma_htmp.pdf")



"""
Dorian Gray rectangularisation
"""
# Define relevant parameters and ranges
param1 = :γ
param2 = :ψ
LE_range = Array{Float64,1}(70:1.0:97)
HLE_range = Array{Float64,1}(65:1.0:97)
# Define options
opts = (param = param2, bio_pars0 = deepcopy(bio_pars0), LE_max = 40, step = 1, HLE = HLE_bool,
    Age_range = [0,20, 40, 60, 80], AgeGrad = 20, AgeRetire = 65,
	plot_shape_3d = false, zoom_pars = [0.5,0.9], redo_z0 = false, no_compress = true,
	prod_age = false, age_start = 0, Wolverine = 0)
# Compute matrix
d_rect_wtp_df = dorian_matrix(bio_pars, param1, param2, LE_range, HLE_range, opts, edge_adj = false)
CSV.write("figures/Olshansky_plots/d_rect_wtp_df.csv", d_rect_wtp_df)
# Reset bio_pars
bio_pars =deepcopy(opts.bio_pars0)
# Plot surface
plts = surface(d_rect_wtp_df.HLE, d_rect_wtp_df.LE, d_rect_wtp_df.WTP_1y,
	camera=(45,15), xlabel= "HLE", ylabel= "LE", zlabel = "WTP",
	title="WTP for ψ", colorbar =:none, margin=3Plots.mm)
savefig("figures/Olshansky_plots/WTP_psi.pdf")
# Heatmap as a check
d_rect_wtp_matrix = chunk(d_rect_wtp_df.WTP_1y[1:(length(LE_range)*length(HLE_range))], length(LE_range))
d_rect_wtp_matrix = Matrix(transpose(hcat(d_rect_wtp_matrix...)))
heatmap(LE_range, HLE_range, d_rect_wtp_matrix, c =:coolwarm, margin=3Plots.mm,
	xlabel= "LE", ylabel= "HLE", subplot= 1,title="WTP for ψ")
savefig("figures/Olshansky_plots/WTP_psi_htmp.pdf")


temp_df = d_rect_wtp_df[(d_rect_wtp_df.LE .< 95),:]

plts = surface(temp_df.LE, temp_df.HLE, temp_df.WTP_1y,
	camera=(45,15), xlabel= "LE", ylabel= "HLE", zlabel = "WTP",
	title="WTP for ψ", colorbar =:none, margin=3Plots.mm)
