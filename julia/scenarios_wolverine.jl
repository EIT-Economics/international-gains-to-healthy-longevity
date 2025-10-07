"""
Murphy and Topel model with Wolverine on frailty
"""

if occursin("jashwin", pwd())
    cd("C://Users/jashwin/Documents/GitHub/international-gains-to-healthy-longevity/")
else
    cd("/Users/julianashwin/Documents/GitHub/international-gains-to-healthy-longevity/")
end

using Statistics, Parameters, DataFrames
using QuadGK, NLsolve, Roots, FiniteDifferences
using Plots, XLSX, ProgressMeter, Formatting, TableView, Latexify, LaTeXStrings

# Import functions
include("src/TargetingAging.jl")

# Plotting backend
gr()

param = :Reset
Wolverine = 50
age_start = 0

"""
Wolverine
"""
## Define options and parameters
# Biological model
bio_pars = BiologicalParameters()
bio_pars.Reset = 0.0001
# Economic model
econ_pars = EconomicParameters()
# Reset bio_pars.γ (exponent on frailty in mortality) to match starting LE
bio_pars.γ = find_zero(function (x) bio_pars.γ = x; bio_pars.LE_base - LE(bio_pars,
	(no_compress = false,age_start = 0., Wolverine = 50), aa = 0.) end, (1e-16, 100.))


# Specify options for scenario to run
opts = (param = param, bio_pars0 = deepcopy(bio_pars),
    Age_range = [0, 20, 40, 60, 80], AgeGrad = 20, AgeRetire = 65, redo_z0 = false, no_compress = false,
	prod_age = false, age_start = age_start, Wolverine = Wolverine)

LE(bio_pars, opts, aa = 0.)
LE(opts.bio_pars0, opts, aa = 0.)
HLE(bio_pars, opts, aa = 0.)

# Folder to export results
export_folder = "figures/Wolverine/"

## Define and test vars_df
# DataFrame to keep track of life-cycle variables
vars_df_base = define_vars_df(bio_pars, econ_pars, opts)
# Plot biological variables
fig1 = plot(layout = (2,3), size = (800,400))
fig1 = plot!(vars_df_base.λ,title = "Mortality",legend = :topleft, subplot = 1,
	label = "LE = "*string(round(LE(bio_pars, opts, aa = 0.), digits = 2)))
fig1 = plot!(vars_df_base.S,title = "S",legend = false, subplot = 2)
fig1 = plot!(vars_df_base.Sgrad,title = "Sgrad",legend = false, subplot = 3)
fig1 = plot!(vars_df_base.D,title = "Disability",legend = false, subplot = 4)
fig1 = plot!(vars_df_base.H,title = "H",legend = false, subplot = 5)
fig1 = plot!(vars_df_base.Hgrad,title = "Hgrad",legend = false, subplot = 6)
#display(fig1)

bio_pars.Reset = 23.15 # 11.225
vars_df_wolv = define_vars_df(bio_pars, econ_pars, opts)
# Plot biological variables
fig1 = plot!(vars_df_wolv.λ,title = "Mortality",legend = :topleft, subplot = 1,
	label = "LE = "*string(round(LE(bio_pars, opts, aa = 0.), digits = 2)))
fig1 = plot!(vars_df_wolv.S,title = "S",legend = false, subplot = 2)
fig1 = plot!(vars_df_wolv.Sgrad,title = "Sgrad",legend = false, subplot = 3)
fig1 = plot!(vars_df_wolv.D,title = "Disability",legend = false, subplot = 4)
fig1 = plot!(vars_df_wolv.H,title = "H",legend = false, subplot = 5)
fig1 = plot!(vars_df_wolv.Hgrad,title = "Hgrad",legend = false, subplot = 6)
#display(fig1)

bio_pars = deepcopy(opts.bio_pars0)
bio_pars.δ2 = 0.75785 # 0.8698
vars_df_pp = define_vars_df(bio_pars, econ_pars, (param = :δ2, no_compress = false, Wolverine = 0,
	redo_z0 = false, prod_age = false, age_start = 30, AgeGrad = 20, AgeRetire = 65, bio_pars0 = opts.bio_pars0))
# Plot biological variables
fig1 = plot!(vars_df_pp.λ,title = "Mortality",legend = :topleft, subplot = 1,
	label = "LE = "*string(round(LE(bio_pars, opts, aa = 0.), digits = 2)))
fig1 = plot!(vars_df_pp.S,title = "S",legend = false, subplot = 2)
fig1 = plot!(vars_df_pp.Sgrad,title = "Sgrad",legend = false, subplot = 3)
fig1 = plot!(vars_df_pp.D,title = "Disability",legend = false, subplot = 4)
fig1 = plot!(vars_df_pp.H,title = "H",legend = false, subplot = 5)
display(fig1)



fig_H = plot(ylims = (0,1), #title = "Health under Peter Pan and Wolverine",
	xlabel = "Age", ylabel = "Health", label = false, legend = :topright)
fig_H = plot!(vars_df_base.age, vars_df_base.H, label = "Baseline", color = :grey)
fig_H = plot!(vars_df_wolv.age, vars_df_wolv.H, label = "Wolverine", color = :blue)
fig_H = plot!(vars_df_pp.age, vars_df_pp.H, label = "Peter Pan", color = :green)
fig_H = plot!(xlims = (0,150), xticks = [0, 30, 50, 100, 150])
fig_H = plot!(size = (350,350))
savefig(export_folder*"H_pp_wolv_comp.pdf")



# Reset
bio_pars = deepcopy(opts.bio_pars0)
vars_df = define_vars_df(bio_pars, econ_pars, opts)


## Initialise results dfs
# Sizes
Reset_grid_n=Int64(11)
Reset_at_age_n=size(opts.Age_range,1)

# Parameters
param_grid= Array{BiologicalParameters,2}(fill(bio_pars,
	(Reset_grid_n,Reset_at_age_n)))

# Save starting point for reference
LE0=LE(bio_pars, opts)
HLE0=HLE(bio_pars, opts)


# Results to keep track of
results_vars = [:param, :LE, :HLE, :dLE, :WTP_0, :WTP_H_0, :WTP_S_0, :WTP_20, :WTP_H_20, :WTP_S_20,
	:WTP_40, :WTP_H_40, :WTP_S_40, :WTP_60, :WTP_H_60, :WTP_S_60, :WTP_80, :WTP_H_80, :WTP_S_80]
results_df = DataFrame(zeros(Reset_grid_n, length(results_vars)), results_vars)
results_df.LE = Float64.(LE0:(LE0+10))

results_pp_df = deepcopy(results_df)


## Run the scenarios
# Initialise figures
fig1 = plot(layout = (2,3), size = (800,400))
fig2 = plot(layout = (2,3), size = (800,400))
fig3 = plot(layout = (2,3), size = (800,400))
# Set up grid points
prog = Progress(Reset_grid_n, desc = "Running scenarios: ")
for ii in 1:Reset_grid_n
	### Identify necessary parameter shift ###
	# Reset the parameters
	bio_pars =deepcopy(opts.bio_pars0)
    # Set parameter to achieve target (healthy) life expectancy
	LE_new = results_df.LE[ii]
	# Identify the parameter value that gives a certain LE (if this doesn't work, set everything to NaN)
	setfield!(bio_pars, opts.param,
		find_zero(function(x) temp_pars=deepcopy(bio_pars);
		setfield!(temp_pars, opts.param, x);
		LE_new - LE(temp_pars, opts, aa = 0.0) end,
		(1e-16, opts.Wolverine)))
	# Save parameter
	results_df.param[ii] = getfield(bio_pars, opts.param)
	# Derivative of LE and HLE wrt parameter
	results_df.dLE[ii] = round(central_fdm(5,1,
		max_range = getfield(bio_pars, opts.param))(function(x)
		temp_pars=deepcopy(bio_pars);
		setfield!(temp_pars, opts.param, x);
		LE(temp_pars, opts, aa = 0.0) end, getfield(bio_pars, opts.param)), digits = 10)

	### Compute the new variables ###
	# Solve life cycle model
	vars_df = define_vars_df(bio_pars, econ_pars, opts)
	results_df.HLE[ii] = HLE(bio_pars, opts)
	# Reset z0 if necessary
	if opts.redo_z0
		econ_pars.z0 = compute_z0(econ_pars, vars_df)
	end
	# Compute WTP
	vars_df = compute_wtp(bio_pars, econ_pars, opts, vars_df, current_age = 0.0)

	### Fill table ###
	# Reset (healthy) life expectancy based on parameters
	for jj in opts.Age_range
		# WTP at target_age
	    results_df[ii,Symbol("WTP_S_"*string(jj))] = vars_df.WTP_S[Int(jj+1)]
	    results_df[ii,Symbol("WTP_H_"*string(jj))] = vars_df.WTP_H[Int(jj+1)]
		results_df[ii,Symbol("WTP_"*string(jj))] = vars_df.WTP[Int(jj+1)]
	end
	# Plot life cycle variables
	plot_periods = [1,6,11]
	if ii in plot_periods
		col = get.([palette(:Dark2_3)], (ii-1)/(maximum(plot_periods)))[1]

		fig1, fig2, fig3 = scenario_plt(vars_df, fig1, fig2, fig3, col)
    end
	next!(prog)
end
plot(fig1)
savefig(export_folder*"/biological_vars_"*string(opts.param)*".pdf")
plot(fig2)
savefig(export_folder*"/economic_vars_"*string(opts.param)*".pdf")
plot(fig3)
savefig(export_folder*"/value_vars_"*string(opts.param)*".pdf")





"""
Peter Pan
"""
## Reset the options for Peter Pan
param = :δ2
Wolverine = 0
age_start = 30
# Biological model
bio_pars = deepcopy(opts.bio_pars0)
# Specify options for scenario to run
opts = (param = param, bio_pars0 = deepcopy(bio_pars),
    Age_range = [0, 20, 40, 60, 80], AgeGrad = 20, AgeRetire = 65, redo_z0 = false, no_compress = false,
	prod_age = false, age_start = age_start, Wolverine = Wolverine)

## Run the scenarios
# Initialise figures
fig1 = plot(layout = (2,3), size = (800,400))
fig2 = plot(layout = (2,3), size = (800,400))
fig3 = plot(layout = (2,3), size = (800,400))
# Set up grid points
prog = Progress(Reset_grid_n, desc = "Running scenarios: ")
for ii in 1:Reset_grid_n
	### Identify necessary parameter shift ###
	# Reset the parameters
	bio_pars =deepcopy(opts.bio_pars0)
    # Set parameter to achieve target (healthy) life expectancy
	LE_new = results_df.LE[ii]
	# Identify the parameter value that gives a certain LE (if this doesn't work, set everything to NaN)
	setfield!(bio_pars, opts.param,
		find_zero(function(x) temp_pars=deepcopy(bio_pars);
		setfield!(temp_pars, opts.param, x);
		LE_new - LE(temp_pars, opts, aa = 0.0) end,
		(1e-16, 2.0)))
	# Save parameter
	results_pp_df.param[ii] = getfield(bio_pars, opts.param)
	# Derivative of LE and HLE wrt parameter
	results_pp_df.dLE[ii] = round(central_fdm(5,1,
		max_range = getfield(bio_pars, opts.param))(function(x)
		temp_pars=deepcopy(bio_pars);
		setfield!(temp_pars, opts.param, x);
		LE(temp_pars, opts, aa = 0.0) end, getfield(bio_pars, opts.param)), digits = 10)

	### Compute the new variables ###
	# Solve life cycle model
	vars_df = define_vars_df(bio_pars, econ_pars, opts)
	results_pp_df.HLE[ii] = HLE(bio_pars, opts)
	# Reset z0 if necessary
	if opts.redo_z0
		econ_pars.z0 = compute_z0(econ_pars, vars_df)
	end
	# Compute WTP
	vars_df = compute_wtp(bio_pars, econ_pars, opts, vars_df, current_age = 0.0)

	### Fill table ###
	# Reset (healthy) life expectancy based on parameters
	for jj in opts.Age_range
		# WTP at target_age
	    results_pp_df[ii,Symbol("WTP_S_"*string(jj))] = vars_df.WTP_S[Int(jj+1)]
	    results_pp_df[ii,Symbol("WTP_H_"*string(jj))] = vars_df.WTP_H[Int(jj+1)]
		results_pp_df[ii,Symbol("WTP_"*string(jj))] = vars_df.WTP[Int(jj+1)]
	end
	# Plot life cycle variables
	plot_periods = [1,6,11]
	if ii in plot_periods
		col = get.([palette(:Dark2_3)], (ii-1)/(maximum(plot_periods)))[1]

		fig1, fig2, fig3 = scenario_plt(vars_df, fig1, fig2, fig3, col)
    end
	next!(prog)
end
plot(fig1)
savefig(export_folder*"/PP_biological_vars_"*string(opts.param)*".pdf")
plot(fig2)
savefig(export_folder*"/PP_economic_vars_"*string(opts.param)*".pdf")
plot(fig3)
savefig(export_folder*"/PP_value_vars_"*string(opts.param)*".pdf")






WTP_vars = vcat([:LE0, :param], Symbol.("Age=".*string.(opts.Age_range)))
WTP_table = DataFrame(zeros(Reset_grid_n, Reset_at_age_n+2), WTP_vars)
WTP_table.LE0 = results_df.LE
WTP_table.param = round.(results_df.param, digits = 3)

WTP_pp_table = deepcopy(WTP_table)
WTP_pp_table.param = round.(results_pp_df.param, digits = 3)

WTP_comp_table = deepcopy(WTP_table)
WTP_comp_table.param = Symbol.("Reverse by ".*string.(WTP_table.param).*" yrs vs slow by ".*string.(WTP_pp_table.param))

for jj in opts.Age_range
	WTP_table[:,Symbol("Age="*string(jj))] = Int.(round.(results_df[:,Symbol("WTP_"*string(jj))]./results_df.dLE))
	WTP_pp_table[:,Symbol("Age="*string(jj))] = Int.(round.(results_pp_df[:,Symbol("WTP_"*string(jj))]./results_pp_df.dLE))
	WTP_comp_table[:,Symbol("Age="*string(jj))] = Int.(WTP_table[:,Symbol("Age="*string(jj))] -
		WTP_pp_table[:,Symbol("Age="*string(jj))])
end


table_string = latexify(WTP_table[[1,2,3,4,5,10],:], env = :tabular)
open(export_folder*"/wtp_table.tex","w") do io
   println(io, table_string)
end

table_string = latexify(WTP_pp_table[[1,2,3,4,5,10],:], env = :tabular)
open(export_folder*"/wtp_pp_table.tex","w") do io
   println(io, table_string)
end

table_string = latexify(WTP_comp_table[[1,2,3,4,5,10],:], env = :tabular)
open(export_folder*"/wtp_comp_table.tex","w") do io
   println(io, table_string)
end




"""
Compare repeated reversals
"""
## Reset the options for Peter Pan
param = :δ2
Wolverine = 0
starting_ages = [30, 50, 70]
δ2s = [1.0, 0.75, 0.5, 0.25, 0.0]
age_start = 30
# Biological model
bio_pars = deepcopy(opts.bio_pars0)
econ_pars = EconomicParameters()
# Specify options for scenario to run
opts = (param = param, bio_pars0 = deepcopy(bio_pars),
    Age_range = [0, 20, 40, 60, 80], AgeGrad = 20, AgeRetire = 65, redo_z0 = false, no_compress = false,
	prod_age = false, age_start = age_start, Wolverine = Wolverine)

## Intialise results table
repeated_vars = [:starting_age, :δ2, :param, :LE, :HLE, :V, :WTP_0, :WTP_20, :WTP_40, :WTP_60, :WTP_80 ]
repeated_df = DataFrame(zeros(length(starting_ages)*length(δ2s), length(repeated_vars)), repeated_vars)
repeated_df.starting_age = repeat(starting_ages, inner = length(δ2s))
repeated_df.δ2 = repeat(δ2s, length(starting_ages))
repeated_df.param = repeat(Symbol.(["None", "3 months", "6 months", "9 months", "12 months"]), length(starting_ages))

## Initial vars_df
vars_df_orig = define_vars_df(bio_pars, econ_pars, opts)
μ_orig = compute_μ(bio_pars, econ_pars, opts, vars_df_orig)
U_orig = compute_U.([bio_pars], [econ_pars], [opts], [vars_df_orig], vars_df_orig.age,
	reset_bio = false, reset_econ = false)

U_orig[isnan.(U_orig)] .= 0.0


## Loop through deltas
# Initialise figures
fig1 = plot(layout = (2,3), size = (800,400))
fig2 = plot(layout = (2,3), size = (800,400))
fig3 = plot(layout = (2,3), size = (800,400))
# Loop over starting ages and δ2s
prog = Progress(length(δ2s)*length(starting_ages), desc = "Running repeated Wolverine scenarios: ")
for age_start in starting_ages
	for δ2_ in δ2s
		# Identify results row
		rowid = Int.(1:nrow(repeated_df))[(repeated_df.starting_age .== age_start) .& (repeated_df.δ2 .== δ2_)][1]
		# Set options
		opts_now = (param = param, bio_pars0 = deepcopy(opts.bio_pars0),
			Age_range = [0, 20, 40, 60, 80], AgeGrad = 20, AgeRetire = 65, redo_z0 = false, no_compress = false,
			prod_age = false, age_start = age_start, Wolverine = Wolverine)
		# Set biological parameters
		bio_pars = deepcopy(opts.bio_pars0)
		bio_pars.MaxAge = 500
		bio_pars.δ2 = δ2_
		# 
		try
			vars_df_new = define_vars_df(bio_pars, econ_pars, opts_now)
		catch
			bio_pars.MaxAge = 20400
			vars_df_new = define_vars_df(bio_pars, econ_pars, opts_now)
		end
		display(print("age_start: "*string(age_start)*", δ2: "*string(bio_pars.δ2)*", MaxAge: "*string(bio_pars.MaxAge)))

		fig1, fig2, fig3 = scenario_plt(vars_df_new[1:501,:], fig1, fig2, fig3, rowid)
		U_new = compute_U.([bio_pars], [econ_pars], [opts], [vars_df_new], vars_df_new.age,
			reset_bio = false, reset_econ = false)

		# As we only look at WTP at a few ages we can keep WTP a bit shorter
		WTP_diff = (U_new[1:101] .- U_orig[1:101])./μ_orig
		WTP_diff[isnan.(WTP_diff)] .= 0.0
		plot!(WTP_diff, subplot=6)

		repeated_df.LE[rowid] = round(LE(bio_pars, opts_now), digits = 1)
		repeated_df.HLE[rowid] = round(HLE(bio_pars, opts_now), digits = 1)
		repeated_df.V[rowid] = round(vars_df_new.V[51]/1e6, digits = 2)
		repeated_df.WTP_0[rowid] = round(WTP_diff[1], digits = 2)
		repeated_df.WTP_20[rowid] = round(WTP_diff[21], digits = 2)
		repeated_df.WTP_40[rowid] = round(WTP_diff[41], digits = 2)
		repeated_df.WTP_60[rowid] = round(WTP_diff[61], digits = 2)
		repeated_df.WTP_80[rowid] = round(WTP_diff[81], digits = 2)
		next!(prog)
	end

end
plot(fig1)
plot(fig2)
plot(fig3)

export_tab = deepcopy(repeated_df)
export_tab[:,7:11] = round.(export_tab[:,7:11]./1e6, digits = 2)
table_string = latexify(export_tab, env = :tabular)
open(export_folder*"/repeated_table.tex","w") do io
   println(io, table_string)
end
CSV.write(export_folder*"repeated_wolverine.csv", repeated_df)
