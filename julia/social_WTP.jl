"""
Social WTP from baseline Murphy and Topel model
"""

using Statistics, Parameters, DataFrames, Dates
using QuadGK, NLsolve, Roots, FiniteDifferences
using Plots, XLSX, ProgressMeter, Formatting, TableView, Latexify, LaTeXStrings

# Import functions
include("julia/TargetingAging.jl")

# Plotting backend
gr()

"""
Import population data
"""
# Choose a country and year
country = "United States of America"
year = 2017
#@assert year%5 == 0 "Years must be multiple of 5 to match to WPP data"
# Import population data
pop_estimates = CSV.read("data/WPP/population.csv", DataFrame, ntasks = 1) # Estimates
pop_projections = CSV.read("data/WPP/population.csv", DataFrame, ntasks = 1) # Projections
pop_df = vcat(pop_estimates, pop_projections[(pop_projections.Year .> 2020),:])
pop_structure = pop_df[(pop_df.Country .== country),:]
# GBD_pop_data
mort_df = CSV.read("data/GBD/mortality_rates.csv", DataFrame, ntasks = 1)
hist_pop_df = mort_df[:,[:location_name, :year, :age, :population]]

# Import LE data
LE_estimates = CSV.read("data/WB/WHO_HLE_data.csv", DataFrame, ntasks = 1)
#LE_projections = CSV.read("data/WPP/WPP_LE_projections.csv", DataFrame, ntasks = 1)
LE_df = LE_estimates
LE_df.LE = LE_df[:,"Life expectancy at birth (years)"]
LE_df.HLE = LE_df[:,"Healthy life expectancy (HALE) at birth (years)"]
LE_df.Year[LE_df.Year.==2019] .= 2020
LE_df.Country[LE_df.Country.=="United Kingdom of Great Britain and Northern Ireland"] .= "United Kingdom"

# Import fertility data
fert_estimates = CSV.read("data/WPP/fertility.csv", DataFrame, ntasks = 1)
fert_projections = CSV.read("data/WPP/fertility.csv", DataFrame, ntasks = 1)
fert_df = vcat(fert_estimates, fert_projections[(fert_projections.Year_low .>= 2020),:])
sort!(fert_df,[:Country, :Year_low])

# Import GDP data
hist_GDP_df = CSV.read("data/WB/real_gdp_data.csv", DataFrame, ntasks = 1)



# Baseline VSL for US
VSL_ref = 11.5e6
# Baseline GDP per capita for US 2020
Y_ref = 58240.80330411436
# Income elasticity of VSL
η = 1.0



# A couple useful variables
trillion = 1e12
export_folder = "figures/"
no_compress = true



"""
Look at effect of no aging after age_start
"""
# Parameter to change
param = :δ2
# Biological model
bio_pars = BiologicalParameters()
# Economic model
econ_pars = EconomicParameters()
# Options
opts = (param = param, bio_pars0 = deepcopy(bio_pars), LE_max = 40, step = 1, HLE = false,
    Age_range = [0], AgeGrad = 20, AgeRetire = 65,
	plot_shape_3d = false, zoom_pars = [0.5,0.9], redo_z0 = false, no_compress = false,
	prod_age = false, age_start = 65, Wolverine = 0, phasing = 5)

# Default variables
bio_pars = BiologicalParameters()
vars_df_old = define_vars_df(bio_pars, econ_pars, opts)
fig_test = plot(vars_df_old.λ, label = "old", color = :black)
plot!(vars_df_old.S, label = false, color = :black)
plot!(vars_df_old.H, label = false, color = :black)
# Phased Peter-Pan/Wolverine
δ2_final = 0.8
for pp in 1:opts.phasing
	bio_pars.δ2 = 1.0 - (pp/opts.phasing)*(1.0 - δ2_final)
	vars_df_new = define_vars_df(bio_pars, econ_pars, opts)
	# Plot comparison
	plot(fig_test)
	fig_test = plot!(vars_df_new.λ, label = string(pp)*" years in", linestyle = :dash, color = pp)
	fig_test = plot!(vars_df_new.S, label = false, linestyle = :dash, color = pp)
	fig_test = plot!(vars_df_new.H, label = false, linestyle = :dash, color = pp)
end
plot(fig_test)

"""
With this phasing, each of the first five years will have a different frailty curve
"""
# Reset bio_pars
bio_pars = BiologicalParameters()
# Solve the life cycle problem at this starting point,
vars_df = define_vars_df(bio_pars, econ_pars, opts)

function compute_wtp_phasing(bio_pars, econ_pars, opts, vars_df, δ2_final; gradual = true)

	# Multiple biological curves
	ages = Int.(0:bio_pars.MaxAge)
	mortality_curves = repeat([Array{Float64,1}(undef, Int(bio_pars.MaxAge+1))], Int(opts.phasing+1))
	health_curves = deepcopy(mortality_curves)
	survivor_curves = deepcopy(mortality_curves)

	## Calculate the curves
	mortality_curves[1] = mortality.([bio_pars], [opts], ages)
	health_curves[1] = health.([bio_pars], [opts], ages)
	survivor_curves[1] = cumprod(1 .- vcat(0.0, mortality_curves[1][1:(end-1)]))
	if gradual
		# If gradual, then δ2 changes linearly until we get to the final one
		for pp in 1:opts.phasing
			temp_pars = deepcopy(bio_pars)
			temp_pars.δ2 = 1.0 - (pp/opts.phasing)*(1.0 - δ2_final)
			mortality_curves[pp+1] = mortality.([temp_pars], [opts], ages)
			health_curves[pp+1] = health.([temp_pars], [opts], ages)
			survivor_curves[pp+1] = cumprod(1 .- vcat(0.0, mortality_curves[pp+1][1:(end-1)]))
		end
	else
		# If not, we have no change until the last period
		temp_pars = deepcopy(bio_pars)
		mortality_curves[2:opts.phasing] .= [mortality.([temp_pars], [opts], ages)]
		health_curves[2:opts.phasing] .= [health.([temp_pars], [opts], ages)]
		survivor_curves[2:opts.phasing] .= [cumprod(1 .- vcat(0.0, mortality_curves[pp+1][1:(end-1)]))]
		# Last period ofphasing we then get the full effect
		temp_pars.δ2 = δ2_final
		mortality_curves[opts.phasing+1] = mortality.([temp_pars], [opts], ages)
		health_curves[opts.phasing+1] = health.([temp_pars], [opts], ages)
		survivor_curves[opts.phasing+1] = cumprod(1 .- vcat(0.0, mortality_curves[opts.phasing+1][1:(end-1)]))
	end
	plot(survivor_curves, size = (1000,1000))
	plot!(health_curves)


	## Each age will have a different Sdelta and Hdelta
	Sdelta = repeat([zeros(Int(bio_pars.MaxAge+1))], Int(bio_pars.MaxAge+1))
	Hdelta = repeat([zeros(Int(bio_pars.MaxAge+1))], Int(bio_pars.MaxAge+1))
	for aa in ages
		# Define original survival curve
		S_old = deepcopy(survivor_curves[1])
		# Before current age, S was obviously 1
		S_old[ages .< aa] .= 1.0
		# Adjust survival going forward to reflect the fact we've made it this far
		S_old[ages .>= aa] = S_old[ages .>= aa]./S_old[aa+1]
		# Define new survival rate
		S_new = zeros(length(S_old))
		# Before current age, S still 1
		S_new[ages .<= aa] .= 1.0
		# Then for each of the next opts.phasing years we have get extra improvement
		for pp in 1:opts.phasing
			if aa+pp < bio_pars.MaxAge
				S_new[ages .== (aa+pp)] = S_new[ages .== (aa+pp-1)]*(1 - mortality_curves[Int(pp+1)][Int(aa+pp)])
			end
		end
		# After that we are in the new steady state
		for aa_after in ages[ages .> aa+opts.phasing]
			S_new[ages .== aa_after] =
				S_new[ages .== Int(aa_after-1)]*(1 - mortality_curves[Int(opts.phasing+1)][Int(aa_after)])
		end
		# Save the Sdelta for this generation
		Sdelta[Int(aa+1)] = S_new - S_old

		# Now get the health
		H_old = deepcopy(health_curves[1])
		# Define new health
		H_new = zeros(length(H_old))
		# Before current age, H hasn't changed
		H_new[ages .<= aa] = H_old[ages .<= aa]
		# Then for each of the next opts.phasing years we have get extra improvement
		for pp in 1:opts.phasing
			if aa+pp < bio_pars.MaxAge
				H_new[ages .== max((aa+pp-1),0)] .= health_curves[Int(pp+1)][ages .== max((aa+pp-1),0)]
			end
		end
		# After that we are in the new steady state
		H_new[ages .>= aa+opts.phasing] = health_curves[Int(opts.phasing+1)][ages .>= aa+opts.phasing]
		# Save the Hdelta for this generation
		Hdelta[Int(aa+1)] = H_new - H_old
	end

	## Compute WTP for each generation
	# Initialise a WTP vector
	WTP_S = zeros(length(ages))
	WTP_H = zeros(length(ages))
	WTP = zeros(length(ages))
	# Loop over ages
	for aa in ages
		# Identify relevant rows
		rows = (aa+1):Int(bio_pars.MaxAge+1)
		# Calculate some stuff for WTP_H
		vars_df.Hgrad = Hdelta[(aa+1)]
		q = compute_q(econ_pars, vars_df)
		# And for WTP_S
		S_a = min.(vars_df.S./vars_df.S[(aa+1)], [1.0])
		Sgrad = Sdelta[(aa+1)]
		#

		# WTP for survival
		WTP_S[(aa+1)]=sum(vars_df.v[rows].*Sgrad[rows].*vars_df.discount[rows])*vars_df.discount[(aa+1)]^-1
		if (isnan(WTP_S[(aa+1)]) | isinf(WTP_S[(aa+1)]))
			WTP_S[(aa+1)] = 0.
		end
		WTP_H[(aa+1)]=sum(q[rows].*S_a[rows].*vars_df.discount[rows]).*vars_df.discount[(aa+1)]^-1
		if (isnan(WTP_H[(aa+1)]) | isinf(WTP_H[(aa+1)]))
			WTP_H[(aa+1)] = 0.
		end

	end

	vars_df.WTP_S = WTP_S
	vars_df.WTP_H = WTP_H
	vars_df.WTP = vars_df.WTP_S + vars_df.WTP_H

	return vars_df
end


# Solve the life cycle problem at this starting point,
vars_df = define_vars_df(bio_pars, econ_pars, opts)
δ2_final = 0.9
vars_df = compute_wtp_phasing(bio_pars, econ_pars, opts, vars_df, δ2_final, gradual = true)
plot(vars_df.WTP)



"""
Define options and parameters
"""
# Parameter to change
param = :δ2
# Biological model
bio_pars = BiologicalParameters()
# Economic model
econ_pars = EconomicParameters()
# Options
opts = (param = param, bio_pars0 = deepcopy(bio_pars), LE_max = 40, step = 1, HLE = false,
    Age_range = [0], AgeGrad = 20, AgeRetire = 65, redo_z0 = false, no_compress = false,
	prod_age = false, age_start = 0, Wolverine = 0, phasing = 1)
# Phase in gradually or only at end
gradual = true

econ_pars.r = 0.02


"""
Initialise table
"""
# Countries
countries = ["Australia", "Canada", "France", "Germany", "Israel", "Italy", "Japan", "Netherlands",
	"New Zealand", "Spain", "Sweden", "United Kingdom", "United States of America"]
# Starting years
years = [2020]
# Empty dataframe to populate
WTP_vars = [:Country, :Year, :LE, :HLE, :Pop, :GDP_pc, :VSL, :WTP_1y, :WTP_avg, :WTP_0, :WTP_unborn]
WTP_table = DataFrame(Array{Any,2}(zeros(length(countries)*length(years), length(WTP_vars))), WTP_vars)
WTP_table.Country .= ""
# US nominal GDP to make 2017 adjustment
US_nominal_GDP = CSV.read("data/WB/US_nominal_income_pc.csv", DataFrame, ntasks = 1)
US_nominal_GDP.year = Dates.year.(US_nominal_GDP.DATE)
US_nominal_GDP.GDPpc = US_nominal_GDP.A792RC0A052NBEA

#US_nominal_GDP = CSV.read("data/WB_GDP_data/US_nominal_GDP.csv", DataFrame, ntasks = 1)
#US_nominal_GDP.year = Dates.year.(US_nominal_GDP.DATE)
#US_nominal_GDP.GDPpc = US_nominal_GDP.GDPA.*1e9



"""
Populate table
"""
# Helper to track row ids
rowid = 1
# Loop through countries
for country in countries
	display("Calculating for "*country)
	# Loop through years
	for year in years

		# Extract the relevant population data
		if year == 2017
			pop_structure = hist_pop_df[(hist_pop_df.location_name .== country),:]
			pop_structure = pop_structure[(pop_structure.year .== year),:]
			GDP_estimates = hist_GDP_df[(hist_GDP_df.location_name .== country),:]
			GDP_est = GDP_estimates[(GDP_estimates.year .== year),:real_gdp_usd][1]
			N = sum(pop_structure.population)
			GDP_pc = GDP_est/N

			VSL_target = VSL_ref.*(US_nominal_GDP.GDPpc[(US_nominal_GDP.year .== 2017)]./
				US_nominal_GDP.GDPpc[(US_nominal_GDP.year .== 2020)])[1]
			defl_2017 = (US_nominal_GDP.GDPpc[(US_nominal_GDP.year .== 2017)]./
				GDP_estimates.real_gdp_usd[(GDP_estimates.year .== 2017)])[1]
			defl_2011 = (US_nominal_GDP.GDPpc[(US_nominal_GDP.year .== 2011)]./
				GDP_estimates.real_gdp_usd[(GDP_estimates.year .== 2011)])[1]
			#VSL_target = 9.9e6*defl_2017/defl_2011

            HLE_est = 68.5
			LE_est = 78.9
		else
			@assert year%5 == 0 "Years must be multiple of 5 to match to WPP data"
			pop_structure = pop_df[(pop_df.Country .== country),:]
			pop_structure = pop_structure[(pop_structure.Year .== year),:]
			pop_structure.population = pop_structure.value .* 1000
			pop_structure.age = pop_structure.Age_mid

			# Extract the relevant LE
			LE_estimates = LE_df[(LE_df.Country .== country),:]
			LE_est = LE_estimates[(LE_estimates.Year .== year),:LE][1]
			HLE_est = LE_estimates[(LE_estimates.Year .== year),:HLE][1]

			# Relevant GDP per capita
			GDP_estimates = GDP_df[(GDP_df.Country .== country),:]
			GDP_est = GDP_estimates[(GDP_estimates.Year .== year),:GDP][1]*1e6
			# per capita
			N = sum(pop_structure.population)
			GDP_pc = GDP_est/N
			# Compute target VSL
			VSL_target = VSL_ref*(GDP_pc/Y_ref)^η
		end
		# Fertility projections
		fert_country = fert_df[fert_df.Country.== country,:]
		all_years = Int64.(minimum(fert_country.Year_low):(maximum(fert_country.Year_high)-1))
		all_values = repeat(fert_country.value, inner = 5)./5
		fert_projs = DataFrame(year = all_years, births = all_values.*1000)
		fert_projs = fert_projs[fert_projs.year .> year,:]
		# Loop through extra years of gains
		for ii in [1]
			# Reset parameters
			bio_pars = deepcopy(opts.bio_pars0)
			# Current LE in that country
			LE_current = LE_est + (ii-1)
			HLE_current = HLE_est + (ii-1)
			# Set lifespan T to match life expectancy from WPP data
			try
				setfield!(bio_pars, :T,
					find_zero(function(x) temp_pars=deepcopy(bio_pars);
					setfield!(temp_pars, :T, x);
					LE_current - LE(temp_pars, opts, aa = 0.0) end,
					getfield(bio_pars, :T)))
			catch
				setfield!(bio_pars, :T,
					find_zero(function(x) temp_pars=deepcopy(bio_pars);
					setfield!(temp_pars, :T, x);
					LE_current - LE(temp_pars, opts, aa = 0.0) end,
					40.))
			end
			# Set health variable α to match life expectancy from WPP data
			if year != 2017
				setfield!(bio_pars, :α,
					find_zero(function(x) temp_pars=deepcopy(bio_pars);
					setfield!(temp_pars, :α, x);
					HLE_current - HLE(temp_pars, opts, aa = 0.0) end,
					getfield(bio_pars, :α)))
			end

			# Calculate dLE for param
			δ2_final = find_zero(function(x) temp_pars=deepcopy(bio_pars);
				setfield!(temp_pars, :δ2, x);
				LE_current + ii - LE(temp_pars, opts, aa = 0.0) end, 1.0)

			# Set WageChild to match target VSL
			age_range = Int.(1:nrow(pop_structure))[
				(pop_structure.age .> 19) .& (pop_structure.age .< 24)]
			setfield!(econ_pars, :WageChild,
				find_zero(function(x) temp_pars=deepcopy(econ_pars);
				setfield!(temp_pars, :WageChild, x);
				temp_df = define_vars_df(bio_pars, temp_pars, opts);
				temp_df = innerjoin(pop_structure[:,[:age, :population]],temp_df, on = :age);
				soc_V = (temp_df.population[age_range]).*temp_df.V[age_range];
				VSL_target - sum(soc_V)/sum(temp_df.population[age_range]) end,
				6.9))
			# Solve model and compute social WTP
			vars_df = define_vars_df(bio_pars, econ_pars, opts)
			vars_df = compute_wtp_phasing(bio_pars, econ_pars, opts, vars_df, δ2_final, gradual = gradual)
			plot(vars_df.WTP)
			# Merge in population data
			soc_df = vars_df[in.(vars_df.age, [pop_structure.age]), :]
			soc_df = innerjoin(pop_structure[:,[:age, :population]],
				soc_df[:,[:age, :WTP_S, :WTP_H, :WTP]], on = :age)
			# Compute social WTP
			total_WTP = sum(soc_df.population .* soc_df.WTP)
			average_WTP = total_WTP/N
			# Unborn WTP
			unborn_WTP = sum(vars_df.WTP[1].*fert_projs.births.*vars_df.discount[2:(nrow(fert_projs)+1)])

			# Populate row
			WTP_table.Country[rowid] = country
			WTP_table.Year[rowid] = year
			WTP_table.LE[rowid] = round(LE_current, digits = 1)
			WTP_table.HLE[rowid] = round(HLE_current, digits = 1)
			WTP_table.Pop[rowid] = round(N/1000000, digits = 2)
			WTP_table.GDP_pc[rowid] = Int(round(GDP_pc))
			WTP_table.VSL[rowid] = round(VSL_target/1000000, digits = 3)
			WTP_table.WTP_1y[rowid] = round(total_WTP/trillion, digits = 3)
			WTP_table.WTP_avg[rowid] = round(average_WTP/1000, digits = 3)
			WTP_table.WTP_0[rowid] = round(vars_df.WTP[1]/1000, digits = 3)
			WTP_table.WTP_unborn[rowid] = round(unborn_WTP/trillion, digits = 3)
			# Progress report
			display(string(country)*" social WTP for extra year with "*string(year)*" population is "*
				string(round(total_WTP/trillion,digits = 2))*" trillion USD")
			# Add increment
			rowid += 1

		end
	end
end




"""
Export table
"""

WTP_table.Country = Symbol.(WTP_table.Country)
table_string = latexify(WTP_table, env = :tabular)
open(export_folder*"/social_wtp_table.tex","w") do io
   println(io, table_string)
end

CSV.write(export_folder*"/social_wtp_table.csv", WTP_table)
