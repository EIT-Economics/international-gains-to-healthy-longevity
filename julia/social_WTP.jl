"""
Social WTP from baseline Murphy and Topel model
"""


using Statistics, Parameters, DataFrames, Dates
using QuadGK, NLsolve, Roots, FiniteDifferences
using Plots, XLSX, ProgressMeter, Formatting, Latexify, LaTeXStrings

# Import functions
include("TargetingAging.jl")

# Plotting backend
gr()


"""
Parse command-line arguments for filtering
"""
# Default: Analyze all available countries and years
# Usage: julia international_empirical.jl [--countries "country1,country2"] [--years 2000:2015]

# Reference list of developed countries (matching Python DEFAULT_COUNTRIES)
DEFAULT_COUNTRIES = [
	"Australia", "Canada", "France", "Germany", "Israel", "Italy", "Japan", "Netherlands",
	"New Zealand", "Spain", "Sweden", "United Kingdom", "United States of America"
]
# Starting years
DEFAULT_YEARS = [2015]

# Parse arguments
target_countries = nothing
target_years = nothing

i = 1
while i <= length(ARGS)
	global target_countries, target_years, i  # Declare as global for assignment in loop
	
	if ARGS[i] == "--countries"
		if i + 1 <= length(ARGS)
			# Split comma-separated list
			target_countries = strip.(split(ARGS[i + 1], ","))
			i += 2
		else
			error("--countries requires a value (comma-separated list)")
		end
	elseif ARGS[i] == "--years"
		if i + 1 <= length(ARGS)
			# Parse year range (e.g., "1990:2000") or comma-separated list
			year_arg = ARGS[i + 1]
			if occursin(":", year_arg)
				# Range format: "1990:2000"
				year_parts = parse.(Int, split(year_arg, ":"))
				target_years = year_parts[1]:year_parts[2]
			else
				# Comma-separated list: "1990,1995,2000"
				target_years = parse.(Int, strip.(split(year_arg, ",")))
			end
			i += 2
		else
			error("--years requires a value (e.g., '2000:2015' or '2000,2005,2010')")
		end
	elseif ARGS[i] == "--default"
		# Use default countries and years (1990-1999)
		target_countries = DEFAULT_COUNTRIES
		target_years = DEFAULT_YEARS
		i += 1
	elseif ARGS[i] == "--help" || ARGS[i] == "-h"
		println("""
		Usage: julia --project=julia julia/social_WTP.jl [OPTIONS]
		
		Options:
		  --countries "country1,country2"    Filter for specific countries (comma-separated)
		  --years YEARS                      Filter for specific years (range: 2000:2015 or list: 2000,2005,2010)
		  --default                          Use default 13 countries and years 2015
		  --help, -h                         Show this help message
		
		Examples:
		  # Analyze all available data (default)
		  julia --project=julia julia/social_WTP.jl
		  
		  # Analyze only USA and Japan for 2010-2020
		  julia --project=julia julia/social_WTP.jl --countries "United States of America,Japan" --years 2010:2020
		  
		  # Use default developed countries
		  julia --project=julia julia/social_WTP.jl --default
		  
		  # Specific years only
		  julia --project=julia julia/social_WTP.jl --years 2000,2005,2010
		""")
		exit(0)
	else
		error("Unknown argument: $(ARGS[i]). Use --help for usage information.")
	end
end



"""
Import population data
"""

# Import population data from intermediate files
pop_df = CSV.read("intermediate/WPP/population.csv", DataFrame, ntasks = 1)

# Import mortality data
mort_df = CSV.read("intermediate/GBD/mortality_rates.csv", DataFrame, ntasks = 1)
mort_df.age = (mort_df.age_low .+ mort_df.age_high) ./ 2

# Import LE data from intermediate files (do I need WHO_HLE_data.csv?)
# LE_df = CSV.read("intermediate/WPP/life_expectancy.csv", DataFrame, ntasks = 1)
LE_df = CSV.read("data/WHO_HLE_data.csv", DataFrame, ntasks = 1)
LE_df.LE = LE_df[:,"Life expectancy at birth (years)"]
LE_df.HLE = LE_df[:,"Healthy life expectancy (HALE) at birth (years)"]
# Rename country to "United Kingdom" where necessary
replace!(LE_df.Country, "United Kingdom of Great Britain and Northern Ireland" => "United Kingdom")


# Import fertility data from intermediate files
fert_df = CSV.read("intermediate/WPP/fertility.csv", DataFrame, ntasks = 1)

# Import GDP data from intermediate files
GDP_df = CSV.read("intermediate/WB/real_gdp_data.csv", DataFrame, ntasks = 1)
# Handle missing values - CSV.jl already parsed as Float64, just replace Missing with NaN
GDP_df.real_gdp_usd = coalesce.(GDP_df.real_gdp_usd, NaN)

# Determine years to analyze
if target_years === nothing
	years = filter(y -> y in [2000, 2010, 2015, 2019], unique(GDP_df.year))
	println("Analyzing all available years $(years)")
else
	years = filter(y -> y in target_years, unique(GDP_df.year))
	println("Analyzing years $(years)")
end

# Determine available countries based on data availability
# Filter for countries that have GDP, mortality, health, and LE data for the selected years
gdp_mask = .!isnan.(GDP_df.real_gdp_usd) .& in.(GDP_df.year, Ref(years))
gdp_countries = unique(GDP_df[gdp_mask, :location_name])

mort_mask = in.(mort_df.year, Ref(years))
mort_countries = unique(mort_df[mort_mask, :location_name])

health_mask = in.(LE_df.Year, Ref(years))
health_countries = unique(LE_df[health_mask, :Country])

pop_mask = in.(pop_df.Year, Ref(years))
pop_countries = unique(pop_df[pop_mask, :Country])

# Find intersection of all data sources
available_countries = intersect(gdp_countries, mort_countries, health_countries, pop_countries)

# Apply user filter if provided; otherwise use available countries only
if target_countries === nothing
	countries = available_countries
	println("Analyzing all countries with GDP and mortality/health data ($(length(countries)) total)")
else
	countries = filter(c -> c ∈ target_countries, available_countries)
	println("Analyzing $(length(countries)) countries (filtered and with data): $(join(countries, ", "))")
end

last_year = maximum(years) # can't work out health benefit here as no previous year

println("Total country-year combinations: $(length(countries) * length(years))")


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
	global fig_test  # Declare as global for assignment in loop
	bio_pars.δ2 = 1.0 - (pp/opts.phasing)*(1.0 - δ2_final)
	vars_df_new = define_vars_df(bio_pars, econ_pars, opts)
	# Plot comparison
	fig_test = plot!(vars_df_new.λ, label = string(pp)*" years in", linestyle = :dash, color = pp)
	fig_test = plot!(vars_df_new.S, label = false, linestyle = :dash, color = pp)
	fig_test = plot!(vars_df_new.H, label = false, linestyle = :dash, color = pp)
end
savefig(export_folder * "phased_biological_curves_julia.pdf")
println("Plot saved to: " * export_folder * "phased_biological_curves_julia.pdf")

"""
With this phasing, each of the first five years will have a different frailty curve
"""
# Reset bio_pars
bio_pars = BiologicalParameters()
# Solve the life cycle problem at this starting point,
vars_df = define_vars_df(bio_pars, econ_pars, opts)

function compute_wtp_phasing(bio_pars, econ_pars, opts, vars_df, δ2_final; gradual = true, country = "", year = "")

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
	if country != "" && year != ""
		savefig(export_folder * "survivor_health_curves_"*string(country)*"_"*string(year)*"_julia.pdf")
		println("Plot saved to: " * export_folder * "survivor_health_curves_"*string(country)*"_"*string(year)*"_julia.pdf")
	end


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

# Empty dataframe to populate
WTP_vars = [:country, :year, :le, :hle, :population, :real_gdp_pc, :vsl, :wtp, :wtp_pc, :wtp_0, :wtp_unborn]
WTP_table = DataFrame(Array{Any,2}(zeros(length(countries)*length(years), length(WTP_vars))), WTP_vars)
WTP_table.country .= ""



"""
Populate table
"""
# Helper to track row ids
rowid = 1
# Loop through countries
for country in countries
	global pop_structure, bio_pars, δ2_final, vars_df, rowid  # Declare as global for assignment in loops
	println("Calculating for $country...")
	# Loop through years
	for year in years

		# Extract the relevant population data
		pop_structure = pop_df[(pop_df.Country .== country),:]
		# Hacky
		if year == 2019
			println("Warning: No population data for $country in $year, trying 2020 instead...")
			pop_structure = pop_structure[(pop_structure.Year .== 2020),:]
		else
			pop_structure = pop_structure[(pop_structure.Year .== year),:]
		end
		if nrow(pop_structure) == 0
			println("Warning: No population data for $country in $year, skipping...")
			continue
		end
		pop_structure.population = pop_structure.Population .* 1e3
		pop_structure.age = pop_structure.Age_mid

		# Extract the relevant LE
		LE_estimates = LE_df[(LE_df.Country .== country),:]
		LE_year_data = LE_estimates[(LE_estimates.Year .== year),:]
		if nrow(LE_year_data) == 0
			println("Warning: No LE data for $country in $year, skipping...")
			continue
		else
			LE_est = LE_year_data.LE[1]
			HLE_est = LE_year_data.HLE[1]
		end

		# Relevant GDP per capita
		GDP_estimates = GDP_df[(GDP_df.location_name .== country),:]
		GDP_year_data = GDP_estimates[(GDP_estimates.year .== year),:]
		if nrow(GDP_year_data) == 0
			println("Warning: No GDP data for $country in $year, skipping...")
			continue
		else
			GDP_est = GDP_year_data.real_gdp_usd[1]
		end

		# per capita
		N = sum(pop_structure.population)
		GDP_pc = GDP_est/N
		# Compute target VSL
		VSL_target = VSL_ref*(GDP_pc/Y_ref)^η

		# Fertility projections
		fert_country = fert_df[fert_df.Country.== country,:]
		all_years = Int64.(minimum(fert_country.Year_low):(maximum(fert_country.Year_high)-1))
		all_values = repeat(fert_country.Fertility_Rate, inner = 5)./5
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
			setfield!(bio_pars, :α,
				find_zero(function(x) temp_pars=deepcopy(bio_pars);
				setfield!(temp_pars, :α, x);
				HLE_current - HLE(temp_pars, opts, aa = 0.0) end,
				getfield(bio_pars, :α)))

		# Calculate dLE for param
		try
			δ2_final = find_zero(function(x) temp_pars=deepcopy(bio_pars);
				setfield!(temp_pars, :δ2, x);
				LE_current + ii - LE(temp_pars, opts, aa = 0.0) end, 1.0)
		catch e
			if isa(e, DomainError) || isa(e, Roots.ConvergenceFailed)
				println("Warning: Parameter calibration failed for $country in $year: $e")
				println("Using default δ2_final = 0.8")
				δ2_final = 0.8
			else
				rethrow(e)
			end
		end

			# Set WageChild to match target VSL
			age_range = Int.(1:nrow(pop_structure))[
				(pop_structure.age .> 19) .& (pop_structure.age .< 24)]
			try
				setfield!(econ_pars, :WageChild,
					find_zero(function(x) temp_pars=deepcopy(econ_pars);
					setfield!(temp_pars, :WageChild, x);
					temp_df = define_vars_df(bio_pars, temp_pars, opts);
					temp_df = innerjoin(pop_structure[:,[:age, :population]],temp_df, on = :age);
					soc_V = (temp_df.population[age_range]).*temp_df.V[age_range];
					VSL_target - sum(soc_V)/sum(temp_df.population[age_range]) end,
					6.9))
			catch e
				if isa(e, Roots.ConvergenceFailed) || isa(e, DomainError)
					println("Warning: VSL calibration failed for $country in $year: $e")
					println("Using default WageChild")
					# Keep the default WageChild value
				else
					rethrow(e)
				end
			end
		# Solve model and compute social WTP
		try
			vars_df = define_vars_df(bio_pars, econ_pars, opts)
			vars_df = compute_wtp_phasing(bio_pars, econ_pars, opts, vars_df, δ2_final, gradual = gradual, country = country, year = year)
			plot(vars_df.WTP)
			savefig(export_folder * "wtp_by_age_"*string(country)*"_"*string(year)*"_julia.pdf")
			# Merge in population data
			soc_df = vars_df[in.(vars_df.age, [pop_structure.age]), :]
			soc_df = innerjoin(pop_structure[:,[:age, :population]],
				soc_df[:,[:age, :WTP_S, :WTP_H, :WTP]], on = :age)
			# Compute social WTP
			total_WTP = sum(soc_df.population .* soc_df.WTP)
			average_WTP = total_WTP/N
			# Unborn WTP
			unborn_WTP = sum(vars_df.WTP[1].*fert_projs.births.*vars_df.discount[2:(nrow(fert_projs)+1)])

			# Populate row
			WTP_table.country[rowid] = country
			WTP_table.year[rowid] = year
			WTP_table.le[rowid] = round(LE_current, digits = 1)
			WTP_table.hle[rowid] = round(HLE_current, digits = 1)
			WTP_table.population[rowid] = round(N/1000000, digits = 2)
			WTP_table.real_gdp_pc[rowid] = Int(round(GDP_pc))
			WTP_table.vsl[rowid] = round(VSL_target/1000000, digits = 3)
			WTP_table.wtp[rowid] = round(total_WTP/trillion, digits = 3)
			WTP_table.wtp_pc[rowid] = round(average_WTP/1000, digits = 3)
			WTP_table.wtp_0[rowid] = round(vars_df.WTP[1]/1000, digits = 3)
			WTP_table.wtp_unborn[rowid] = round(unborn_WTP/trillion, digits = 3)
			# Progress report
			println(string(country)*" social WTP for extra year with "*string(year)*" population is "*
				string(round(total_WTP/trillion,digits = 2))*" trillion USD")
		catch e
			if isa(e, DomainError)
				println("Warning: Domain error in model computation for $country in $year: $e")
				println("Skipping this country-year combination")
				# Fill in the table with zeros
				WTP_table.Country[rowid] = country
				WTP_table.Year[rowid] = year
				WTP_table.LE[rowid] = round(LE_current, digits = 1)
				WTP_table.HLE[rowid] = round(HLE_current, digits = 1)
				WTP_table.Pop[rowid] = round(N/1000000, digits = 2)
				WTP_table.GDP_pc[rowid] = Int(round(GDP_pc))
				WTP_table.VSL[rowid] = round(VSL_target/1000000, digits = 3)
				WTP_table.WTP_1y[rowid] = 0.0
				WTP_table.WTP_avg[rowid] = 0.0
				WTP_table.WTP_0[rowid] = 0.0
				WTP_table.WTP_unborn[rowid] = 0.0
				println(string(country)*" social WTP for extra year with "*string(year)*" population is 0.0 trillion USD (computation failed)")
			else
				rethrow(e)
			end
		end
		# Add increment
		rowid += 1

		end
	end
end




"""
Export table
"""

WTP_table.country = Symbol.(WTP_table.country)
CSV.write("output/social_wtp_table.csv", WTP_table)
