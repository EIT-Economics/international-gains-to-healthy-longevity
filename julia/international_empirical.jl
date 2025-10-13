"""
Quantify gains from changes in S and H across countries and time
Based on baseline Murphy and Topel model with observed H and S curves
"""

using Statistics, Parameters, DataFrames
using QuadGK, NLsolve, Roots, FiniteDifferences, Interpolations
using Plots, XLSX, ProgressMeter, Formatting, Latexify, LaTeXStrings

# Import functions
include("TargetingAging.jl")

# Plotting backend
gr()

"""
Import data
"""
# Import GDP data
GDP_df = CSV.read("intermediate/WB/real_gdp_data.csv", DataFrame, ntasks = 1)
GDP_df.location_name = string.(GDP_df.location_name)


# Handle missing values - CSV.jl already parsed as Float64, just replace Missing with NaN
GDP_df.real_gdp_ppp = coalesce.(GDP_df.real_gdp_ppp, NaN)
GDP_df.real_gdp_usd = coalesce.(GDP_df.real_gdp_usd, NaN)
# FX rate
GDP_df.fx_rate = GDP_df.real_gdp_ppp./GDP_df.real_gdp_usd


# Import empirical mortality curve by cause
mort_df = CSV.read("intermediate/GBD/mortality_rates.csv", DataFrame, ntasks = 1)
# Define age as the midpoint between age_low and age_high
mort_df.age = (mort_df.age_low .+ mort_df.age_high) ./ 2


# Import empirical disability curve by cause
health_df = CSV.read("intermediate/GBD/morbidity_rates.csv", DataFrame, ntasks = 1)
# Define age as the midpoint between age_low and age_high
health_df.age = (health_df.age_low .+ health_df.age_high) ./ 2


# Population structure can be taken from mortality data
pop_df = mort_df[:,[:location_name, :year, :age, :population]]


# Baseline VSL for US
VSL_ref = 11.5e6
# Baseline GDP per capita for US 2019
Y_ref = 65349.35971265863
# Income elasticity of VSL
η = 1.0


# A couple useful variables
trillion = 1e12
export_folder = "output/"


"""
Parse command-line arguments for filtering
"""
# Default: Analyze all available countries and years
# Usage: julia international_empirical.jl [--countries "country1,country2"] [--years 1990:2000]

# Reference list of developed countries (matching Python DEFAULT_COUNTRIES)
DEFAULT_COUNTRIES = [
	"Australia",
	"France",
	"Germany",
	"Italy",
	"Japan",
	"Netherlands",
	"Spain",
	"Sweden",
	"United Kingdom",
	"United States of America"
]

DEFAULT_YEARS = 1990:1999

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
			error("--years requires a value (e.g., '1990:2000' or '1990,1995,2000')")
		end
	elseif ARGS[i] == "--default"
		# Use default countries and years (1990-1999)
		target_countries = DEFAULT_COUNTRIES
		target_years = DEFAULT_YEARS
		i += 1
	elseif ARGS[i] == "--help" || ARGS[i] == "-h"
		println("""
		Usage: julia international_empirical.jl [OPTIONS]
		
		Options:
		  --countries "country1,country2"    Filter for specific countries (comma-separated)
		  --years YEARS                      Filter for specific years (range: 1990:2000 or list: 1990,1995,2000)
		  --default                          Use default 10 countries and years 1990-1999
		  --help, -h                         Show this help message
		
		Examples:
		  # Analyze all available data (default)
		  julia international_empirical.jl
		  
		  # Analyze only USA and Japan for 2010-2020
		  julia international_empirical.jl --countries "United States of America,Japan" --years 2010:2020
		  
		  # Use default developed countries
		  julia international_empirical.jl --default
		  
		  # Specific years only
		  julia international_empirical.jl --years 1990,2000,2010
		""")
		exit(0)
	else
		error("Unknown argument: $(ARGS[i]). Use --help for usage information.")
	end
end


"""
Define options and parameters
"""
# Biological model
bio_pars = BiologicalParameters()
# Economic model
econ_pars = EconomicParameters()
# Options (slimmed down)
opts = (param = :none, bio_pars0 = deepcopy(bio_pars), AgeGrad = 20, AgeRetire = 65,
	redo_z0 = false, prod_age = false, age_start = 0, no_compress = false, Wolverine =0)


"""
Set up a table to fill
"""
# Filter for target countries and years (or use all available data)
# 1) Determine years first (based on GDP table, then we will check mortality/health availability)
year_range = 1990:2019
if target_years === nothing
	years = filter(y -> y in year_range, unique(GDP_df.year))
	println("Analyzing all available years (1990-2019): $(minimum(years))-$(maximum(years))")
else
	years = filter(y -> (y ∈ target_years) && (y in year_range), unique(GDP_df.year))
	println("Analyzing $(length(years)) years (limited to 1990-2019): $(minimum(years))-$(maximum(years))")
end

# 2) Determine countries available across GDP (non-missing real_gdp_usd), mortality, and health for the selected years
gdp_mask = .!isnan.(GDP_df.real_gdp_usd) .& in.(GDP_df.year, Ref(years))
gdp_countries = unique(GDP_df[gdp_mask, :location_name])

mort_mask = in.(mort_df.year, Ref(years))
mort_countries = unique(mort_df[mort_mask, :location_name])

health_mask = in.(health_df.year, Ref(years))
health_countries = unique(health_df[health_mask, :location_name])

available_countries = intersect(gdp_countries, mort_countries, health_countries)

# 3) Apply user filter if provided; otherwise use available countries only
if target_countries === nothing
	countries = available_countries
	println("Analyzing all countries with GDP and mortality/health data ($(length(countries)) total)")
else
	countries = filter(c -> c ∈ target_countries, available_countries)
	println("Analyzing $(length(countries)) countries (filtered and with data): $(join(countries, ", "))")
end

last_year = maximum(years) # can't work out health benefit here as no previous year

println("Total country-year combinations: $(length(countries) * length(years))")

results_df = DataFrame(country = repeat(countries, inner = length(years)),
	year = repeat(years, length(countries)))
results_vars = [:population, :real_gdp, :real_gdp_pc, :vsl, :le, :hle,
	 :wtp_s, :wtp_h, :wtp, :wtp_pc]
results_df = hcat(results_df, DataFrame(zeros(nrow(results_df), length(results_vars)), results_vars))


"""
Run through each country-year case
"""
# Pretty inefficient loop atm, but should do the job as there's only around 1,000 rows
prog = Progress(nrow(results_df), desc = "Running through countries and years: ")
for ii in 1:nrow(results_df)
	# Define country and year (should eventually nest this in separate steps)
	country = results_df.country[ii]
	year = results_df.year[ii]
	###
	# Extract the relevant data
	###
	# Population structure
	pop_structure = pop_df[(pop_df.year .== year) .& (pop_df.location_name .== country),:]
	sort!(pop_structure, :age)  # Ensure ages are sorted
	N = sum(pop_structure.population)
	# Compute target VSL in base year
	GDP_est = GDP_df[(GDP_df.year .== year) .& (GDP_df.location_name .== country),:real_gdp_usd][1]
	FX_1990 = GDP_df[(GDP_df.year .== 1990) .& (GDP_df.location_name .== country),:fx_rate][1]

	if !isnan(GDP_est)
		GDP_pc = GDP_est/N
		VSL_target = VSL_ref*(GDP_pc/Y_ref)^η
		# Base year mortality and health
		mort_orig = mort_df[(mort_df.year .== year) .& (mort_df.location_name .== country), :]
		sort!(mort_orig, :age)  # Ensure ages are sorted for interpolation
		health_orig = health_df[(health_df.year .== year) .& (health_df.location_name .== country), :]
		sort!(health_orig, :age)  # Ensure ages are sorted for interpolation
		# Next year mortality and health
		next_year = min(year+1, last_year)
		mort_new = mort_df[(mort_df.year .== next_year) .& (mort_df.location_name .== country), :]
		sort!(mort_new, :age)  # Ensure ages are sorted for interpolation
		health_new = health_df[(health_df.year .== next_year) .& (health_df.location_name .== country), :]
		sort!(health_new, :age)  # Ensure ages are sorted for interpolation
		
		# Skip if any dataframes are empty (no matching data)
		if nrow(mort_orig) > 0 && nrow(health_orig) > 0 && nrow(mort_new) > 0 && nrow(health_new) > 0
			# Solve the model and generate vars_df
			# econ_pars = EconomicParameters() # Removed this line as it is already defined at the top
			vars_df = vars_df_from_biodata(bio_pars, econ_pars, opts, mort_orig, mort_new,
				health_orig, health_new, pop_structure, VSL_target)
		else
			# Skip this iteration - no data available
			next!(prog)
			continue
		end
		# Populate the rest of the results_df row
		results_df.population[ii] = round(N/1e6, digits = 1)
		results_df.real_gdp[ii] = round(GDP_est/1e9, digits = 1)
		results_df.real_gdp_pc[ii] = round(GDP_pc/1e3,digits = 3)
		results_df.le[ii] = round(sum(vars_df.S), digits = 2)
		results_df.hle[ii] = round(sum(vars_df.S.*vars_df.H), digits = 2)
		results_df.vsl[ii] = round(VSL_target/1e6, digits = 3)
		results_df.wtp_s[ii] = round(sum(vars_df.population .* vars_df.WTP_S)/trillion, digits = 3)
		results_df.wtp_h[ii] = round(sum(vars_df.population .* vars_df.WTP_H)/trillion, digits = 3)
		results_df.wtp[ii] = round(sum(vars_df.population .* vars_df.WTP)/trillion, digits = 3)
		results_df.wtp_pc[ii] = round((sum(vars_df.population .* vars_df.WTP)/N)/1e3, digits = 3)
	else
		results_df[ii,results_vars] .= NaN
	end


	next!(prog)
end



CSV.write(export_folder*"international_comp.csv", results_df)
