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

# Filter for specific countries and years (matching Python analysis.py DEFAULT_COUNTRIES)
target_countries = [
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
target_years = 1990:2020


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
# Filter for target countries and years only
countries = filter(c -> c ∈ target_countries, unique(GDP_df.location_name))
years = filter(y -> y ∈ target_years, unique(GDP_df.year))
last_year = maximum(years) # can't work out health benefit here as no previous year

println("Analyzing $(length(countries)) countries: $(join(countries, ", "))")
println("Analyzing $(length(years)) years: $(minimum(years))-$(maximum(years))")
println("Total country-year combinations: $(length(countries) * length(years))")

results_df = DataFrame(country = repeat(countries, inner = length(years)),
	year = repeat(years, length(countries)))
results_vars = [:population, :real_gdp, :real_gdp_pc, :le, :hle,
	:vsl, :wtp_s, :wtp_h, :wtp, :wtp_pc]
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
			econ_pars = EconomicParameters()
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
