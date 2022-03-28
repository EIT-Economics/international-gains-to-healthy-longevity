"""
US Empirical WTP from baseline Murphy and Topel model with observed H and S curves
"""

if occursin("jashwin", pwd())
    cd("C://Users/jashwin/Documents/GitHub/international-gains-to-healthy-longevity/")
else
    cd("/Users/julianashwin/Documents/GitHub/international-gains-to-healthy-longevity/")
end

using Statistics, Parameters, DataFrames
using QuadGK, NLsolve, Roots, FiniteDifferences, Interpolations
using Plots, XLSX, ProgressMeter, Formatting, TableView, Latexify, LaTeXStrings

# Import functions
include("src/TargetingAging.jl")

# Plotting backend
gr()

# Choose a country
country = "United States of America"

"""
Import data
"""
# Import GDP data
GDP_df = CSV.read("data/GBD/real_gdp_data.csv", DataFrame, ntasks = 1)
GDP_df.real_gdp_usd[(GDP_df.real_gdp_usd .== "NA")] .= "NaN"
GDP_df.real_gdp = parse.([Float64], GDP_df.real_gdp_usd)
GDP_df = GDP_df[(GDP_df.location_name .== country),:]
# Import empirical mortality curve by cause
mort_df = CSV.read("data/GBD/mortality_data.csv", DataFrame, ntasks = 1)
mort_df = mort_df[(mort_df.location_name .== country),:]
# Import empirical disability curve by cause
health_df = CSV.read("data/GBD/health_data.csv", DataFrame, ntasks = 1)
health_df = health_df[(health_df.location_name .== country),:]
# Import historical mortality rates
hist_mort_df = CSV.read("data/USA_life.csv", DataFrame, ntasks = 1)
hist_mort_df.Total = hist_mort_df.mx
hist_mort_df = hist_mort_df[:,[:year, :age, :Total]]
hist_pop_df = CSV.read("data/USA_population.csv", DataFrame, ntasks = 1)
hist_pop_df.population = hist_pop_df.Total
hist_pop_df = hist_pop_df[:,[:year, :age, :population]]


# Baseline VSL for US
VSL_ref = 11.5e6
# Baseline GDP per capita for US 2019
Y_ref = 60901.91981821335
# Income elasticity of VSL
η = 1.0

# A couple useful variables
trillion = 1e12
export_folder = "figures/Olshansky_plots/"


"""
Define options and parameters
"""
# Biological model
bio_pars = BiologicalParameters()
# Economic model
econ_pars = EconomicParameters()
# Options (slimmed down)
opts = (param = :none, bio_pars0 = deepcopy(bio_pars), AgeGrad = 20, AgeRetire = 65,
	redo_z0 = false, prod_age = false, age_start = 0, no_compress = false,
    Wolverine = 0, phasing = false)


"""
Compare 2010 to 2019
"""
# Define population structure in base year
pop_structure = mort_df[mort_df.year .== 2010,:]
pop_structure = pop_structure[:,[:age,:population]]
# Compute target VSL in base year
GDP_est = GDP_df[(GDP_df.year .== 2010),:real_gdp][1]
GDP_pc = GDP_est/GDP_df[(GDP_df.year .== 2010),:population][1]
VSL_target = VSL_ref*(GDP_pc/Y_ref)^η
# Select/create the mort_df and health_df to compare
mort_orig = mort_df[mort_df.year .==2010, :]
mort_new = mort_df[mort_df.year .==2019, :]
health_orig = health_df[health_df.year .==2010, :]
health_new = health_df[health_df.year .==2019, :]

# Original life cycle
vars_df_orig = vars_df_from_biodata(bio_pars, econ_pars, opts, mort_orig, mort_orig,
	health_orig, health_orig, pop_structure, VSL_target)
μ_orig = compute_μ(bio_pars, econ_pars, opts, vars_df_orig)

U_orig = compute_U.([bio_pars], [econ_pars], [opts], [vars_df_orig], vars_df_orig.age,
	reset_bio = false)
LE_orig = sum(vars_df_orig.S)
HLE_orig = sum(vars_df_orig.S.*vars_df_orig.H)

# New life cycle
vars_df_new = vars_df_from_biodata(bio_pars, econ_pars, opts, mort_new, mort_new,
	health_new, health_new, pop_structure, VSL_target)
μ_new = compute_μ(bio_pars, econ_pars, opts, vars_df_new)

U_new = compute_U.([bio_pars], [econ_pars], [opts], [vars_df_new], vars_df_orig.age,
	reset_bio = false)
LE_new = sum(vars_df_new.S)
HLE_new = sum(vars_df_new.S.*vars_df_new.H)

WTP_diff = (U_new .- U_orig)./μ_orig
WTP_diff[isnan.(WTP_diff)] .= 0.0


total_WTP = round(sum(vars_df_orig.population.*WTP_diff)/trillion, digits = 1)


WTP_diff_plt = WTP_diff[vars_df_orig.age .<= 110]
ages = vars_df_orig.age[vars_df_orig.age .<= 110]
plot(ages, WTP_diff_plt, title = "WTP for changes from 2010-2019", legend = :bottomright,
	label = "WTP: "*string(total_WTP)*" trillion USD", xlabel = "Age", ylabel = "WTP")
#plot!(ages, WTP_plt_df.WTP_S, title = "WTP for changes from 2010-2019",
#	label = "WTP (S): "*string(total_WTP_S)*" trillion USD")
#plot!(ages, WTP_plt_df.WTP_H, title = "WTP for changes from 2010-2019",
#	label = "WTP (H): "*string(total_WTP_H)*" trillion USD")
plot!(size = (500, 300))
savefig("figures/Olshansky_plots/WTP_2010vs2019.pdf")



"""
Compare reductions in disease
"""
# Define diseases to reduce
diseases = ["Cardiovascular diseases", "Neoplasms", "Chronic respiratory diseases",
	"Neurological disorders", "Diabetes and kidney diseases"]
#diseases = ["Cardiovascular diseases", "Neurological disorders"]
# Baseline year values
year = 2019
mort_orig = mort_df[mort_df.year .== 2019, :]
health_orig = health_df[health_df.year .== 2019, :]
# Define population structure in base year
pop_structure = mort_df[mort_df.year .== 2019,:]
pop_structure = pop_structure[:,[:age,:population]]
# Compute target VSL in base year
GDP_est = GDP_df[(GDP_df.year .== 2019),:real_gdp][1]
GDP_pc = GDP_est/GDP_df[(GDP_df.year .== 2019),:population][1]
VSL_target = VSL_ref*(GDP_pc/Y_ref)^η

# Original life cycle
vars_df_orig = vars_df_from_biodata(bio_pars, econ_pars, opts, mort_orig, mort_orig,
	health_orig, health_orig, pop_structure, VSL_target)
μ_orig = compute_μ(bio_pars, econ_pars, opts, vars_df_orig)

U_orig = compute_U.([bio_pars], [econ_pars], [opts], [vars_df_orig], vars_df_orig.age,
	reset_bio = false)
LE_orig = sum(vars_df_orig.S)
HLE_orig = sum(vars_df_orig.S.*vars_df_orig.H)



starting_ages = [0, 30, 50, 65, 75, 80, 90]
# Factor by which mortality/disability are transformed
factor = 0.5
# Age at which disease improves
age_start = 65

# Initialise WTP at birth table
WTP_birth_vars = vcat(["Starting Age", "Factor"], diseases, ["Sum of separate effects", "Total effect"])
WTP_birth_df = DataFrame(zeros(length(starting_ages), length(WTP_birth_vars)), WTP_birth_vars)
WTP_birth_df[:,"Starting Age"] = starting_ages
WTP_birth_df[:,"Factor"] .= factor

## Loop over staring ages
prog = Progress(length(starting_ages), desc = "Assessing for different starting ages: ")
for age_start in starting_ages
	# Initialise DataFrame
	#WTP_reduction_vars = vcat(["age"], diseases.*" (S)", diseases.*" (H)", diseases,
	#	["Sum of separate effects (S)", "Sum of separate effects (H)", "Sum of separate effects",
	#	"Total effect (S)", "Total effect (H)", "Total effect"])
	WTP_reduction_vars = vcat(["age", "population"], diseases, ["Sum of separate effects", "Total effect"])
	WTP_reduction_df = DataFrame(zeros(Int(bio_pars.MaxAge+1), length(WTP_reduction_vars)), WTP_reduction_vars)
	WTP_reduction_df.age = Float64.(0:bio_pars.MaxAge)
	WTP_reduction_df.population = vars_df_orig.population


	# Go through each individual disease
	obs = 1:111
	fig1 = plot(layout = (2,2), size = (800,800))
	fig1 = plot!(vars_df_orig.age[obs], vars_df_orig.S[obs], subplot =1, legend = false)
	fig1 = plot!(vars_df_orig.age[obs], vars_df_orig.H[obs], subplot = 3, legend = false)
	fig1 = plot!(vars_df_orig.age[obs], vars_df_orig.WTP_S[obs], subplot = 2, legend = false)
	fig1 = plot!(vars_df_orig.age[obs], vars_df_orig.WTP_H[obs], subplot = 4, label = "Baseline")
	#fig1 = plot!(vars_df_orig.age[obs], vars_df_orig.WTP_H[obs], subplot = 4, legend = false)
	for disease in diseases
		mort_new = disease_reduction(mort_orig, [disease], factor = factor, age_start = age_start)
		health_new = disease_reduction(health_orig, [disease], factor = factor, age_start = age_start)

		#vars_df = age_group_benefits(bio_pars, econ_pars, opts, mort_orig, mort_new,
		#	health_orig, health_new, pop_structure, VSL_target, age_start = age_start)

		# New life cycle
		vars_df_new = vars_df_from_biodata(bio_pars, econ_pars, opts, mort_new, mort_new,
			health_new, health_new, pop_structure, VSL_target)
		U_new = compute_U.([bio_pars], [econ_pars], [opts], [vars_df_new], vars_df_orig.age,
			reset_bio = false)
		LE_new = sum(vars_df_new.S)
		HLE_new = sum(vars_df_new.S.*vars_df_new.H)

		WTP_diff = (U_new .- U_orig)./μ_orig
		WTP_diff[isnan.(WTP_diff)] .= 0.0

		total_WTP = round(sum(vars_df_orig.population.*WTP_diff)/trillion, digits = 1)

		plot(fig1)
		fig1 = plot!(vars_df_orig.age[obs], vars_df_new.S[obs], subplot = 1)
		fig1 = plot!(vars_df_orig.age[obs], vars_df_new.H[obs], subplot = 3)
		fig1 = plot!(vars_df_orig.age[obs], WTP_diff[obs], subplot = 2)
		fig1 = plot!(vars_df_orig.age[obs], vars_df_new.WTP[obs], subplot = 4,
			label = disease*" (USD "*string(round(sum(vars_df_orig.population.*WTP_diff)/trillion, digits = 1))*" trillion)")
		#fig1 = plot!(vars_df_orig.age[obs], vars_df_new.WTP[obs], subplot = 4)

		#WTP_reduction_df[:,disease*" (S)"] = vars_df.WTP_S
		#WTP_reduction_df[:,disease*" (H)"] = vars_df.WTP_H
		WTP_reduction_df[:,disease] = WTP_diff #vars_df.WTP

		WTP_birth_df[(WTP_birth_df[:,"Starting Age"] .== age_start), disease] .= WTP_diff[1]
	end
	plot(fig1)

	## Add them up for Sum of Separate Effects
	WTP_diff = sum.(eachrow(WTP_reduction_df[:,diseases]))
	WTP_birth_df[(WTP_birth_df[:,"Starting Age"] .== age_start), "Sum of separate effects"] .= WTP_diff[1]
	WTP_reduction_df[:,"Sum of separate effects"] = WTP_diff
	plot(fig1)
	fig1 = plot!(vars_df_orig.age[obs], vars_df_new.WTP[obs], subplot = 1, title = "Survival")
	fig1 = plot!(vars_df_orig.age[obs], vars_df_new.WTP[obs], subplot = 3, title = "Health")
	fig1 = plot!(vars_df_orig.age[obs], WTP_diff[obs], color = "black", linestyle = :dot,
		title = "WTP", subplot = 2)
	fig1 = plot!(vars_df_orig.age[obs], vars_df_new.WTP[obs], subplot = 4, color = "black", linestyle = :dot,
		label = "Sum of effects (USD "*string(round(sum(vars_df_orig.population.*WTP_diff)/trillion, digits = 1))*" trillion)")

	## Total effect
	mort_new = disease_reduction(mort_orig, diseases, factor = factor, age_start = age_start)
	health_new = disease_reduction(health_orig, diseases, factor = factor, age_start = age_start)
	# New life cycle
	vars_df_new = vars_df_from_biodata(bio_pars, econ_pars, opts, mort_new, mort_new,
		health_new, health_new, pop_structure, VSL_target)
	U_new = compute_U.([bio_pars], [econ_pars], [opts], [vars_df_new], vars_df_orig.age,
		reset_bio = false)
	LE_new = sum(vars_df_new.S)
	HLE_new = sum(vars_df_new.S.*vars_df_new.H)

	WTP_diff = (U_new .- U_orig)./μ_orig
	WTP_diff[isnan.(WTP_diff)] .= 0.0
	WTP_birth_df[(WTP_birth_df[:,"Starting Age"] .== age_start), "Total effect"] .= WTP_diff[1]

	WTP_reduction_df[:,"Total effect"] = WTP_diff

	plot(fig1)
	fig1 = plot!(vars_df_orig.age[obs], vars_df_new.S[obs], color = "black", linestyle = :dash, subplot = 1)
	fig1 = plot!(vars_df_orig.age[obs], vars_df_new.H[obs], color = "black", linestyle = :dash, subplot = 3)
	fig1 = plot!(vars_df_orig.age[obs], WTP_diff[obs], color = "black", linestyle = :dash, subplot = 2)
	fig1 = plot!(vars_df_orig.age[obs], vars_df_new.WTP[obs], subplot = 4, color = "black", linestyle = :dash,
		label = "Total effect (USD "*string(round(sum(vars_df_orig.population.*WTP_diff)/trillion, digits = 1))*" trillion)")
	plot(fig1)

	WTP_reduction_df[1,vcat(diseases, ["Sum of separate effects", "Total effect"])]

	next!(prog)

end

WTP_birth_df.Ratio = WTP_birth_df[:,"Total effect"]./WTP_birth_df[:,"Sum of separate effects"]


CSV.write(export_folder*"WTP_birth_comorbidity.csv", WTP_birth_df)



"""
Constant health and VSL, various historical mortalities
"""
# Age at which to start slowdown
age_start = 50
factor = 0.9
# Set to default values
bio_pars = BiologicalParameters()
econ_pars = EconomicParameters()
ages_old = 0:bio_pars.MaxAge
ages_new = vcat(ages_old[1:age_start], (age_start .+
	factor.*(ages_old[(age_start+1):end] .- age_start)))
plot(ages_old); plot!(ages_new)
H_old = health.([bio_pars], [opts], ages_old)
H_new = health.([bio_pars], [opts], ages_new)
S_default = survivor.([bio_pars], [opts], [0.0], ages_new)
plot(H_old); plot!(H_new)
# Set up table
years = unique(hist_mort_df.year)
results_vars = [:year, :population, :wtp_diffS, :wtp_diffpop]
results_df = DataFrame(zeros(length(years), length(results_vars)), results_vars)
results_df.year = years

prog = Progress(nrow(results_df), desc = "Running through countries and years: ")
for ii in 1:nrow(results_df)
	# Define year
	year = results_df.year[ii]
	###
	# Extract the relevant data
	###
	# Population structure
	pop_structure = hist_pop_df[(hist_pop_df.year .== year),:]
	results_df.population[ii] = sum(pop_structure.population)
	# Mortality rates
	mort_orig = hist_mort_df[(hist_mort_df.year .== year), :]
	mort_new = hist_mort_df[(hist_mort_df.year .== year), :]

	## Changing S over time, WTP at birth
	S_old = Deaths_to_survivor(bio_pars, mort_orig.Total, mort_orig.age)
	S_new = Deaths_to_survivor(bio_pars, mort_new.Total, mort_new.age)
	# Solve the model and generate vars_df
	bio_data = define_bio_data(bio_pars, S_old, S_new, H_old, H_new, pop_structure[:,[:age, :population]])
	vars_df = solve_econ_from_biodata(bio_pars, econ_pars, opts, bio_data)
	vars_df = compute_wtp_from_biodata(bio_pars, econ_pars, opts, vars_df)
	results_df.wtp_diffS[ii] = vars_df.WTP[1]

	## Change pop structure
	S_old = deepcopy(S_default)
	S_new = deepcopy(S_default)
	# Solve the model and generate vars_df
	bio_data = define_bio_data(bio_pars, S_old, S_new, H_old, H_new, pop_structure[:,[:age, :population]])
	vars_df = solve_econ_from_biodata(bio_pars, econ_pars, opts, bio_data)
	vars_df = compute_wtp_from_biodata(bio_pars, econ_pars, opts, vars_df)
	results_df.wtp_diffpop[ii] = sum(vars_df.population .* vars_df.WTP)

	next!(prog)
end

# Plot WTP at birth as S shifts over time
plot(results_df.year, results_df.wtp_diffS./1000, label = false,
	ylabel = "Thousand US dollars (2019)", xlabel = "Year", size = (400, 350))
	#title = "WTP(0) for 10% slowed aging with historic S")
savefig(export_folder*"US_diffS.pdf")

# Plot WTP per capita as population structure shifts over time
plot(results_df.year, results_df.wtp_diffpop./results_df.population./1000, label = false,
	ylabel = "Thousand US dollars (2019)", xlabel = "Year", size = (400, 350))
	#title = "WTP p.c. for 10% slowed aging with historic population")
savefig(export_folder*"US_diffpop.pdf")

# Plot total WTP per capita as population structure shifts over time
plot(results_df.year, results_df.wtp_diffpop, label = false,
	ylabel = "2019 US dollars", xlabel = "Year", size = (700, 350),
	title = "Total WTP for 10% slowed aging with historic population")
savefig(export_folder*"US_diffpop_total.pdf")


CSV.write(export_folder*"US_timeseries.csv", results_df)
