"""
Create a health variable from GBD YLD data
"""
function YLD_to_health(bio_pars::BiologicalParameters, YLD_dist::Array{Float64,1}, ages)
	# Convert YLD to "health"
	health_dist = 1 .- YLD_dist
	# Linear interpolation with extrapolation
	interp_linear = LinearInterpolation(ages, health_dist, extrapolation_bc=Line())
	# Constrain to be positive
	health_out = max.(interp_linear(Float64.(0:bio_pars.MaxAge)), [0.1])

	return health_out
end


"""
Create a survivor rate variable from GBD Death rate data
"""
function Deaths_to_survivor(bio_pars::BiologicalParameters, death_dist::Array{Float64,1}, ages)
	# Interpolate mortality first (chance of dying in a year, given you're in that bracket)
	interp_linear = LinearInterpolation(ages, death_dist, extrapolation_bc=Line())
	# Constrain to be at most 1
	ages_dist = Float64.(0:bio_pars.MaxAge)
	death_out = min.(interp_linear(Float64.(0:bio_pars.MaxAge)), [1.0])
	# Convert to survival first
	survive_dist = ones(length(ages_dist))
	for ii in 2:length(survive_dist)
		survive_dist[ii] = survive_dist[ii-1]*(1 - death_out[ii-1])
	end
	# Make sure last period is zero
	survive_dist[end] = 0.0

	return survive_dist
end


"""
Define bio_data in correct format for vars_df from given survival, heatlh and population
"""
function define_bio_data(bio_pars::BiologicalParameters, S_old::Array{Float64,1}, S_new::Array{Float64,1},
	H_old::Array{Float64,1}, H_new::Array{Float64,1}, pop_structure::DataFrame)
	# Define changes in S and H across the age distribution
	S_delta = S_new - S_old
	H_delta = H_new - H_old
	# Create DataFrame
	bio_data = DataFrame(age = 0:bio_pars.MaxAge, S = S_old, S_new = S_new, Sgrad = S_delta,
		H = H_old, H_new = H_new, Hgrad = H_delta, dH = 0.0)
	bio_data.dH[2:end]=(bio_data.H[2:end]-bio_data.H[1:(end-1)])./bio_data.H[1:(end-1)]
	# Merge with population structure
	bio_data = leftjoin(bio_data, pop_structure, on = :age)
	sort!(bio_data, order(:age, rev=false))
	bio_data.population[ismissing.(bio_data.population)] .= 0.0

	return bio_data

end



"""
Create vars_df from options and observed bio_data (which can be created with define_bio_data)
"""
function solve_econ_from_biodata(bio_pars::BiologicalParameters, econ_pars::EconomicParameters,
	opts, bio_data::DataFrame)
	# Define some economic variables
	econ_vars = [:age, :C, :Cdot, :L, :SL, :W, :dW, :Y, :A, :Sav, :discount,
		:Wgrad, :Wacc, :Z, :zc, :uc, :ul, :V, :v, :WTP, :WTP_H, :WTP_S, :WTP_W]
	econ_df = DataFrame(zeros(Int(bio_pars.MaxAge+1), length(econ_vars)), econ_vars)
	# Age and discount rate (fixed)
	econ_df.age = Float64.(0:bio_pars.MaxAge)
	econ_df.discount = discount(econ_pars, econ_df.age)
	# Import US consumption data for pre-work consumption
	econ_df.C .= 1.0
	econ_df.C[1:opts.AgeGrad+1] = vcat(USc[1], USc[1:opts.AgeGrad])
	# Pre graduation all time is leisure (seems a bit problematic haha)
	econ_df.L .= econ_pars.MaxHours
	econ_df.Z[1:opts.AgeGrad+1] = compute_Z(econ_pars, econ_df[1:opts.AgeGrad+1,:])
	econ_df.SL[1:opts.AgeGrad+1] = compute_sL(econ_pars, econ_df[1:opts.AgeGrad+1,:])

	# Combine econ and bio vars into one df
	vars_df = innerjoin(bio_data, econ_df, on = :age)

	# Define wage profile
	vars_df.W = wage.([bio_pars], [econ_pars], [opts], vars_df.age)
	vars_df.dW[2:end]=(vars_df.W[2:end].-vars_df.W[1:end-1])./vars_df.W[1:end-1]
	# Solve HH problem
	vars_df = solve_HH(bio_pars, econ_pars, opts, vars_df)
	# Can recompute z0 with new consumption and leisure
	if opts.redo_z0
		econ_pars.z0 = compute_z0(econ_pars, vars_df)
	end

	# Calculate composite variable and value of year
	vars_df.Z = compute_Z(econ_pars, vars_df)
	vars_df.zc = compute_zc(econ_pars, vars_df)
	vars_df.uc = compute_uc(econ_pars, vars_df)
	vars_df.ul = compute_ul(econ_pars, vars_df)
	vars_df.v = compute_v(econ_pars, vars_df)
	vars_df.q = compute_q(econ_pars, vars_df)

	# Calculate (remaining) value of statistical life (VSL)
	for kk in Int.(1:bio_pars.MaxAge+1)
		rows = kk:Int(bio_pars.MaxAge+1)
		vars_df.V[kk]=sum(vars_df.v[rows].*vars_df.S[rows].* vars_df.discount[rows]) *
			vars_df.S[kk]^-1 * vars_df.discount[kk]^-1
		if (isnan(vars_df.V[kk]) | isinf(vars_df.V[kk]))
			vars_df.V[kk] = 0.
		end

	end

	return vars_df
end



"""
Compute WTP variables from vars_df with empirical Sgrad and Hgrad
"""
function compute_wtp_from_biodata(bio_pars::BiologicalParameters, econ_pars::EconomicParameters,
	opts, vars_df::DataFrame)
	# Initialise WTP
	vars_df[:,[:WTP, :WTP_S, :WTP_H, :WTP_W]] .=  0.
	# For each age we need to recompute Sgrad(a) given that they have made it this far
	for kk in Int.(1:bio_pars.MaxAge+1)
		rows = kk:Int(bio_pars.MaxAge+1)
		# Sgrad with age a as starting point.
		if all(vars_df.Sgrad[rows] .== 0.)
			Sa_grad = vars_df.Sgrad[rows]
		else
			Sa = vars_df.S[kk]
			Sa_new = vars_df.S_new[kk]
			Sa_grad = vars_df.S_new[rows]/Sa_new - vars_df.S[rows]/Sa
		end
		# WTP for survival
		vars_df.WTP_S[kk]=sum(vars_df.v[rows].*Sa_grad.*
			vars_df.discount[rows])*vars_df.discount[kk]^-1
		if (isnan(vars_df.WTP_S[kk]) | isinf(vars_df.WTP_S[kk]))
			vars_df.WTP_S[kk] = 0.
		end
		# WTP for health
		vars_df.WTP_H[kk]=sum(vars_df.q[rows].*vars_df.S[rows].*vars_df.discount[rows]).*
			vars_df.S[kk]^-1*vars_df.discount[kk]^-1
		if (isnan(vars_df.WTP_H[kk]) | isinf(vars_df.WTP_H[kk]))
			vars_df.WTP_H[kk] = 0.
		end
		# WTP for wage (part of WTP_H)
		vars_df.WTP_W[kk]=sum(vars_df.Wgrad[rows].*(econ_pars.MaxHours .- vars_df.L[rows]).*
			vars_df.S[rows].*vars_df.discount[rows])*
			vars_df.S[kk]^-1*vars_df.discount[kk]^-1
		if (isnan(vars_df.WTP_W[kk]) | isinf(vars_df.WTP_W[kk]))
			vars_df.WTP_W[kk] = 0.
		end
	end
	vars_df.WTP = vars_df.WTP_S + vars_df.WTP_H

	return vars_df
end


"""
Get full vars_df from original and new mort_df and health_df
"""
function vars_df_from_biodata(bio_pars::BiologicalParameters, econ_pars::EconomicParameters, opts,
	mort_orig::DataFrame, mort_new::DataFrame, health_orig::DataFrame,
	health_new::DataFrame, pop_structure::DataFrame, VSL_target::Float64)

	# Define H, S and H_delta/S_delta
	S_old = Deaths_to_survivor(bio_pars, mort_orig.Total, mort_orig.age)
	S_new = Deaths_to_survivor(bio_pars, mort_new.Total, mort_new.age)
	H_old = YLD_to_health(bio_pars, health_orig.Total, health_orig.age)
	H_new = YLD_to_health(bio_pars, health_new.Total, health_new.age)

	# Combine in to bio_data
	bio_data = define_bio_data(bio_pars, S_old, S_new, H_old, H_new, pop_structure[:,[:age, :population]])

	# Set WageChild to match target VSL
	try
		setfield!(econ_pars, :WageChild,
			find_zero(function(x) temp_pars=deepcopy(econ_pars);
			setfield!(temp_pars, :WageChild, x);
			VSL_target - mean(solve_econ_from_biodata(bio_pars, temp_pars, opts, bio_data).V[51]) end,
			6.9))
	catch
		try
			setfield!(econ_pars, :WageChild,
				find_zero(function(x) temp_pars=deepcopy(econ_pars);
				setfield!(temp_pars, :WageChild, abs(x));
				VSL_target - mean(solve_econ_from_biodata(bio_pars, temp_pars, opts, bio_data).V[51]) end,
				0.42))
		catch
			try
				setfield!(econ_pars, :WageChild,
					find_zero(function(x) temp_pars=deepcopy(econ_pars);
					setfield!(temp_pars, :WageChild, x);
					VSL_target - mean(solve_econ_from_biodata(bio_pars, temp_pars, opts, bio_data).V[51]) end,
					(0.5, 0.6)))
			catch
				try
					econ_pars = EconomicParameters()
					setfield!(econ_pars, :WageChild,
						find_zero(function(x) temp_pars=deepcopy(econ_pars);
						setfield!(temp_pars, :WageChild, x);
						VSL_target - mean(solve_econ_from_biodata(bio_pars, temp_pars, opts, bio_data).V[51]) end,
						(3.0, 3.4)))
				catch
					econ_pars = EconomicParameters()
					setfield!(econ_pars, :WageChild,
						find_zero(function(x) temp_pars=deepcopy(econ_pars);
						setfield!(temp_pars, :WageChild, x);
						VSL_target - mean(solve_econ_from_biodata(bio_pars, temp_pars, opts, bio_data).V[51]) end,
						(0.2, 10.0)))
				end
			end
		end
	end
	#econ_pars.WageChild = 0.8#abs(econ_pars.WageChild)
	#solve_econ_from_biodata(bio_pars, econ_pars, opts, bio_data).V[51]
	#VSL_target
	# Solve for econ variables
	vars_df = solve_econ_from_biodata(bio_pars, econ_pars, opts, bio_data)
	# SOlve for WTP variables
	vars_df = compute_wtp_from_biodata(bio_pars, econ_pars, opts, vars_df)

	return vars_df
end



"""
Function to reduce diseases by given percentage (default 10%) in GBD data
"""
function disease_reduction(df_orig::DataFrame, diseases::Array{String,1}; factor::Float64 = 0.9,
		age_start::Int = 0, excl_cols = [["location_name", "age_name", "age", "year", "population", "Total"]])

	# Initialise new variables
	df_new = deepcopy(df_orig)
	# Reduce (each) diseases by 10%
	for disease in diseases
		df_new[df_new.age .>= age_start,disease] .*= factor
	end
	# Sum up the new total S and H
	all_diseases = names(df_new)[.!in.(names(df_new), excl_cols)]
	df_new.Total = sum.(eachrow(df_new[:,all_diseases]))

	return df_new

end



"""
Function to calculate benefits to a specific age group
"""
function age_group_benefits(bio_pars::BiologicalParameters, econ_pars::EconomicParameters, opts,
	mort_orig::DataFrame, mort_new::DataFrame, health_orig::DataFrame,
	health_new::DataFrame, pop_structure::DataFrame, VSL_target::Float64;
	age_start::Int = 0)

	# Define H, S and H_delta/S_delta
	S_old = Deaths_to_survivor(bio_pars, mort_orig.Total, mort_orig.age)
	S_new = Deaths_to_survivor(bio_pars, mort_new.Total, mort_new.age)
	H_old = YLD_to_health(bio_pars, health_orig.Total, health_orig.age)
	H_new = YLD_to_health(bio_pars, health_new.Total, health_new.age)

	# Combine in to bio_data
	bio_data = define_bio_data(bio_pars, S_old, S_new, H_old, H_new, pop_structure[:,[:age, :population]])
	# Set WageChild to match target VSL
	setfield!(econ_pars, :WageChild,
		find_zero(function(x) temp_pars=deepcopy(econ_pars);
		setfield!(temp_pars, :WageChild, x);
		VSL_target - mean(solve_econ_from_biodata(bio_pars, temp_pars, opts, bio_data).V[51]) end,
		6.9))
	# Solve for econ variables
	vars_df = solve_econ_from_biodata(bio_pars, econ_pars, opts, bio_data)

	# COmpute WTP, just for a 75 year old
	vars_df[:,[:WTP, :WTP_S, :WTP_H, :WTP_W]] .=  0.

	rows = (age_start+1):Int(bio_pars.MaxAge+1)
	# Sgrad with age_start as starting point.
	Sa = vars_df.S[(age_start+1)]
	Sa_new = vars_df.S_new[(age_start+1)]
	Sa_grad = vars_df.S_new[rows]/Sa_new - vars_df.S[rows]/Sa
	discounting = vars_df.discount[rows]./vars_df.discount[(age_start+1)]
	# WTP for survival
	vars_df.WTP_S[rows] = vars_df.v[rows].*Sa_grad.*discounting
	# WTP for health
	vars_df.WTP_H[rows] = vars_df.q[rows].*vars_df.S[rows].*discounting./Sa
	# Combined
	vars_df.WTP = vars_df.WTP_S + vars_df.WTP_H

	return vars_df
end
