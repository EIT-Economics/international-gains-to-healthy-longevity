"""
Define biological and economic variables in vars_df
"""
function define_vars_df(bio_pars::BiologicalParameters, econ_pars::EconomicParameters,
	opts)

	# Biological variables
	bio_vars = [:age, :F, :D, :H, :λ, :S, :Hgrad, :Hacc, :Sgrad, :Sacc, :dH]
	bio_df = DataFrame(zeros(Int(bio_pars.MaxAge+1), length(bio_vars)), bio_vars)
	bio_df.age = Float64.(0:bio_pars.MaxAge)

	# Fill in biological variables for baseline case
	bio_df.F = frailty.([bio_pars], [opts], bio_df.age)
	bio_df.D = disability.([bio_pars], [opts], bio_df.age)
	bio_df.H = health.([bio_pars], [opts], bio_df.age)
	bio_df.λ = mortality.([bio_pars], [opts], bio_df.age)
	bio_df.S = survivor.([bio_pars], [opts], [0.0], bio_df.age)
	bio_df.dH[2:end]=(bio_df.H[2:end]-bio_df.H[1:(end-1)])./bio_df.H[1:(end-1)]

	# Check that MaxAge is high enough
	@assert bio_df.S[Int(bio_pars.MaxAge+1)] == 0.0 "bio_pars.MaxAge not high enough"

	# Shape of S and H
	if opts.param != :none
		bio_df.Sgrad = round.(compute_Sgrad.([bio_pars], [opts], [0.0], bio_df.age), digits = 12)
		bio_df.Sacc = round.(compute_Sacc.([bio_pars], [opts], [0.0], bio_df.age), digits =9)
		bio_df.Hgrad = round.(compute_Hgrad.([bio_pars], [opts], bio_df.age), digits = 12)
		bio_df.Hacc = round.(compute_Hacc.([bio_pars], [opts], bio_df.age), digits = 9)
	end
	# Economic variables
	econ_vars = [:age, :C, :Cdot, :L, :SL, :W, :dW, :Y, :A, :Sav, :discount,
		:Wgrad, :Wacc, :Z, :zc, :uc, :ul, :V, :v, :WTP, :WTP_H, :WTP_S, :WTP_W]
	econ_df = DataFrame(zeros(Int(bio_pars.MaxAge+1), length(econ_vars)), econ_vars)
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
	vars_df = innerjoin(bio_df, econ_df, on = :age)

	# Define wage (may use health data)
	vars_df.W = wage.([bio_pars], [econ_pars], [opts], vars_df.age)
	if opts.param != :none
		vars_df.Wgrad = round.(compute_Wgrad.([bio_pars], [econ_pars], [opts], vars_df.age), digits = 10)
		vars_df.Wacc = round.(compute_Wacc.([bio_pars], [econ_pars], [opts], vars_df.age), digits = 7)
	end
	vars_df.dW[2:end]=(vars_df.W[2:end].-vars_df.W[1:end-1])./vars_df.W[1:end-1]
	vars_df.dH[2:end] = (vars_df.H[2:end]-vars_df.H[1:(end-1)])./vars_df.H[1:(end-1)]

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

	# Output vars_df
	return vars_df
end





"""
Create matrix of Struld WTP for different HLE and LE
"""
function struld_matrix(bio_pars, param1, param2, LE_range, HLE_range, opts; edge_adj = true)
	# Initialise WTP_df
	WTP_df = DataFrame(LE = vcat(repeat(LE_range, length(HLE_range)),
		(minimum(LE_range)+0.001):0.001:maximum(LE_range)),
		HLE = vcat(repeat(HLE_range, inner = length(LE_range)),
		(minimum(LE_range):0.001:(maximum(LE_range)-0.001))))
	# Some empty columns to fill
	WTP_df[:,:param1] .= 0.0
	WTP_df[:,:param2] .= 0.0
	WTP_df[:,:dLE] .= 0.0
	WTP_df[:,:WTP] .= 0.0

	# Fill in WTP_df
	prog = Progress(nrow(WTP_df), desc = "Computing WTP: ")
	for ii in 1:nrow(WTP_df)
		# Set param1 to hit certain LE
		LE_new = WTP_df.LE[ii]
		HLE_new = WTP_df.HLE[ii]

		setfield!(bio_pars, param1,
			find_zero(function(x) temp_pars=deepcopy(bio_pars);
			setfield!(temp_pars, param1, x);
			LE_new - LE(temp_pars, opts, aa = 0.0) end,
			(1e-16, 100.)))
		WTP_df.param1[ii] = getfield(bio_pars, param1)

		if HLE_new < LE_new
			try
				try
					setfield!(bio_pars, param2,
						find_zero(function(x) temp_pars=deepcopy(bio_pars);
						setfield!(temp_pars, param2, x);
						HLE_new - HLE(temp_pars, opts, aa = 0.0) end,
						(1e-12, 1.)))
				catch
					setfield!(bio_pars, param2,
						find_zero(function(x) temp_pars=deepcopy(bio_pars);
						setfield!(temp_pars, param2, max(x, 1e-16));
						HLE_new - HLE(temp_pars, opts, aa = 0.0) end,
						getfield(bio_pars, param2)))
				end
				# Store health param value
				WTP_df.param2[ii] = getfield(bio_pars, param2)
				# Solve life cycle model
				try
					vars_df = define_vars_df(bio_pars, econ_pars, opts)
					# Calculate dLE for mortality parameter
					dLE = round(central_fdm(5,1,
						max_range = getfield(bio_pars, param1))(function(x)
						temp_pars=deepcopy(bio_pars);
						setfield!(temp_pars, param1, x);
						LE(temp_pars, opts, aa = 0.0) end, getfield(bio_pars, param1)), digits = 10)
					WTP_df.dLE[ii] = dLE
					# Compute WTP at birth
					WTP_df.WTP[ii] = sum(vars_df.v .* vars_df.Sgrad .* vars_df.discount)
				catch
					WTP_df.dLE[ii] = WTP_df.dLE[ii-1]
					WTP_df.WTP[ii] = WTP_df.WTP[ii-1]
				end
			catch
				WTP_df.param2[ii] = NaN
				WTP_df.dLE[ii] = NaN
				WTP_df.WTP[ii] = NaN
			end

		else

			WTP_df.param2[ii] = NaN
			WTP_df.dLE[ii] = NaN
			WTP_df.WTP[ii] = NaN

		end

		next!(prog)
	end
	WTP_df.WTP_1y = WTP_df.WTP./WTP_df.dLE
	# Reset bio_pars
	bio_pars =deepcopy(opts.bio_pars0)

	return WTP_df
end






"""
Create matrix of Struld WTP for different HLE and LE
"""
function dorian_matrix(bio_pars, param1, param2, LE_range, HLE_range, opts;
	edge_adj = true)
	# Initialise WTP_df
	if edge_adj
		WTP_df = DataFrame(LE = vcat(repeat(LE_range, length(HLE_range)),
			(minimum(LE_range)+0.001):0.001:maximum(LE_range)),
			HLE = vcat(repeat(HLE_range, inner = length(LE_range)),
			(minimum(LE_range):0.001:(maximum(LE_range)).-0.001)))
	else
		WTP_df = DataFrame(LE = repeat(LE_range, length(HLE_range)),
			HLE = repeat(HLE_range, inner = length(LE_range)))
	end
	# Some empty columns to fill
	WTP_df[:,:param1] .= 0.0
	WTP_df[:,:param2] .= 0.0
	WTP_df[:,:dHLE] .= 0.0
	WTP_df[:,:WTP] .= 0.0

	# Fill in WTP_df
	prog = Progress(nrow(WTP_df), desc = "Computing WTP: ")
	for ii in 1:nrow(WTP_df)
		# Set param1 to hit certain LE
		LE_new = WTP_df.LE[ii]
		HLE_new = WTP_df.HLE[ii]

		setfield!(bio_pars, param1,
			find_zero(function(x) temp_pars=deepcopy(bio_pars);
			setfield!(temp_pars, param1, x);
			LE_new - LE(temp_pars, opts, aa = 0.0) end,
			(1e-16, 100.)))
		WTP_df.param1[ii] = getfield(bio_pars, param1)

		if HLE_new < LE_new
			try
				try
					setfield!(bio_pars, param2,
						find_zero(function(x) temp_pars=deepcopy(bio_pars);
						setfield!(temp_pars, param2, x);
						HLE_new - HLE(temp_pars, opts, aa = 0.0) end,
						(1e-12, 1.)))
				catch
					setfield!(bio_pars, param2,
						find_zero(function(x) temp_pars=deepcopy(bio_pars);
						setfield!(temp_pars, param2, max(x, 1e-16));
						HLE_new - HLE(temp_pars, opts, aa = 0.0) end,
						getfield(bio_pars, param2)))
				end
				# Store health param value
				WTP_df.param2[ii] = getfield(bio_pars, param2)
				# Solve life cycle model
				try
					vars_df = define_vars_df(bio_pars, econ_pars, opts)
					# Calculate dLE for mortality parameter
					dHLE = round(central_fdm(5,1,
						max_range = getfield(bio_pars, param2))(function(x)
						temp_pars=deepcopy(bio_pars);
						setfield!(temp_pars, param2, x);
						HLE(temp_pars, opts, aa = 0.0) end, getfield(bio_pars, param2)), digits = 10)
					WTP_df.dHLE[ii] = dHLE
					# Compute WTP at birth
					WTP_df.WTP[ii]=sum(vars_df.q.*vars_df.S.*vars_df.discount)
				catch
					WTP_df.dHLE[ii] = WTP_df.dHLE[ii-1]
					WTP_df.WTP[ii] = WTP_df.WTP[ii-1]
				end
			catch
				WTP_df.param2[ii] = NaN
				WTP_df.dHLE[ii] = NaN
				WTP_df.WTP[ii] = NaN
			end
		else

			WTP_df.param2[ii] = NaN
			WTP_df.dHLE[ii] = NaN
			WTP_df.WTP[ii] = NaN

		end

		next!(prog)
	end
	WTP_df.WTP_1y = WTP_df.WTP./WTP_df.dHLE

	# Reset bio_pars
	bio_pars =deepcopy(opts.bio_pars0)

	return WTP_df
end
