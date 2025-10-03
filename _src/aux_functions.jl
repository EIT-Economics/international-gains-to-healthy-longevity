function standardise(x::Array{Float64})
    x_copy = deepcopy(x)
    x_copy[isinf.(x_copy)] .= NaN

    x_nonan = x_copy[.!isnan.(x_copy)]
    x_new = (x_nonan.-mean(x_nonan))./std(x_nonan)

    x_copy[.!isnan.(x_copy)] .= x_new

    return x_copy
end


"""
Generate a grid of bio_par where opts.param is varied
"""
function bio_par_grid(bio_pars::BiologicalParameters, opts; N::Int=101, prange = [0.0, 0.0])
	# N should be odd so that current level is in the middle
	@assert isodd(N) "Odd number of grid points, please"
	# Define the range
	if prange == [0.0, 0.0]
		upper_lim = 1.95*getfield(bio_pars, opts.param)
		lower_lim = 0.05*getfield(bio_pars, opts.param)
	else
		upper_lim = maximum(prange)
		lower_lim = minimum(prange)
	end
	stepsize = (upper_lim - lower_lim)/(N-1)
	paramvals = Float64.(lower_lim:stepsize:upper_lim)

	# Create array of BiologicalParameters structs
	bp_grid = Array{BiologicalParameters,1}([])
	for ii in 1:N
		temp_pars = deepcopy(bio_pars)
		setfield!(temp_pars, opts.param, paramvals[ii])
		push!(bp_grid, temp_pars)
	end

	return bp_grid
end



chunk(arr, n) = [arr[i:min(i + n - 1, end)] for i in 1:n:length(arr)]



function compute_S0grad_γ(bio_pars::BiologicalParameters, opts, aa::Real)
	@unpack γ,M1,T  = bio_pars
	# First deriviate of surviving until aa wrt γ
	grad = -(M1/(γ^2)) * ( (γ*(aa-T)-1)*exp(γ*(aa-T)) + (γ*T+1)*exp(-γ*T) )*survivor(bio_pars, aa)
	return grad
end

function compute_S0acc_γ(bio_pars::BiologicalParameters, opts, aa::Real)
	@unpack γ,M1,T  = bio_pars
	# d lnS/d γ
	dlS_dγ = -(M1/(γ^2)) * ( (γ*(aa-T)-1)*exp(γ*(aa-T)) + (γ*T+1)*exp(-γ*T) )
	# d^2 lnS/d γ^2
	d2lS_dγ2 = -(M1/(γ^4)) * ( ( γ^3*(aa-T)^2 - 2*γ^2*(aa-T) + 2*γ )*exp(γ*(aa-T)) -
		( γ^3*T^2 + 2*γ^2*T + 2*γ )*exp(-γ*T) )
	# First deriviate of surviving until aa wrt M1
	acc = (d2lS_dγ2 + dlS_dγ^2)*survivor(bio_pars, aa)
	return acc
end


function compute_S0grad_M1(bio_pars::BiologicalParameters, opts, aa::Real)
	@unpack γ,F1,T  = bio_pars
	# First deriviate of surviving until aa wrt M1
	grad =  ((F1^γ)/(γ))*(exp(γ*(0 - T)) - exp(γ*(aa - T))) *
		survivor(bio_pars, aa)
	return grad
end

function compute_lnS0grad_grad(bio_pars::BiologicalParameters, opts, aa::Real)
	# First deriviate of surviving until aa wrt opts.param
	grad = central_fdm(5,2,
		max_range = getfield(bio_pars, opts.param))(function(x)
		temp_pars=deepcopy(bio_pars);
		setfield!(temp_pars, opts.param, x);
		log(survivor(temp_pars,aa)) end, getfield(bio_pars, opts.param))
	return grad
end



function compute_S0acc_M1(bio_pars::BiologicalParameters, opts, aa::Real)
	@unpack γ,F1,T  = bio_pars
	# First deriviate of surviving until aa wrt M1
	grad =  (((F1^γ)/(γ))*(exp(γ*(0 - T)) - exp(γ*(aa - T))))^2 *
		survivor(bio_pars, aa)
	return grad
end




"""
Function to find param value for certain utlility
"""
function find_par_Ustar(F::Array{Float64,1}, x::Array{Float64,1}, param2::Symbol,
	Ustar::Float64, bio_pars::BiologicalParameters, econ_pars::EconomicParameters,
	opts, vars_df::DataFrame, aa::Real)

	temp_pars=deepcopy(bio_pars)
	setfield!(temp_pars, param2, abs(x[1]))
	F[1] = Ustar - compute_U(temp_pars, econ_pars, opts, vars_df, aa)
	return F
end
#Ustar = compute_U(opts.bio_pars0, econ_pars, opts, vars_df, 0.0)
"""
Find combination of two parameters that give constant Ubar
"""
function compute_indiff(bio_pars::BiologicalParameters, econ_pars::EconomicParameters,
		opts, vars_df::DataFrame, param1, param2, Ustar::Real;
		npoints::Int = 41, aa::Real = 0.0, prange = [0.0, 0.0], prange2 = [0.0001, 0.9])

	fig1 = plot(layout = (2,3), size = (1200,600))
	# Define range for param1
	if length(prange) == 2
		if prange == [0.0, 0.0]
			upper_lim = 1.10*getfield(bio_pars, param1)
			lower_lim = 1.0*getfield(bio_pars, param1)
		else
			upper_lim = maximum(prange)
			lower_lim = minimum(prange)
		end
		stepsize = (upper_lim - lower_lim)/(npoints-1)
		paramvals = Float64.(lower_lim:stepsize:upper_lim)
	else
		paramvals = prange
		npoints = length(paramvals)
	end

	# DataFrame to store results
	indiff_vars = [:param1, :param2, :LE0, :HLE0, :Ustar,
		Symbol("LE_"*string(Int(age_eval))), Symbol("HLE_"*string(Int(age_eval)))]
	indiff_df = DataFrame(zeros(npoints, length(indiff_vars)), indiff_vars)
	# Calculate param1-param2 pairs
	prog = Progress(npoints, desc = "Computing indifference curve at Ustar = "*
		string(round(Ustar, digits = 4))*": ")
	for pp in vcat(1:npoints) # Do first one twice?
		# Reset the biological parameters
		bio_pars = deepcopy(opts.bio_pars0)
		# Set param1 to the desired value
		indiff_df.param1[pp] = setfield!(bio_pars, param1, paramvals[pp])
		try
			# Compute param2 that keeps U(0) constant
			res =nlsolve((F,x) -> find_par_Ustar(F, x, param2, Ustar, bio_pars,
				econ_pars, opts, vars_df, aa), [getfield(bio_pars, param2)])
			#if !converged(res)
			#	res =nlsolve((F,x) -> find_par_Ustar(F, x, param2, Ustar, bio_pars,
			#		econ_pars, opts, vars_df, aa), [prange2[1]])
			#end
			# if !converged(res)
			# 	res =nlsolve((F,x) -> find_par_Ustar(F, x, param2, Ustar, bio_pars,
			# 		econ_pars, opts, vars_df, aa), [prange2[2]])
			# end
		 	@assert converged(res) "Couldn't find value of param2"
			indiff_df.param2[pp] = setfield!(bio_pars, param2, res.zero[1])

			# Plot variables as a sense check
			vars_df = define_vars_df(bio_pars, econ_pars, opts)
			# Solve again (there's a bug somewhere as Ustar doesn't come out quite right??)
			res =nlsolve((F,x) -> find_par_Ustar(F, x, param2, Ustar, bio_pars,
				econ_pars, opts, vars_df, aa), [getfield(bio_pars, param2)])
			fig1 = plot!(vars_df.C, subplot = 1, label = false, title = "Consumption")
			fig1 = plot!(vars_df.Z, subplot = 2, label = false, title = "Z")
			fig1 = plot!(vars_df.L, subplot = 3, label = false, title = "L")
			fig1 = plot!(vars_df.W, subplot = 4, label = false, title = "Wage")
			fig1 = plot!(vars_df.S, subplot = 5, label = false, title = "Survival")
			fig1 = plot!(vars_df.H, subplot = 6, label = false, title = "Health")
			# Compute HLE, LE and Ustar
			indiff_df.HLE0[pp] = HLE(bio_pars, opts, aa = 0.0)
			indiff_df.LE0[pp] = LE(bio_pars, opts, aa = 0.0)
			indiff_df.Ustar[pp] = compute_U(bio_pars, econ_pars, opts, vars_df, aa)
			#Remaining LE and HLE
			indiff_df[pp, Symbol("LE_"*string(Int(aa)))] = LE(bio_pars, opts, aa = aa)
			indiff_df[pp, Symbol("HLE_"*string(Int(aa)))] = HLE(bio_pars, opts, aa = aa)
			catch
			# If it's not possible to find a param2, then just fill with NAs
			indiff_df[pp,[:param2, :LE0, :HLE0, :Ustar, Symbol("LE_"*string(Int(age_eval))),
				Symbol("HLE_"*string(Int(age_eval)))]] .= NaN
		end
		next!(prog)
	end
	display(fig1)

	return indiff_df

end

# try
# 	indiff_df.param2[pp] = setfield!(bio_pars, param2,
# 		find_zero(function(x) temp_pars=deepcopy(bio_pars);
# 		setfield!(temp_pars, param2, abs(x));
# 		Ustar - compute_U(temp_pars, econ_pars, opts, vars_df, aa) end,
# 		prange2[2], atol=1e-8))
# 	display("second")
# catch
# 	indiff_df.param2[pp] = setfield!(bio_pars, param2,
# 		find_zero(function(x) temp_pars=deepcopy(bio_pars);
# 		setfield!(temp_pars, param2, abs(x));
# 		Ustar - compute_U(temp_pars, econ_pars, opts, vars_df, aa) end,
# 		prange2[1], atol=1e-8))
# 	display("third")
# end





























"""
Transformation of wages that appear in Euler Equation
"""
function w_tilde(econ_pars::EconomicParameters, w)
    @unpack η, ϕ = econ_pars
    v = ϕ .+ (1 - ϕ).*((ϕ.*w)./(1-ϕ)).^(1 - η)
    return v
end

"""
Function for finding C(0) with the discrete model
"""
function find_C0_broken(F::Array{Float64,1}, x::Array{Float64,1}, vars_df::DataFrame,
    econ_pars::EconomicParameters, bio_pars::BiologicalParameters, opts::NamedTuple)
    # Unpack relevant parameters
    @unpack σ, r, β, ρ, η, ϕ, MaxHours = econ_pars
    @unpack MaxAge = bio_pars
    # global Max_age Age0 sigma realr rho dHt eta SL dWt DLeisureShiftt C Cdot L phi W LeisureShift r MaxH S d LTheta;

    for tt in Int.((opts.AgeGrad+1):(MaxAge+1))

        if tt == opts.AgeGrad+1
            vars_df.C[tt]=x[1]
        else

            vars_df.dW[tt] = w_tilde(econ_pars, vars_df.W[tt-1])./w_tilde(econ_pars, vars_df.W[tt])
            # Consumption using Euler equation when intratemporal Euler holds
            vars_df.Cdot[tt]= (β*(1+r)*vars_df.H[tt]/vars_df.H[tt-1])^((η^2)/(η-1)) *
                (w_tilde(econ_pars, vars_df.W[tt-1])./w_tilde(econ_pars, vars_df.W[tt])
                )^((η*(η-σ)*η)/(σ*(η-1)*(η-1)))

            vars_df.C[tt]=vars_df.C[tt-1].*vars_df.Cdot[tt]

            #( σ*(r-ρ)+σ*vars_df.dH[tt]+(η-σ)*vars_df.SL[tt-1]*
            #    (vars_df.dW[tt]) )
            #vars_df.C[tt]=vars_df.C[tt-1].*(1+vars_df.Cdot[tt])
        end
        vars_df.L[tt]= vars_df.C[tt] * ((ϕ/(1-ϕ))*(vars_df.W[tt]))^(-η)

        # Leisure>maxH then intratemporal Euler fails and need to re-calculate
        if vars_df.L[tt] > MaxHours

            vars_df.Z[(tt-1):tt] = compute_Z(econ_pars, vars_df[(tt-1):tt,:])
            if tt == opts.AgeGrad+1
                vars_df.C[tt] = x[1]
            else

                vars_df.C[tt] = find_zero(function (x) vars_df.C[tt] = x;
                    RHS = (β*(1+r)*vars_df.H[tt]/vars_df.H[tt-1])^((σ*(η-1))/(η-σ)) *
                        vars_df.C[tt-1]^((σ*(η-1))/(η*(η-σ)))*vars_df.Z[tt-1]^((η-1)/η);
                    LHS = vars_df.C[tt]^(((η-1)*σ)/((η-σ)*η)) * (ϕ*vars_df.C[tt-1]^((η-1)/η) +
                        (1-ϕ)*MaxHours^((η-1)/η) );
                    RHS - LHS
                end, vars_df.C[tt-1])
            end
            vars_df.L[tt]= MaxHours
        end
        # Share of leisure in total consumption
        vars_df.SL[tt]=( (1-ϕ)*(ϕ* vars_df.C[tt].^((η-1)/η) +
            (1-ϕ)*vars_df.L[tt].^((η-1)/η)).^-1.0* vars_df.L[tt].^((η-1)/η) )
    end

    # Intertemporal Budget Constraint
    ages = Int.(opts.AgeGrad+1:Int(MaxAge+1))
    expenditure =sum(vars_df.C[ages].*vars_df.S[ages].* vars_df.discount[ages])
    income = sum((MaxHours .- vars_df.L[ages]).*vars_df.W[ages].*
        vars_df.S[ages].*vars_df.discount[ages])

    F[1] = expenditure - income

    return F
end
