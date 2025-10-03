"""
Compute the real interest rate discount for future periods
"""
function discount(econ_pars::EconomicParameters, tt)
    # Unpack relevant parameters
    @unpack r = econ_pars
    # Compute variable
    v=(1/(1+r)).^tt
    return v
end



"""
Function to compute work experience
"""
function experience(econ_pars, aa)
	@unpack AgeGrad, AgeRetire = econ_pars
	X = 0.0
	if aa > AgeGrad
		#X = 4.0.*(aa./(2.0*AgeRetire) - (aa./(2.0*AgeRetire)).^2)
		X = min.(aa, 50.0)
	end
	return X
end


"""
Function to compute the wage from health and age
"""
function wage(bio_pars::BiologicalParameters, econ_pars::EconomicParameters,
    opts, aa::Real)
    # Unpack relevant parameters
    @unpack A, ζ1, ζ2, AgeGrad, AgeRetire, WageChild = econ_pars
    @unpack MaxAge = bio_pars
	#wh = 0.3
    # Initialise the wage
	w = 0.0
    if opts.prod_age
		H = health(bio_pars, opts, aa)
        # Age post graduation
		if aa <= AgeGrad
			w = WageChild
		elseif aa > AgeGrad
			#X = min((aa)/AgeRetire, 1.0)
			#X = min(log(aa),5)/5
			#X = 4.0.*(aa./(2.0*AgeRetire) - (aa./(2.0*AgeRetire)).^2)
			X = min.(aa, 50.0)
        	w = A*(X^ζ1)*(H^ζ2)
			#aa
			#w = max((-8.996437e+01 + 1.189574e+01*aa -6.283531e-01*aa^2 + 1.915767e-02*aa^3 -
			#	3.187624e-04*aa^4 + 2.621153e-06*aa^5 - 8.315269e-09 *aa^6), 1.0)

		end
        # Add a retirement penalty
		if aa > AgeRetire
        	#w *= 0.68
		end
    else
        H = health(opts.bio_pars0, opts, aa)
		if aa <= AgeGrad
			w = WageChild
		elseif (aa > AgeGrad) & (aa <= AgeRetire)
        	w = 1.35 .* log.(aa - AgeGrad).* WageChild .+ WageChild
		elseif aa > AgeRetire
			w_Retire = 1.35 .* log.(AgeRetire - AgeGrad).* WageChild .+ WageChild
			H_Retire = health(opts.bio_pars0, opts, (AgeRetire+1))
			w = w_Retire *0.68*(H/H_Retire).^1.75
		end

    end

	if (w < 1e-7) && (aa > AgeRetire)
		w = 1e-7
	end

    return w

end

if false
	vars_df = define_vars_df(bio_pars, econ_pars, opts)
	vars_df.W = wage.([bio_pars], [econ_pars], [opts], vars_df.age)
	vars_df.X = experience.([econ_pars], vars_df.age)/50
	plot(vars_df.X, label = "Experience")
	plot!(vars_df.H, label ="Health")
	plot!(standardise(vars_df.W), label ="Wage")
	plot!(standardise(vars_df.L), label = "Leisure")
	plot(vars_df.W)
	plot(vars_df.L)
end


"""
Computes gradient of wage at a given age aa, wrt opts.param
"""
function compute_Wgrad(bio_pars::BiologicalParameters, econ_pars::EconomicParameters,
	opts, aa::Real)
	# First deriviate of surviving until aa wrt opts.param
	grad = central_fdm(5,1,
		max_range = getfield(bio_pars, opts.param))(function(x)
		temp_pars=deepcopy(bio_pars);
		setfield!(temp_pars, opts.param, x);
		wage(temp_pars, econ_pars, opts, aa) end,
		getfield(bio_pars, opts.param))
	return grad
end


"""
Computes second deriv of wage from 0 to a given age, wrt opts.param
"""
function compute_Wacc(bio_pars::BiologicalParameters, econ_pars::EconomicParameters,
	opts, aa::Real)
	# Second deriviate of surviving until aa wrt opts.param
	grad = central_fdm(5,2,
		max_range = getfield(bio_pars, opts.param))(function(x)
		temp_pars=deepcopy(bio_pars);
		setfield!(temp_pars, opts.param, x);
		wage(temp_pars, econ_pars, opts, aa) end,
		getfield(bio_pars, opts.param))
	return grad
end


"""
Function for finding C(0)
"""
function find_C0(F::Array{Float64,1}, x::Array{Float64,1}, vars_df::DataFrame,
    econ_pars::EconomicParameters, bio_pars::BiologicalParameters, opts::NamedTuple)
    # Unpack relevant parameters
    @unpack σ, r, β, ρ, η, ϕ, MaxHours = econ_pars
    @unpack MaxAge = bio_pars
    # global Max_age Age0 sigma realr rho dHt eta SL dWt DLeisureShiftt C Cdot L phi W LeisureShift r MaxH S d LTheta;

    for tt in Int.((opts.AgeGrad+1):(MaxAge+1))
        # Initialise consumption in graduation period with x
        if tt == opts.AgeGrad+1
            vars_df.C[tt]=x[1]
        else
            # Consumption using Euler equation when intratemporal Euler holds
            vars_df.Cdot[tt] = σ*(r-ρ)+σ*vars_df.dH[tt]+(η-σ)*vars_df.SL[tt-1]*
                (vars_df.dW[tt])
            vars_df.C[tt] = max(vars_df.C[tt-1].*(1+vars_df.Cdot[tt]), 1e-7)
        end
        vars_df.L[tt] = vars_df.C[tt]*(((ϕ/(1-ϕ))*(vars_df.W[tt]))^(-η))
        # Leisure>maxH then intratemporal Euler fails and need to re-calculate
        if vars_df.L[tt] > MaxHours
            if tt != opts.AgeGrad+1
                vars_df.Cdot[tt]= (1+((σ-η)/η)*vars_df.SL[tt-1])^-1*(σ*(r-ρ)+σ*vars_df.dH[tt])
                vars_df.C[tt]= max(vars_df.C[tt-1]*(1+vars_df.Cdot[tt]), 1e-7)
            end
            vars_df.L[tt]= MaxHours
        end
        # Share of leisure in total consumption
        vars_df.SL[tt]=( (1-ϕ)*(ϕ* vars_df.C[tt].^((η-1)/η) +
            (1-ϕ)*vars_df.L[tt].^((η-1)/η)).^-1.0* vars_df.L[tt].^((η-1)/η) )
    end

    # Cut off survival rate at 1e-20, as non-zero survival at the end seems to cause probs
    Srate = round.(vars_df.S, digits = 400)
    # Intertemporal Budget Constraint
    ages = Int.((opts.AgeGrad+1):(MaxAge+1))
    expenditure =sum(vars_df.C[ages].*Srate[ages].* vars_df.discount[ages])
    income = sum((MaxHours .- vars_df.L[ages]).*vars_df.W[ages].*
        Srate[ages].*vars_df.discount[ages])

    F[1] = expenditure - income

    return F
end



""


"""
Compute Z from C, L and parameters
"""
function compute_Z(econ_pars::EconomicParameters, vars_df::DataFrame)
    # Unpack relevant parameters
    @unpack η, ϕ = econ_pars
    # Compute the conumption-leisure composite z at each age
    v = (ϕ.*vars_df.C.^((η-1)/η) .+ (1-ϕ).*vars_df.L.^((η-1)/η)).^(η/(η-1))
    return v
end


"""
Compute z0, the subsistence level of consumption/leisure
"""
function compute_z0(econ_pars::EconomicParameters, vars_df::DataFrame)
    # Unpack relevant parameters
    @unpack η, ϕ, z_z0 = econ_pars
    # Calibrate z0 so that ratio is z_z0 at age 49
    z0= z_z0.*(ϕ.* vars_df.C[49].^((η - 1)/η) .+
        (1-ϕ).* vars_df.L[49].^((η - 1)/η)).^(η/(η - 1))
    # Compute instantaneous utlity based on Z
    return z0
end


"""
Compute zc, coefficient on consumption in Z
"""
function compute_zc(econ_pars::EconomicParameters, vars_df::DataFrame)
    # Unpack relevant parameters
    @unpack η, ϕ = econ_pars
    # Z_c is coefficient on C in Z
    zc= ϕ.*(vars_df.Z./vars_df.C).^(1/η)
    # Compute instantaneous utlity based on Z
    return zc
end

"""
Compute zl, coefficient on leisure in Z
"""
function compute_zl(econ_pars::EconomicParameters, vars_df::DataFrame)
    # Unpack relevant parameters
    @unpack η, ϕ = econ_pars
    # Z_c is coefficient on C in Z
    zl= (1-ϕ).*(vars_df.Z./vars_df.L).^(1/η)
    # Compute instantaneous utlity based on Z
    return zl
end

"""
Compute u, instantaneous utility from Z
"""
function compute_u(econ_pars::EconomicParameters, vars_df::DataFrame)
    # Unpack relevant parameters
    @unpack σ, z0 = econ_pars
    # Compute u
    u = (σ/(σ-1)).*(vars_df.Z.^((σ-1)/σ) .- z0^((σ - 1)/σ))
    return u
end


"""
Compute uc, marginal utility of consumption from C and L
"""
function compute_uc(econ_pars::EconomicParameters, vars_df::DataFrame)
    # Unpack relevant parameters
    @unpack σ, η, ϕ, = econ_pars
    # Compute Z and Zc
    vars_df.Z = compute_Z(econ_pars, vars_df)
    zc= compute_zc(econ_pars, vars_df)
    # Compute marginal utility
    uc = zc.* vars_df.Z.^(-1/σ)
    return uc
end



"""
Compute ul, marginal utility of leisure from C and L
"""
function compute_ul(econ_pars::EconomicParameters, vars_df::DataFrame)
    # Unpack relevant parameters
    @unpack σ = econ_pars
    # Compute Z and Zc
    vars_df.Z = compute_Z(econ_pars, vars_df)
    zl= compute_zl(econ_pars, vars_df)
    # Compute marginal utility
    ul = zl.* vars_df.Z.^(-1/σ)
    return ul
end



"""
Compute SL from C, L and parameters
"""
function compute_sL(econ_pars::EconomicParameters, vars_df::DataFrame)
    # Unpack relevant parameters
    @unpack η, ϕ = econ_pars
    # Compute the share of leisure in z at each age
    v =( (1-ϕ).*vars_df.L.^((η-1)/η) ) .*
        (ϕ.*vars_df.C.^((η-1)/η) .+ (1-ϕ).*vars_df.L.^((η-1)/η)).^-1.0
    return v
end



"""
Solve Household problem
"""
function solve_HH(bio_pars::BiologicalParameters, econ_pars::EconomicParameters,
		opts, vars_df::DataFrame)

    # Change in health for euler equation
    vars_df.dH[2:end] = (vars_df.H[2:end]-vars_df.H[1:(end-1)])./vars_df.H[1:(end-1)]

    # Solve HH problem
    res = nlsolve((F,x) -> find_C0(F,x, vars_df, econ_pars, bio_pars, opts),
        [vars_df.C[opts.AgeGrad]])
	if !converged(res)
        res = nlsolve((F,x) -> find_C0(F,x, vars_df, econ_pars, bio_pars, opts),
            [15000.])
    end
    if !converged(res)
		res = nlsolve((F,x) -> find_C0(F,x, vars_df, econ_pars, bio_pars, opts),
			[50000.0])
	end
	if !converged(res)
		res = nlsolve((F,x) -> find_C0(F,x, vars_df, econ_pars, bio_pars, opts),
			[100000.0])
	end
    if !converged(res)
        res = nlsolve((F,x) -> find_C0(F,x, vars_df, econ_pars, bio_pars, opts),
            [vars_df.C[opts.AgeGrad]], method = :newton)
    end
	if !converged(res)
        res = nlsolve((F,x) -> find_C0(F,x, vars_df, econ_pars, bio_pars, opts),
            [46450.], method = :newton)
    end
	if !converged(res)
        res = nlsolve((F,x) -> find_C0(F,x, vars_df, econ_pars, bio_pars, opts),
            [1770.], method = :newton)
    end
	if !converged(res)
        res = nlsolve((F,x) -> find_C0(F,x, vars_df, econ_pars, bio_pars, opts),
            [17000.])
    end

    @assert converged(res) "find_C0 failed to converge"

    vars_df.Y=vars_df.W .* (econ_pars.MaxHours .- vars_df.L)
    vars_df.Sav=vars_df.W.*(econ_pars.MaxHours .-vars_df.L).-vars_df.C
    vars_df.A[(opts.AgeGrad+1):end]=cumsum((vars_df.Y[(opts.AgeGrad+1):end].-
        vars_df.C[(opts.AgeGrad+1):end]).*
        vars_df.S[(opts.AgeGrad+1):end].*vars_df.discount[(opts.AgeGrad+1):end])

    return vars_df

end



"""
Compute U, the net present value of utility at age a, including recalculating C and L
"""
function compute_U(bio_pars::BiologicalParameters, econ_pars::EconomicParameters,
		opts, vars_df::DataFrame, aa::Real; reset_bio::Bool = false, reset_econ::Bool = false)
    # Unpack relevant parameters
    @unpack ρ, AgeGrad, AgeRetire = econ_pars
    @unpack MaxAge = bio_pars
	if reset_bio
		# Compute health
		vars_df.H = health.([bio_pars], [opts], vars_df.age)
		# Compute survivor rate (decide on C and L at birth and stick with this)
		vars_df.S = survivor.([bio_pars], [opts], [0.0], vars_df.age)
	end
    # Update econ (wage may use health data)
	if reset_econ
		vars_df.W = wage.([bio_pars], [econ_pars], [opts], vars_df.age)
		vars_df.dW[2:end]=(vars_df.W[2:end].-vars_df.W[1:end-1])./vars_df.W[1:end-1]
    	# Solve HH problem
    	vars_df = solve_HH(bio_pars, econ_pars, opts, vars_df)
    	# Can recompute z0 with new consumption and leisure
    	if opts.redo_z0
        	econ_pars.z0 = compute_z0(econ_pars, vars_df)
    	end
    	# Compute Zs
    	vars_df.Z = compute_Z(econ_pars, vars_df)
    	# Compute u
	end
    u = compute_u(econ_pars, vars_df)
    # Compute U
    rows = Int.((aa+1):nrow(vars_df))
    df1 = vars_df[rows,:]
	if reset_bio
		S_aa = survivor.([bio_pars], [opts], [aa], df1.age)
	else
		S_a = vars_df.S/vars_df.S[Int.(aa+1)]
		S_aa = S_a[rows]
	end

    U = sum( exp.(-ρ.*(df1.age.-aa)).*df1.H.*u[rows].*S_aa )

    return U
end

"""
Compute μ the Lagranage multiplier on the budget constraint (i.e. the marginal utility of wealth)
	(you need a vars_df that already has H and uc)
"""
function compute_μ(bio_pars::BiologicalParameters, econ_pars::EconomicParameters,
		opts, vars_df::DataFrame)
		# Unpack relevant parameters
    @unpack ρ, r = econ_pars
	@unpack MaxAge = bio_pars

	μs = exp.((r - ρ).*(vars_df.age .- 0.0)).*vars_df.H.*vars_df.uc
	μ = mean(μs[Int.((opts.AgeGrad+1):(opts.AgeRetire+1))])

	return μ
end




"""
Compute v, the value of each life year from Z, C, W and L
"""
function compute_v(econ_pars::EconomicParameters, vars_df::DataFrame)
    # Unpack relevant parameters
    @unpack σ, η, ϕ, z0, MaxHours = econ_pars
    # Compute the value of each life year
    v = (σ/(σ-1)).*( vars_df.Z.^((σ-1)/σ) .- z0^((σ-1)/σ) )./
        (vars_df.zc .* vars_df.Z.^(-1/σ)) .+
        vars_df.W .*(MaxHours.- vars_df.L) .- vars_df.C
    return v
end


function compute_v1(econ_pars::EconomicParameters, vars_df::DataFrame)
    # Unpack relevant parameters
    @unpack σ, η, ϕ, z0, MaxHours = econ_pars
    # Compute the value of each life year
    u = compute_u(econ_pars, vars_df)
    u_c = compute_uc(econ_pars, vars_df)
    v = u./u_c .+ vars_df.W .*(MaxHours.- vars_df.L) .- vars_df.C
    return v
end


"""
Compute q, the impact of a change in health (and wage) on quality of life
    Need to edit to include Wgrad once we have this
"""
function compute_q(econ_pars::EconomicParameters, vars_df::DataFrame)
    # Unpack relevant parameters
    @unpack σ, η, ϕ, z0, MaxHours = econ_pars
    # Compute change in quality of life given Hgrad
    q = (vars_df.Hgrad./vars_df.H) .* (σ/(σ-1)).*(
        vars_df.Z.^((σ-1)/σ) .- z0^((σ-1)/σ) )./(vars_df.zc .* vars_df.Z.^(-1/σ)) .+
		vars_df.Wgrad.*(MaxHours .- vars_df.L)
    return q
end




"""
Function that computes WTP
"""
function compute_wtp(bio_pars, econ_pars, opts, vars_df; current_age = 0.0)

	# Set WTP variables to zero
	vars_df[:,[:WTP, :WTP_S, :WTP_H, :WTP_W]] .=  0.

	# For each age we need to recompute Sgrad(a) given that they have made it this far
	for kk in Int.((current_age+1):bio_pars.MaxAge+1)
		rows = kk:Int(bio_pars.MaxAge+1)
		# Sgrad with age a as starting point.
		if all(vars_df.Sgrad .== 0.)
			Sa_grad = vars_df.Sgrad[rows]
		else
			Sa_grad = round.(compute_Sgrad.([bio_pars], [opts],
				[kk-1], vars_df.age[rows]), digits = 11)
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
Alternative specification where solutions are fully numerical (as a debugging tool)
"""




"""
Intratemporal condition
"""
function intra(econ_pars::EconomicParameters, vars_df::DataFrame, tt::Int)
    # Compute uc and ul at time tt
    uc = compute_uc(econ_pars, vars_df[[tt],:])
    ul = compute_ul(econ_pars, vars_df[[tt],:])
    w = vars_df.W[tt]
    # Compute output, which should be zero
    output = uc./ul .- 1/w
    return output
end

"""
Solve intratemporal condition for L given C
"""
function solve_intra(econ_pars, vars_df, tt)
    find_zero(function (x) vars_df.L[tt] = x;
        intra(econ_pars, vars_df,tt)
    end, (0.0, 9e100))
    return vars_df.L[tt]
end


"""
Intertemporal condition
"""
function inter(econ_pars::EconomicParameters, vars_df::DataFrame, tt::Int)
    # Unpack relevant parameters
    @unpack β, r = econ_pars
    # Compute uc at time tt and tt-1
    uc_t = compute_uc(econ_pars, vars_df[[tt],:])
    uc_tm1 = compute_uc(econ_pars, vars_df[[tt-1],:])
    # Identify health at time tt and tt-1
    H_t = vars_df.H[tt]
    H_tm1 = vars_df.H[tt-1]

    # Compute output, which should be zero
    output = H_tm1.*uc_tm1 .- β*(1+r).*H_t.*uc_t
    return output
end

"""
Solve intertemporal for t given t-1
"""
function solve_inter(econ_pars::EconomicParameters, vars_df::DataFrame, tt::Int)
    find_zero(function (x) vars_df.C[tt] = x;
        vars_df.L[tt] = min(solve_intra(econ_pars, vars_df, tt), econ_pars.MaxHours);
        inter(econ_pars, vars_df,tt)
    end, (0.1, 9e10))
    return vars_df
end




"""
Find c0, solving all conditions numerically
"""
function find_C0_numerical(F::Array{Float64,1}, x::Array{Float64,1}, vars_df::DataFrame,
    econ_pars::EconomicParameters, bio_pars::BiologicalParameters, opts::NamedTuple)
    # Unpack relevant parameters
    @unpack σ, r, β, ρ, η, ϕ, MaxHours = econ_pars
    @unpack MaxAge = bio_pars
    # global Max_age Age0 sigma realr rho dHt eta SL dWt DLeisureShiftt C Cdot L phi W LeisureShift r MaxH S d LTheta;

    for tt in Int.((opts.AgeGrad+1):(MaxAge+1))

        if tt == opts.AgeGrad+1
            vars_df.C[tt]=x[1]
            vars_df.L[tt]= min(solve_intra(econ_pars, vars_df, tt), MaxHours)
        else
            vars_df = solve_inter(econ_pars, vars_df, tt)
        end
        # Share of leisure in total consumption
        vars_df.SL[[tt]] .= compute_sL(econ_pars, vars_df[[tt],:])
    end

    # Cut off survival rate at 1e-20, as non-zero survival at the end seems to cause probs
    Srate = round.(vars_df.S, digits = 20)
    # Intertemporal Budget Constraint
    ages = Int.((opts.AgeGrad+1):(MaxAge+1))
    expenditure =sum(vars_df.C[ages].*Srate[ages].* vars_df.discount[ages])
    income = sum((MaxHours .- vars_df.L[ages]).*vars_df.W[ages].*
        Srate[ages].*vars_df.discount[ages])

    F[1] = expenditure - income

    return F
end
