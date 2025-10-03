



"""
Computes frailty for each age of age vector
"""
function frailty(bio_pars::BiologicalParameters, opts, age)
    # Unpack relevant parameters
	if age < opts.age_start
		@unpack F1,δ1,T = bio_pars
		F_a = F1 .* exp.(δ1.*(age .- T))
	else
		@unpack F1,δ1,δ2,T = bio_pars
		age = opts.age_start + δ2*(age - opts.age_start)
		F_a = F1 .* exp.(δ1.*(age .- T))
	end
	if age >= opts.Wolverine
		@unpack F1,δ1,δ2,T, Reset = bio_pars
		age -= Reset
		F_a = F1 .* exp.(δ1.*(age .- T))
	end

    return F_a
end

"""
AgeReset = 50
Reset = 4.0

age = Int.(0:240)
age_w = Float64.(deepcopy(age))
age_w[age_w .>= AgeReset] .-= Reset

plot(age, age, label = "Age")
plot!(age, age_w, label = "Wolverine")
"""

"""
Computes disability for each age of age vector
"""
function disability(bio_pars::BiologicalParameters, opts, age)
    # Unpack relevant parameters
	@unpack D0,D1,ψ = bio_pars
    # Compute variable
    F_a = frailty(bio_pars, opts, age)
    D_a = D0 .+ D1.*F_a.^ψ

    return D_a
end

"""
Computes health for each age of age vector
"""
function health(bio_pars::BiologicalParameters, opts, age::Real)
    # Unpack relevant parameters
    @unpack α, T = bio_pars
    # Depends on whether we have no_compress option
	if !opts.no_compress
		D_0 = disability(bio_pars, opts, 0.)
    	D_a = disability(bio_pars, opts, age)
	else
		if age<T
			D_0 = disability(bio_pars, opts, 0.)
			D_a = disability(bio_pars, opts, age)
		end
		if age>=T
			D_0 = disability(opts.bio_pars0, opts, 0.)
			D_a = disability(opts.bio_pars0, opts, age)
		end
	end
    H_a = (D_0./D_a).^α

	if (H_a < 1e-7)
		H_a = 1e-7
	end

    return H_a
end


"""
Computes mortality for each age of age vector (bounds above at 1.0)
"""
function mortality(bio_pars::BiologicalParameters, opts, age)
    # Unpack relevant parameters
	@unpack M0,M1,γ = bio_pars
    # Compute variable
    F_a = frailty(bio_pars, opts, age)
    μ_a = M0 .+ M1 .* F_a.^γ
    μ_a = min.(μ_a, 1.0)
    return μ_a
end


"""
Computes survivor rate for person of age aa until tt
"""
function survivor(bio_pars::BiologicalParameters, opts, aa::Real, tt::Real)
    # Unpack relevant parameters
	if (opts.age_start > 0) | (opts.Wolverine > 0)

		all_mort = mortality.([bio_pars], [opts], 0:tt)
		all_S = cumprod(1 .- vcat(0.0, all_mort[1:(end-1)]))

		S_a = all_S[Int(round(tt+1))]/all_S[Int(round(aa+1))]

	else
		if tt < opts.age_start
			@unpack M0,M1,γ,F0,F1,δ1,T = bio_pars
		else
			@unpack M0,M1,γ,F0,F1,δ1,δ2,T = bio_pars
			tt = opts.age_start + δ2*(tt - opts.age_start)
		end
	    # Compute variable
	    S_a = exp(M0*(aa-tt) + ((M1*F1^γ)/(γ*δ1))*(
	        exp(γ*δ1*(aa - T)) - exp(γ*δ1*(tt - T))) )
		# If opts.no_compress, we artificially prevent negative side of compression
		if opts.no_compress
			@unpack M0,M1,γ,F0,F1,δ1,T  = opts.bio_pars0
			if tt >= T
				S_a = exp(M0*(aa-tt) + ((M1*F1^γ)/(γ*δ1))*(
			        exp(γ*δ1*(aa - T)) - exp(γ*δ1*(δ2*tt - T))) )
			end
		end
	end
	S_a = min(S_a, 1.0)

	# Stop really small survival rates (at some point we can't have fractions of people surviving for centuries)
	if S_a < 1e-5
		S_a = 0.0
	end

    return S_a
end
#plot!(survivor.([bio_pars], [0.0], 0:241))



if false
	bio_pars.ψ = 1.0
	plot(health.([bio_pars], [opts], 1:110), label = "ψ = "*string(bio_pars.ψ), legend = :bottomleft,
		ylabel = "Health", xlabel = "Age")
	bio_pars.ψ = 0.1
	plot!(health.([bio_pars], [opts], 1:110), label = "ψ = "*string(bio_pars.ψ))
	bio_pars.ψ = 0.0516
	plot!(health.([bio_pars], [opts], 1:110), label = "ψ = "*string(bio_pars.ψ))
	bio_pars.ψ = 0.01
	plot!(health.([bio_pars], [opts], 1:110), label = "ψ = "*string(bio_pars.ψ))
	bio_pars.ψ = 0.001
	plot!(health.([bio_pars], [opts], 1:110), label = "ψ = "*string(bio_pars.ψ))

	bio_pars.ψ = 1.0
	plot(disability.([bio_pars], [opts], 1:110), label = "ψ = "*string(bio_pars.ψ), legend = :topleft,
		ylabel = "Disability", xlabel = "Age", ylim = (0,2))
	bio_pars.ψ = 0.1
	plot!(disability.([bio_pars], [opts], 1:110), label = "ψ = "*string(bio_pars.ψ))
	bio_pars.ψ = 0.0516
	plot!(disability.([bio_pars], [opts], 1:110), label = "ψ = "*string(bio_pars.ψ))
	bio_pars.ψ = 0.01
	plot!(disability.([bio_pars], [opts], 1:110), label = "ψ = "*string(bio_pars.ψ))
	bio_pars.ψ = 0.001
	plot!(disability.([bio_pars], [opts], 1:110), label = "ψ = "*string(bio_pars.ψ))

	bio_pars.γ = 0.2
	plot(survivor.([bio_pars], [opts], [0.0], 1:110), label = "γ = "*string(bio_pars.γ),
		legend = :bottomleft, ylabel = "Survivor", xlabel = "Age")
	bio_pars.γ = 0.0967
	plot!(survivor.([bio_pars], [opts], [0.0], 1:110), label = "γ = "*string(bio_pars.γ))
	bio_pars.γ = 0.05
	plot!(survivor.([bio_pars], [opts], [0.0], 1:110), label = "γ = "*string(bio_pars.γ))
	bio_pars.γ = 0.02
	plot!(survivor.([bio_pars], [opts], [0.0], 1:110), label = "γ = "*string(bio_pars.γ))

end




"""
Computes gradient of survivor rate from 0 to a given age, wrt opts.param
"""
function compute_Sgrad(bio_pars::BiologicalParameters, opts, aa::Real, tt::Real)
	# First deriviate of surviving until aa wrt opts.param
	grad = central_fdm(5,1,
		max_range = getfield(bio_pars, opts.param))(function(x)
		temp_pars=deepcopy(bio_pars);
		setfield!(temp_pars, opts.param, x);
		survivor(temp_pars, opts, aa, tt) end, getfield(bio_pars, opts.param))
	return grad
end


"""
Computes gradient of health at a given age aa, wrt opts.param
"""
function compute_Hgrad(bio_pars::BiologicalParameters, opts, aa::Real)
	# First deriviate of surviving until aa wrt opts.param
	grad = central_fdm(5,1,
		max_range = getfield(bio_pars, opts.param))(function(x)
		temp_pars=deepcopy(bio_pars);
		setfield!(temp_pars, opts.param, x);
		health(temp_pars, opts, aa) end, getfield(bio_pars, opts.param))
	return grad
end


"""
Computes second deriv of survivor rate from 0 to a given age, wrt opts.param
"""
function compute_Sacc(bio_pars::BiologicalParameters, opts, aa::Real, tt::Real)
	# Second deriviate of surviving until aa wrt opts.param
	grad = central_fdm(5,2,
		max_range = getfield(bio_pars, opts.param))(function(x)
		temp_pars=deepcopy(bio_pars);
		setfield!(temp_pars, opts.param, x);
		survivor(temp_pars, opts, aa, tt) end, getfield(bio_pars, opts.param))
	return grad
end


"""
Computes second deriv of health from 0 to a given age, wrt opts.param
"""
function compute_Hacc(bio_pars::BiologicalParameters, opts, aa::Real)
	# Second deriviate of surviving until aa wrt opts.param
	grad = central_fdm(5,2,
		max_range = getfield(bio_pars, opts.param))(function(x)
		temp_pars=deepcopy(bio_pars);
		setfield!(temp_pars, opts.param, x);
		health(temp_pars, opts, aa) end, getfield(bio_pars, opts.param))
	return grad
end


"""
Computes life expectancy from age a given some biological parameters
"""
function LE(bio_pars::BiologicalParameters, opts; aa::Real = 0.0)
    # Unpack relevant parameters
    @unpack MaxAge = bio_pars
    # Compute variable
	if (opts.age_start > 0) | (opts.Wolverine > 0)
    	v = sum(survivor.([bio_pars], [opts], [aa], aa:MaxAge))
	else
		v, err = quadgk(tt -> survivor(bio_pars, opts, aa, tt),aa,MaxAge)
	end
    life_exp = v
    return life_exp
end



"""
Computes healthy life expectancy from age a given biological parameters
"""
function HLE(bio_pars::BiologicalParameters, opts; aa = 0.0)
    # Unpack relevant parameters
    @unpack MaxAge = bio_pars
    # Compute variable
	if (opts.age_start > 0) | (opts.Wolverine > 0)
    	v = sum(survivor.([bio_pars], [opts], [aa], aa:MaxAge).*health.([bio_pars], [opts], aa:MaxAge))
	else
		v, err = quadgk(tt -> health(bio_pars,opts,tt)*survivor(bio_pars, opts, aa, tt),
			aa, MaxAge)
	end

    hlife_exp = v
    return hlife_exp
end
