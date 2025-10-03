"""
Update all scenario figures
"""
function scenario_plt(vars_df, fig1, fig2, fig3, col)
	plot(fig1)
	fig1 = plot!(vars_df.λ,title = "Mortality",legend = :topleft, subplot = 1,
		color = col, label = "LE = "*string(round(sum(vars_df.S), digits = 2)))
	fig1 = plot!(vars_df.S,title = "S",legend = false, color = col, subplot = 2)
	fig1 = plot!(vars_df.Sgrad,title = "Sgrad",legend = false, color = col, subplot = 3)
	fig1 = plot!(vars_df.D,title = "Disability",legend = false, color = col, subplot = 4)
	fig1 = plot!(vars_df.H,title = "H",legend = false, color = col, subplot = 5)
	fig1 = plot!(vars_df.Hgrad,title = "Hgrad",legend = false, color = col, subplot = 6)
	display(fig1)

	plot(fig2)
	fig2 = plot!(vars_df.C,title = "Consumption",legend = :topright, subplot = 1,
		color = col, label = "LE = "*string(round(sum(vars_df.S), digits = 2)))
	fig2 = plot!(vars_df.L,title = "Leisure",legend = false, color = col, subplot = 2)
	fig2 = plot!(vars_df.W,title = "Wage",legend = false, color = col, subplot = 3)
	fig2 = plot!(vars_df.A,title = "Assets",legend = false, color = col, subplot = 4)
	fig2 = plot!(vars_df.Z,title = "Z",legend = false, color = col, subplot = 5)
	fig2 = plot!(vars_df.Wgrad,title = "Wgrad",legend = false, color = col, subplot = 6)
	display(fig2)

	plot(fig3)
	fig3 = plot!(vars_df.v,title = "v(t)",legend = :topright, subplot = 1,
		color = col, label = "LE = "*string(round(sum(vars_df.S), digits = 2)))
	fig3 = plot!(vars_df.WTP_S,title = "WTP_S",legend = false, color = col, subplot = 2)
	fig3 = plot!(vars_df.V,title = "V_bar",legend = false, color = col, subplot = 3)
	fig3 = plot!(vars_df.q,title = "q(t)",legend = false, color = col, subplot = 4)
	fig3 = plot!(vars_df.WTP_H,title = "WTP_H",legend = false, color = col, subplot = 5)
	fig3 = plot!(vars_df.WTP_W,linestyle =:dash,legend = false, color = col, subplot = 5)
	fig3 = plot!(vars_df.WTP,title = "WTP",legend = false, color = col, subplot = 6)
	display(fig3)

	return fig1, fig2, fig3
end



"""
Plot shape of survival function wrt opts.param
"""
function survivor_deriv_plots(bio_pars, opts, vars_df; aa = 0.0, surf= false, npoints = 101, prange = [0.0, 0.0])
	# Define variables
	bp_grid = bio_par_grid(bio_pars, opts, N=npoints, prange = prange)
	ages = Matrix(transpose(hcat(repeat([vars_df.age], npoints)...)))
	Ss = survivor.(bp_grid, [opts], [0.0], ages)
	paramvals = getfield.(bp_grid, [opts.param])
	# Gradient and acceleration
	Sgrads = []; Saccs = []
	for ii in 1:npoints
		push!(Sgrads, round.(compute_Sgrad.([bp_grid[ii]], [opts], [aa], ages[ii,:]), digits = 12))
		push!(Saccs, round.(compute_Sacc.([bp_grid[ii]], [opts], [aa], ages[ii,:]), digits = 9))
	end
	# Plot as 3d surface plot
	if surf
		gr()
		plts = plot(layout = (1,3), size = (1000,300))
		plts = surface!(vcat(transpose(ages)...),
			repeat(paramvals, inner = length(vars_df.age)),
			vcat(transpose(Ss)...),
			camera=(30,30), xlabel= "Age", ylabel=string(opts.param), zlabel = "S",
			subplot= 1,title="S(0,t)", colorbar =:none, margin=3Plots.mm)
		plts = surface!(vcat(transpose(ages)...),
			repeat(paramvals, inner = length(vars_df.age)),
			vcat(Sgrads...),
			camera=(30,30), xlabel= "Age", ylabel=string(opts.param), zlabel = "S_"*string(opts.param),
			subplot= 2,title="S_"*string(opts.param)*"(0,t)", colorbar =:none, margin=3Plots.mm)
		plts = surface!(vcat(transpose(ages)...),
			repeat(paramvals, inner = length(vars_df.age)),
			vcat(Saccs...),
			camera=(30,30), xlabel= "Age", ylabel=string(opts.param), zlabel = "S_"*string(opts.param)*string(opts.param),
			subplot= 3,title="S_"*string(opts.param)*string(opts.param)*"(0,t)", colorbar =:none, margin=3Plots.mm)
	else
		# Or as heatmap
		gr()
		# Survival rate
		plts = plot(layout = (1,3), size = (800,300))
		plts = heatmap!(ages[1,:], paramvals, Ss, c =:coolwarm, margin=3Plots.mm,
			xlabel= "Age", ylabel=string(opts.param), subplot= 1,title="S(0,1)")
		# Gradient of survival rate
		plts = heatmap!(ages[1,:], paramvals, transpose(hcat(Sgrads...)), c =:coolwarm, margin=3Plots.mm,
			xlabel= "Age", ylabel=string(opts.param), subplot = 2,title="S_"*string(opts.param)*"(0,t)")
		# Second deriv of survival rate
		plts = heatmap!(ages[1,:], paramvals, transpose(hcat(Saccs...)), c =:coolwarm, margin=3Plots.mm,
			xlabel= "Age", ylabel=string(opts.param), subplot = 3,
			title="S_"*string(opts.param)*string(opts.param)*"(0,t)")
	end

	return plts

end




"""
Plot shape of health function wrt opts.param
"""
function health_deriv_plots(bio_pars, opts, vars_df; aa=0.0, surf= false, npoints = 101)
	# Define variables
	bp_grid = bio_par_grid(bio_pars, opts, N=npoints)
	ages = Matrix(transpose(hcat(repeat([vars_df.age], npoints)...)))
	vals = health.(bp_grid,[opts],ages)
	paramvals = getfield.(bp_grid, [opts.param])
	# Gradient and acceleration
	grads = []; accs = []
	for ii in 1:npoints
		push!(grads, round.(compute_Hgrad.([bp_grid[ii]], [opts], ages[ii,:]), digits = 12))
		push!(accs, round.(compute_Hacc.([bp_grid[ii]], [opts], ages[ii,:]), digits = 9))
	end
	# Plot as 3d surface plot
	if surf
		gr()
		plts = plot(layout = (1,3), size = (1000,300))
		plts = surface!(vcat(transpose(ages)...),
			repeat(paramvals, inner = length(vars_df.age)),
			vcat(transpose(vals)...),
			camera=(30,30), xlabel= "Age", ylabel=string(opts.param), zlabel = "H",
			subplot= 1,title="H(t)", colorbar =:none, margin=3Plots.mm)
		plts = surface!(vcat(transpose(ages)...),
			repeat(paramvals, inner = length(vars_df.age)),
			vcat(grads...),
			camera=(30,30), xlabel= "Age", ylabel=string(opts.param), zlabel = "H_"*string(opts.param),
			subplot= 2,title="H_"*string(opts.param)*"(t)", colorbar =:none, margin=3Plots.mm)
		plts = surface!(vcat(transpose(ages)...),
			repeat(paramvals, inner = length(vars_df.age)),
			vcat(accs...),
			camera=(30,30), xlabel= "Age", ylabel=string(opts.param), zlabel = "H_"*string(opts.param)*string(opts.param),
			subplot= 3,title="H_"*string(opts.param)*string(opts.param)*"(t)", colorbar =:none, margin=3Plots.mm)
	else
		# Or as heatmap
		gr()
		# Survival rate
		plts = plot(layout = (1,3), size = (800,300))
		plts = heatmap!(ages[1,:], paramvals, vals, c =:coolwarm, margin=3Plots.mm,
			xlabel= "Age", ylabel=string(opts.param), subplot= 1,title="H(t)")
		# Gradient of survival rate
		plts = heatmap!(ages[1,:], paramvals, transpose(hcat(grads...)), c =:coolwarm, margin=3Plots.mm,
			xlabel= "Age", ylabel=string(opts.param), subplot = 2,title="H_"*string(opts.param)*"(t)")
		# Second deriv of survival rate
		plts = heatmap!(ages[1,:], paramvals, transpose(hcat(accs...)), c =:coolwarm, margin=3Plots.mm,
			xlabel= "Age", ylabel=string(opts.param), subplot = 3,
			title="H_"*string(opts.param)*string(opts.param)*"(t)")
	end

	return plts

end




"""
Plot shape of wage function wrt opts.param
"""
function wage_deriv_plots(bio_pars, econ_pars, opts, vars_df; aa=0.0, surf= false, npoints = 101)
	# Define variables
	bp_grid = bio_par_grid(bio_pars, opts, N=npoints)
	ages = Matrix(transpose(hcat(repeat([vars_df.age], npoints)...)))
	vals = wage.(bp_grid, [econ_pars], [opts], ages)
	paramvals = getfield.(bp_grid, [opts.param])
	# Gradient and acceleration
	grads = []; accs = []
	for ii in 1:npoints
		push!(grads, round.(compute_Wgrad.([bp_grid[ii]], [econ_pars],[opts], ages[ii,:]), digits = 12))
		push!(accs, round.(compute_Wacc.([bp_grid[ii]], [econ_pars],[opts], ages[ii,:]), digits = 9))
	end
	# Plot as 3d surface plot
	if surf
		gr()
		plts = plot(layout = (1,3), size = (1000,300))
		plts = surface!(vcat(transpose(ages)...),
			repeat(paramvals, inner = length(vars_df.age)),
			vcat(transpose(vals)...),
			camera=(30,30), xlabel= "Age", ylabel=string(opts.param), zlabel = "W",
			subplot= 1,title="W(t)", colorbar =:none, margin=3Plots.mm)
		plts = surface!(vcat(transpose(ages)...),
			repeat(paramvals, inner = length(vars_df.age)),
			vcat(grads...),
			camera=(30,30), xlabel= "Age", ylabel=string(opts.param), zlabel = "W_"*string(opts.param),
			subplot= 2,title="W_"*string(opts.param)*"(t)", colorbar =:none, margin=3Plots.mm)
		plts = surface!(vcat(transpose(ages)...),
			repeat(paramvals, inner = length(vars_df.age)),
			vcat(accs...),
			camera=(30,30), xlabel= "Age", ylabel=string(opts.param), zlabel = "W_"*string(opts.param)*string(opts.param),
			subplot= 3,title="W_"*string(opts.param)*string(opts.param)*"(t)", colorbar =:none, margin=3Plots.mm)
	else
		# Or as heatmap
		gr()
		# Survival rate
		plts = plot(layout = (1,3), size = (800,300))
		plts = heatmap!(ages[1,:], paramvals, vals, c =:coolwarm, margin=3Plots.mm,
			xlabel= "Age", ylabel=string(opts.param), subplot= 1,title="W(t)")
		# Gradient of survival rate
		plts = heatmap!(ages[1,:], paramvals, transpose(hcat(grads...)), c =:coolwarm, margin=3Plots.mm,
			xlabel= "Age", ylabel=string(opts.param), subplot = 2,title="W_"*string(opts.param)*"(t)")
		# Second deriv of survival rate
		plts = heatmap!(ages[1,:], paramvals, transpose(hcat(accs...)), c =:coolwarm, margin=3Plots.mm,
			xlabel= "Age", ylabel=string(opts.param), subplot = 3,
			title="W_"*string(opts.param)*string(opts.param)*"(t)")
	end

	return plts

end
