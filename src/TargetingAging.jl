using Statistics, Parameters, DataFrames
using QuadGK, NLsolve, Roots, FiniteDifferences, SpecialFunctions
using Plots, ProgressMeter, Formatting, TableView, Latexify, LaTeXStrings
using CSV, XLSX


"""
Parameters of biological model
"""
@with_kw mutable struct BiologicalParameters
    T::Float64 = 97.6 # Biological Life span
    TH::Float64 = 0.0 # Biological Life span health adjustment
    TS::Float64 = 0.0 # Biological Life span mortality adjustment
    δ1::Float64 = 1.0 # ageing coefficient on frailty
    δ2::Float64 = 1.0 # rectangularisation coefficient on frailty
    F0::Float64 = 0.0 # Constant on frailty (might want to try -exp(-T))
    F1::Float64 = 1.0  # Coefficient on frailty
    γ::Float64 = 0.0966163007208525 # Exponent on frailty in mortality
    M0::Float64 = 0.00 # Baseline mortailty
    M1::Float64 = 0.3319 # Coefficient on frailty in mortailty
    ψ::Float64 = 0.0516 # Exponent on frailty in disability
    D0::Float64 = 0.0821 # Constant in disability
    D1::Float64 = 0.6386 # Coefficient on frailty in disability
    α::Float64 = 0.34 # Exponent on disability ratio in health
    MaxAge::Float64 = 240 # Max age used for numerical integration
    LE_base::Float64 = 78.9 # LE at starting values of parameters
    HLE_base::Float64 = 68.5 # LE at starting values of parameters
    Reset::Float64 = 0.0 # Size of Wolverine Reset
end

"""
Parameters of economic model
"""
@with_kw mutable struct EconomicParameters
    ρ::Float64 = 0.02 # Pure time preference
    r::Float64 = 0.02 # Real interest rate (1/β - 1.0)
    β::Float64 = (1/(1+r)) # Pure time preference
    ζ1::Float64 = 0.5 # Cobb-Douglas coefficient on health in productivity
    ζ2::Float64 = 0.5 # Cobb-Douglas coefficient on health in productivity
    A::Float64 = 50.0 # TFP
    σ::Float64 = 1/1.5 # Elasticity of intertemporal substitution
    η::Float64 = 1.509 # Elasticity of substitution between consumption and leisure
    #r_η::Float64 = (η-1)/η # Power parameter in CES (function of elasticity)
    ϕ::Float64 = 0.224 # Weight on consumption in z
    z_z0::Float64 = 0.1 # Ratio z to z0 at age 50
    z0::Float64 = 600 # Subsistence level of consumption-lesiure composite
    WageChild::Float64 = 6.975478 # Wage pre-graduation
    MaxHours::Float64 = 4000 # Maximum number of hours worked in a year
    AgeGrad::Real =  20 # Graduation age
    AgeRetire::Real = 65 # Retirement age
end

USc = Float64.(XLSX.readdata("data/USCData2018.xlsx","Sheet1","B19:B145"))

# Helper functions
include("aux_functions.jl")
# Biological functions and structures
include("biology.jl")
# Economic functions and structures
include("economics.jl")
# General object management
include("objects.jl")
# Plotting
include("plotting.jl")
# Functions to include data
include("data_functions.jl")
