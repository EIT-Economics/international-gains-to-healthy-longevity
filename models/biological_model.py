#!/usr/bin/env python3
"""
Biological Model for Health and Longevity Analysis

This module implements the core biological functions from the Julia biology.jl file,
including frailty, mortality, health, and survival calculations.
"""

import numpy as np
from scipy.integrate import quad
from dataclasses import dataclass
from typing import Union

@dataclass
class BiologicalParameters:
    """Biological model parameters"""
    T: float = 97.6  # Biological lifespan
    TH: float = 0.0  # Biological lifespan health adjustment
    TS: float = 0.0  # Biological lifespan mortality adjustment
    delta1: float = 1.0  # Ageing coefficient on frailty
    delta2: float = 1.0  # Rectangularisation coefficient on frailty
    F0: float = 0.0  # Constant on frailty
    F1: float = 1.0  # Coefficient on frailty
    gamma: float = 0.0966163007208525  # Exponent on frailty in mortality
    M0: float = 0.00  # Baseline mortality
    M1: float = 0.3319  # Coefficient on frailty in mortality
    psi: float = 0.0516  # Exponent on frailty in disability
    D0: float = 0.0821  # Constant in disability
    D1: float = 0.6386  # Coefficient on frailty in disability
    alpha: float = 0.34  # Exponent on disability ratio in health
    MaxAge: float = 240  # Max age for numerical integration
    LE_base: float = 78.9  # LE at starting values
    HLE_base: float = 68.5  # HLE at starting values
    Reset: float = 0.0  # Size of Wolverine Reset

def is_numeric(value):
    try:
        float(value)
        return True
    except (ValueError, TypeError):
        return False

class BiologicalModel:
    """Core biological model for health and longevity calculations"""
    
    def __init__(self, params: BiologicalParameters):
        self.params = params
    
    def frailty(self, ages: Union[list, np.ndarray], 
                age_start: float = 0.0, wolverine: float = 0.0) -> np.ndarray:
        """
        Compute frailty for given age(s)
        
        Args:
            ages: Age or array of ages
            age_start: Age at which intervention starts
            wolverine: Wolverine reset parameter
            
        Returns:
            (np.ndarray) of frailty values
        """
        assert isinstance(ages, np.ndarray) or isinstance(ages, list), "Frailty ages must be array or list"
        result = [0] * len(ages)
        
        for i, a in enumerate(ages):
            if a < age_start:
                # Before intervention
                result[i] = self.params.F1 * np.exp(self.params.delta1 * (a - self.params.T))
            else:
                # After intervention - apply delta2 transformation
                transformed_age = age_start + self.params.delta2 * (a - age_start)
                result[i] = self.params.F1 * np.exp(self.params.delta1 * (transformed_age - self.params.T))
            
            # Apply Wolverine reset if applicable
            if a >= wolverine:
                result[i] = self.params.F1 * np.exp(self.params.delta1 * (a - self.params.Reset - self.params.T))
        
        return np.array(result)

    def disability(self, ages: Union[list, np.ndarray], age_start: float = 0.0, wolverine: float = 0.0) -> np.ndarray:
        """
        Compute disability for given age(s)
        
        Args:
            ages: Array-like of ages
            age_start: Age at which intervention starts
            wolverine: Wolverine reset parameter
            
        Returns:
            Disability value(s)
        """
        assert isinstance(ages, np.ndarray) or isinstance(ages, list), "Disability ages must be array or list"
        F_a = self.frailty(ages, age_start, wolverine)
        return self.params.D0 + self.params.D1 * (F_a ** self.params.psi)
    
    def health(self, ages: Union[list, np.ndarray], 
               age_start: float = 0.0, wolverine: float = 0.0,
               no_compress: bool = False) -> Union[float, np.ndarray]:
        """
        Compute health for given age(s)
        
        Args:
            ages: Array-like of ages
            age_start: Age at which intervention starts
            wolverine: Wolverine reset parameter
            no_compress: Whether to prevent compression
            
        Returns:
            Health value(s)
        """
        assert isinstance(ages, np.ndarray) or isinstance(ages, list), "Health ages must be array or list"
        result = [0] * len(ages)
        
        for i, a in enumerate(ages):
            if not no_compress:
                D_0 = self.disability([0.0], age_start, wolverine)[0]
                D_a = self.disability([a], age_start, wolverine)[0]
            else:
                if a < self.params.T:
                    D_0 = self.disability([0.0], age_start, wolverine)[0]
                    D_a = self.disability([a], age_start, wolverine)[0]
                else:
                    # Use baseline parameters for ages >= T
                    baseline_params = BiologicalParameters()
                    D_0 = baseline_params.D0 + baseline_params.D1 * (baseline_params.F1 ** baseline_params.psi)
                    D_a = baseline_params.D0 + baseline_params.D1 * (baseline_params.F1 * np.exp(baseline_params.delta1 * (a - baseline_params.T)) ** baseline_params.psi)
            
            result[i] = (D_0 / D_a) ** self.params.alpha
            result[i] = max(result[i], 1e-7)  # Prevent very small values
        
        return np.asarray(result)
    
    def mortality(self, ages: Union[list, np.ndarray], 
                  age_start: float = 0.0, wolverine: float = 0.0) -> np.ndarray:
        """
        Compute mortality rate for given age(s)
        
        Args:
            ages: Array-like of ages
            age_start: Age at which intervention starts
            wolverine: Wolverine reset parameter
            
        Returns:
            Mortality rate(s)
        """
        assert isinstance(ages, np.ndarray) or isinstance(ages, list), "Mortality ages must be array or list"
        F_a = self.frailty(ages, age_start, wolverine)
        mu_a = self.params.M0 + self.params.M1 * (F_a ** self.params.gamma)
        return np.minimum(mu_a, 1.0)  # Bound above at 1.0
    
    def survivor(self, age_start: float, age_end: float, 
                 intervention_start: float = 0.0, wolverine: float = 0.0) -> float:
        """
        Compute survival probability from age_start to age_end
        
        Args:
            age_start: Starting age
            age_end: Ending age
            intervention_start: Age at which intervention starts
            wolverine: Wolverine reset parameter
            
        Returns:
            Survival probability
        """
        assert is_numeric(age_start) and is_numeric(age_end), "Age start and end must be numeric"
        if age_start >= age_end:
            return 0.0
        
        # For complex scenarios with interventions, use numerical integration
        if (intervention_start > 0) or (wolverine > 0):
            ages = np.linspace(age_start, age_end, int((age_end - age_start) * 10) + 1)
            mort_rates = self.mortality(ages, intervention_start, wolverine)
            # Approximate survival using trapezoidal rule
            survival = np.exp(-np.trapz(mort_rates, ages))
            return min(survival, 1.0)
        
        # For simple cases, use analytical solution
        try:
            F_start = self.frailty([age_start], intervention_start, wolverine)
            F_end = self.frailty([age_end], intervention_start, wolverine)
            
            if self.params.delta1 == 0:
                return np.exp(self.params.M0 * (age_start - age_end))
            
            survival = np.exp(
                self.params.M0 * (age_start - age_end) + 
                (self.params.M1 * (self.params.F1 ** self.params.gamma) / (self.params.gamma * self.params.delta1)) *
                (np.exp(self.params.gamma * self.params.delta1 * (age_start - self.params.T)) - 
                 np.exp(self.params.gamma * self.params.delta1 * (age_end - self.params.T)))
            )
            
            return min(survival, 1.0)
        except:
            # Fallback to numerical integration
            ages = np.linspace(age_start, age_end, int((age_end - age_start) * 10) + 1)
            mort_rates = self.mortality(ages, intervention_start, wolverine)
            survival = np.exp(-np.trapz(mort_rates, ages))
            return min(survival, 1.0)
    
    def life_expectancy(self, age_start: float = 0.0, 
                       intervention_start: float = 0.0, wolverine: float = 0.0) -> float:
        """
        Compute life expectancy from given age
        
        Args:
            age_start: Starting age
            intervention_start: Age at which intervention starts
            wolverine: Wolverine reset parameter
            
        Returns:
            Life expectancy
        """
        if (intervention_start > 0) or (wolverine > 0):
            # Use numerical integration for complex scenarios
            ages = np.arange(age_start, self.params.MaxAge + 1, 0.1)
            survival_probs = np.array([self.survivor(age_start, a, intervention_start, wolverine) 
                                     for a in ages])
            return np.trapz(survival_probs, ages)
        else:
            # Use analytical solution for simple cases
            try:
                def integrand(t):
                    return self.survivor(age_start, t, intervention_start, wolverine)
                
                result, _ = quad(integrand, age_start, self.params.MaxAge)
                return result
            except:
                # Fallback to numerical integration
                ages = np.arange(age_start, self.params.MaxAge + 1, 0.1)
                survival_probs = np.array([self.survivor(age_start, a, intervention_start, wolverine) 
                                         for a in ages])
                return np.trapz(survival_probs, ages)
    
    def healthy_life_expectancy(self, age_start: float = 0.0, 
                               intervention_start: float = 0.0, 
                               wolverine: float = 0.0,
                               age_delta: float = 0.1) -> float:
        """
        Compute healthy life expectancy from given age
        
        Args:
            age_start: Starting age
            intervention_start: Age at which intervention starts
            wolverine: Wolverine reset parameter
            age_delta: Age delta for numerical integration
            
        Returns:
            Healthy life expectancy
        """
        if (intervention_start > 0) or (wolverine > 0):
            # Use numerical integration for complex scenarios
            ages = np.arange(age_start, self.params.MaxAge + 1, 0.1)
            survival_probs = np.array([self.survivor(age_start, a, intervention_start, wolverine) 
                                     for a in ages])
            health_values = self.health(ages, intervention_start, wolverine)
            return np.trapz(survival_probs * health_values, ages)
        else:
            # Use analytical solution for simple cases
            try:
                def integrand(t):
                    return (self.survivor(age_start, t, intervention_start, wolverine) * 
                           self.health(t, intervention_start, wolverine))
                
                result, _ = quad(integrand, age_start, self.params.MaxAge)
                return result
            except:
                # Fallback to numerical integration
                ages = np.arange(age_start, self.params.MaxAge + 1, age_delta)
                survival_probs = np.array([self.survivor(age_start, a, intervention_start, wolverine) 
                                         for a in ages])
                health_values = self.health(ages, intervention_start, wolverine)
                return np.trapz(survival_probs * health_values, ages)
    
    def compute_gradients(self, age: Union[list, np.ndarray], param_name: str,
                         intervention_start: float = 0.0, wolverine: float = 0.0) -> np.ndarray:
        """
        Compute gradients of biological functions with respect to parameters
        
        Args:
            age: Array-like of ages
            param_name: Name of parameter to differentiate with respect to
            intervention_start: Age at which intervention starts
            wolverine: Wolverine reset parameter
            
        Returns:
            Gradient value(s)
        """
        assert isinstance(age, np.ndarray) or isinstance(age, list), "Age must be array or list"

        # Use finite differences for gradient computation
        h = 1e-6
        original_value = getattr(self.params, param_name)
        
        # Forward difference
        setattr(self.params, param_name, original_value + h)
        f_plus = self.health(age, intervention_start, wolverine)
        
        # Backward difference
        setattr(self.params, param_name, original_value - h)
        f_minus = self.health(age, intervention_start, wolverine)
        
        # Restore original value
        setattr(self.params, param_name, original_value)
        
        return (f_plus - f_minus) / (2 * h)
