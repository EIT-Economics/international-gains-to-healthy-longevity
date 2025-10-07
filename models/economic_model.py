#!/usr/bin/env python3
"""
Economic Model for Health and Longevity Analysis

This module implements the core economic functions from the Julia economics.jl file,
including household optimization, wage functions, and utility calculations.
"""

import numpy as np
from scipy.optimize import fsolve
from dataclasses import dataclass
from typing import Union, Tuple, Dict

@dataclass
class EconomicParameters:
    """Economic model parameters"""
    rho: float = 0.02  # Pure time preference
    r: float = 0.02  # Real interest rate
    beta: float = 0.980392156862745  # Discount factor (1/(1+r))
    zeta1: float = 0.5  # Cobb-Douglas coefficient on health in productivity
    zeta2: float = 0.5  # Cobb-Douglas coefficient on health in productivity
    A: float = 50.0  # TFP
    sigma: float = 1/1.5  # Elasticity of intertemporal substitution
    eta: float = 1.509  # Elasticity of substitution between consumption and leisure
    phi: float = 0.224  # Weight on consumption in z
    z_z0: float = 0.1  # Ratio z to z0 at age 50
    z0: float = 600  # Subsistence level of consumption-leisure composite
    WageChild: float = 6.975478  # Wage pre-graduation
    MaxHours: float = 4000  # Maximum number of hours worked in a year
    AgeGrad: int = 20  # Graduation age
    AgeRetire: int = 65  # Retirement age

class EconomicModel:
    """Core economic model for household optimization and utility calculations"""
    
    def __init__(self, params: EconomicParameters):
        self.params = params
    
    def discount_factor(self, age: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Compute discount factor for given age(s)
        
        Args:
            age: Age or array of ages
            
        Returns:
            Discount factor(s)
        """
        return (1 / (1 + self.params.r)) ** age
    
    def experience(self, age: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Compute work experience for given age(s)
        
        Args:
            age: Age or array of ages
            
        Returns:
            Experience value(s)
        """
        age = np.asarray(age)
        experience = np.zeros_like(age)
        
        for i, a in enumerate(age):
            if a > self.params.AgeGrad:
                experience[i] = min(a, 50.0)
        
        return experience if isinstance(age, np.ndarray) else float(experience[0])
    
    def wage(self, age: Union[float, np.ndarray], health: Union[float, np.ndarray],
             prod_age: bool = False) -> Union[float, np.ndarray]:
        """
        Compute wage for given age(s) and health level(s)
        
        Args:
            age: Age or array of ages
            health: Health level(s)
            prod_age: Whether to use productivity-based wage calculation
            
        Returns:
            Wage value(s)
        """
        age = np.asarray(age)
        health = np.asarray(health)
        wage = np.zeros_like(age)
        
        for i, (a, h) in enumerate(zip(age, health)):
            if a <= self.params.AgeGrad:
                wage[i] = self.params.WageChild
            elif a > self.params.AgeGrad:
                if prod_age:
                    # Productivity-based wage
                    X = self.experience(a)
                    wage[i] = self.params.A * (X ** self.params.zeta1) * (h ** self.params.zeta2)
                else:
                    # Experience-based wage
                    wage[i] = 1.35 * np.log(a - self.params.AgeGrad) * self.params.WageChild + self.params.WageChild
                
                # Retirement penalty
                if a > self.params.AgeRetire:
                    if not prod_age:
                        w_retire = 1.35 * np.log(self.params.AgeRetire - self.params.AgeGrad) * self.params.WageChild + self.params.WageChild
                        h_retire = health[min(i, len(health)-1)]  # Use current health as proxy
                        wage[i] = w_retire * 0.68 * (h / h_retire) ** 1.75
                    else:
                        wage[i] *= 0.68
                
                # Minimum wage constraint
                if wage[i] < 1e-7 and a > self.params.AgeRetire:
                    wage[i] = 1e-7
        
        return wage if isinstance(age, np.ndarray) else float(wage[0])
    
    def consumption_leisure_composite(self, consumption: Union[float, np.ndarray], 
                                     leisure: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Compute consumption-leisure composite z
        
        Args:
            consumption: Consumption level(s)
            leisure: Leisure level(s)
            
        Returns:
            Composite z value(s)
        """
        return (self.params.phi * consumption ** ((self.params.eta - 1) / self.params.eta) + 
                (1 - self.params.phi) * leisure ** ((self.params.eta - 1) / self.params.eta)) ** (self.params.eta / (self.params.eta - 1))
    
    def marginal_utility_consumption(self, consumption: Union[float, np.ndarray], 
                                   leisure: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Compute marginal utility of consumption
        
        Args:
            consumption: Consumption level(s)
            leisure: Leisure level(s)
            
        Returns:
            Marginal utility of consumption
        """
        z = self.consumption_leisure_composite(consumption, leisure)
        zc = self.params.phi * (z / consumption) ** (1 / self.params.eta)
        return zc * (z ** (-1 / self.params.sigma))
    
    def marginal_utility_leisure(self, consumption: Union[float, np.ndarray], 
                               leisure: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Compute marginal utility of leisure
        
        Args:
            consumption: Consumption level(s)
            leisure: Leisure level(s)
            
        Returns:
            Marginal utility of leisure
        """
        z = self.consumption_leisure_composite(consumption, leisure)
        zl = (1 - self.params.phi) * (z / leisure) ** (1 / self.params.eta)
        return zl * (z ** (-1 / self.params.sigma))
    
    def instantaneous_utility(self, consumption: Union[float, np.ndarray], 
                             leisure: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Compute instantaneous utility
        
        Args:
            consumption: Consumption level(s)
            leisure: Leisure level(s)
            
        Returns:
            Instantaneous utility
        """
        z = self.consumption_leisure_composite(consumption, leisure)
        return (self.params.sigma / (self.params.sigma - 1)) * (z ** ((self.params.sigma - 1) / self.params.sigma) - 
                                                               self.params.z0 ** ((self.params.sigma - 1) / self.params.sigma))
    
    def solve_household_optimization(self, ages: np.ndarray, health: np.ndarray, 
                                   wages: np.ndarray, survival: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Solve household optimization problem
        
        Args:
            ages: Array of ages
            health: Array of health levels
            wages: Array of wage levels
            survival: Array of survival probabilities
            
        Returns:
            Dictionary with consumption, leisure, and other variables
        """
        n_ages = len(ages)
        consumption = np.zeros(n_ages)
        leisure = np.zeros(n_ages)
        
        # Initial guess for consumption at graduation age
        c0_guess = 15000.0
        
        def budget_constraint(c0):
            """Budget constraint function"""
            consumption[0] = c0
            
            # Forward solve consumption and leisure
            for i in range(1, n_ages):
                if i == self.params.AgeGrad:
                    consumption[i] = c0
                else:
                    # Euler equation for consumption
                    if i > 0:
                        dH = (health[i] - health[i-1]) / max(health[i-1], 1e-7)
                        dW = (wages[i] - wages[i-1]) / max(wages[i-1], 1e-7)
                        
                        # Intratemporal condition for leisure
                        leisure[i-1] = consumption[i-1] * ((self.params.phi / (1 - self.params.phi)) * wages[i-1]) ** (-self.params.eta)
                        leisure[i-1] = min(leisure[i-1], self.params.MaxHours)
                        
                        # Consumption growth
                        if leisure[i-1] < self.params.MaxHours:
                            cdot = self.params.sigma * (self.params.r - self.params.rho) + self.params.sigma * dH + (self.params.eta - self.params.sigma) * (leisure[i-1] / self.params.MaxHours) * dW
                        else:
                            cdot = (1 + ((self.params.sigma - self.params.eta) / self.params.eta) * (leisure[i-1] / self.params.MaxHours)) ** (-1) * (self.params.sigma * (self.params.r - self.params.rho) + self.params.sigma * dH)
                        
                        consumption[i] = max(consumption[i-1] * (1 + cdot), 1e-7)
                
                # Leisure choice
                leisure[i] = consumption[i] * ((self.params.phi / (1 - self.params.phi)) * wages[i]) ** (-self.params.eta)
                leisure[i] = min(leisure[i], self.params.MaxHours)
            
            # Calculate budget constraint
            expenditure = np.sum(consumption * survival * self.discount_factor(ages))
            income = np.sum((self.params.MaxHours - leisure) * wages * survival * self.discount_factor(ages))
            
            return expenditure - income
        
        # Solve for initial consumption
        try:
            c0_solution = fsolve(budget_constraint, c0_guess)[0]
        except:
            # Try different initial guesses
            for guess in [50000.0, 100000.0, 46450.0, 1770.0, 17000.0]:
                try:
                    c0_solution = fsolve(budget_constraint, guess)[0]
                    break
                except:
                    continue
            else:
                raise ValueError("Household optimization failed to converge")
        
        # Final solution
        budget_constraint(c0_solution)
        
        # Calculate other variables
        income = (self.params.MaxHours - leisure) * wages
        savings = income - consumption
        assets = np.cumsum(savings * survival * self.discount_factor(ages))
        
        return {
            'consumption': consumption,
            'leisure': leisure,
            'income': income,
            'savings': savings,
            'assets': assets
        }
    
    def compute_value_of_life(self, ages: np.ndarray, health: np.ndarray, 
                            consumption: np.ndarray, leisure: np.ndarray,
                            survival: np.ndarray) -> np.ndarray:
        """
        Compute value of statistical life at each age
        
        Args:
            ages: Array of ages
            health: Array of health levels
            consumption: Array of consumption levels
            leisure: Array of leisure levels
            survival: Array of survival probabilities
            
        Returns:
            Array of value of statistical life
        """
        n_ages = len(ages)
        vsl = np.zeros(n_ages)
        
        # Compute utility and marginal utility
        utility = self.instantaneous_utility(consumption, leisure)
        mu_c = self.marginal_utility_consumption(consumption, leisure)
        
        for i in range(n_ages):
            # Remaining lifetime value
            remaining_ages = ages[i:]
            remaining_health = health[i:]
            remaining_utility = utility[i:]
            remaining_survival = survival[i:] / max(survival[i], 1e-7)
            remaining_discount = self.discount_factor(remaining_ages - ages[i])
            
            if len(remaining_ages) > 0:
                vsl[i] = np.sum(remaining_utility * remaining_health * remaining_survival * remaining_discount) / max(mu_c[i], 1e-7)
            
            if np.isnan(vsl[i]) or np.isinf(vsl[i]):
                vsl[i] = 0.0
        
        return vsl
    
    def compute_willingness_to_pay(self, ages: np.ndarray, health: np.ndarray,
                                 consumption: np.ndarray, leisure: np.ndarray,
                                 survival: np.ndarray, health_grad: np.ndarray,
                                 survival_grad: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute willingness to pay for health and survival improvements
        
        Args:
            ages: Array of ages
            health: Array of health levels
            consumption: Array of consumption levels
            leisure: Array of leisure levels
            survival: Array of survival probabilities
            health_grad: Array of health gradients
            survival_grad: Array of survival gradients
            
        Returns:
            Tuple of (WTP_S, WTP_H, WTP_total)
        """
        n_ages = len(ages)
        wtp_s = np.zeros(n_ages)
        wtp_h = np.zeros(n_ages)
        
        # Compute utility and marginal utility
        utility = self.instantaneous_utility(consumption, leisure)
        mu_c = self.marginal_utility_consumption(consumption, leisure)
        
        for i in range(n_ages):
            # WTP for survival improvements
            remaining_ages = ages[i:]
            remaining_utility = utility[i:]
            remaining_survival_grad = survival_grad[i:]
            remaining_discount = self.discount_factor(remaining_ages - ages[i])
            
            if len(remaining_ages) > 0:
                wtp_s[i] = np.sum(remaining_utility * remaining_survival_grad * remaining_discount) / max(self.discount_factor(ages[i]), 1e-7)
            
            # WTP for health improvements
            remaining_health = health[i:]
            remaining_survival = survival[i:] / max(survival[i], 1e-7)
            
            if len(remaining_ages) > 0:
                # Health impact on utility
                health_impact = (health_grad[i:] / health[i:]) * utility[i:]
                wtp_h[i] = np.sum(health_impact * remaining_survival * remaining_discount) / max(self.discount_factor(ages[i]), 1e-7)
            
            # Handle numerical issues
            if np.isnan(wtp_s[i]) or np.isinf(wtp_s[i]):
                wtp_s[i] = 0.0
            if np.isnan(wtp_h[i]) or np.isinf(wtp_h[i]):
                wtp_h[i] = 0.0
        
        wtp_total = wtp_s + wtp_h
        return wtp_s, wtp_h, wtp_total
