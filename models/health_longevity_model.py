#!/usr/bin/env python3
"""
Unified Health and Longevity Model

This module combines the biological and economic models into a single framework
that can handle all the scenarios from the original Julia scripts.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from dataclasses import dataclass
import warnings
from scipy.optimize import fsolve

from models.biological_model import BiologicalModel, BiologicalParameters
from models.economic_model import EconomicModel, EconomicParameters

@dataclass
class ModelOptions:
    """Options for model configuration"""
    param: str = 'none'  # Parameter being varied
    bio_pars0: Optional[BiologicalParameters] = None  # Baseline biological parameters
    age_start: float = 0.0  # Age at which intervention starts
    wolverine: float = 0.0  # Wolverine reset parameter
    no_compress: bool = False  # Whether to prevent compression
    prod_age: bool = False  # Whether to use productivity-based wages
    redo_z0: bool = False  # Whether to recompute z0
    age_range: list = None  # Ages for analysis
    age_grad: int = 20  # Graduation age
    age_retire: int = 65  # Retirement age
    phasing: int = 1  # Number of phases for gradual interventions
    
    def __post_init__(self):
        if self.age_range is None:
            self.age_range = [0, 20, 40, 60, 80]
        if self.bio_pars0 is None:
            self.bio_pars0 = BiologicalParameters()

class HealthLongevityModel:
    """Unified model combining biological and economic components"""
    
    def __init__(self, bio_params: BiologicalParameters = None, 
                 econ_params: EconomicParameters = None,
                 options: ModelOptions = None):
        """
        Initialize the unified model
        
        Args:
            bio_params: Biological parameters
            econ_params: Economic parameters  
            options: Model options
        """
        self.bio_params = bio_params or BiologicalParameters()
        self.econ_params = econ_params or EconomicParameters()
        self.options = options or ModelOptions()
        
        # Initialize sub-models
        self.bio_model = BiologicalModel(self.bio_params)
        self.econ_model = EconomicModel(self.econ_params)
        
        # Cache for computed variables
        self._cache = {}
    
    def compute_life_cycle_variables(self, ages: np.ndarray = None) -> pd.DataFrame:
        """
        Compute all life cycle variables for given ages
        
        Args:
            ages: Array of ages to compute variables for
            
        Returns:
            DataFrame with all life cycle variables
        """
        if ages is None:
            ages = np.arange(0, self.bio_params.MaxAge + 1)
        
        # Clear cache if parameters changed
        cache_key = f"lifecycle_{hash(str(self.bio_params))}_{hash(str(self.econ_params))}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        # Biological variables
        frailty = self.bio_model.frailty(ages, self.options.age_start, self.options.wolverine)
        disability = self.bio_model.disability(ages, self.options.age_start, self.options.wolverine)
        health = self.bio_model.health(ages, self.options.age_start, self.options.wolverine, self.options.no_compress)
        mortality = self.bio_model.mortality(ages, self.options.age_start, self.options.wolverine)
        
        # Survival probabilities
        survival = np.array([self.bio_model.survivor(0, a, self.options.age_start, self.options.wolverine) 
                            for a in ages])
        
        # Economic variables
        wages = self.econ_model.wage(ages, health, self.options.prod_age)
        
        # Solve household optimization
        household_vars = self.econ_model.solve_household_optimization(
            ages, health, wages, survival
        )
        
        # Compute value of statistical life
        vsl = self.econ_model.compute_value_of_life(
            ages, health, household_vars['consumption'], 
            household_vars['leisure'], survival
        )
        
        # Compute gradients if parameter is being varied
        health_grad = np.zeros_like(ages)
        survival_grad = np.zeros_like(ages)
        wage_grad = np.zeros_like(ages)
        
        if self.options.param != 'none':
            try:
                health_grad = self.bio_model.compute_gradients(
                    ages, self.options.param, self.options.age_start, self.options.wolverine
                )
                # For survival gradient, we need to compute it numerically
                h = 1e-6
                original_value = getattr(self.bio_params, self.options.param)
                
                setattr(self.bio_params, self.options.param, original_value + h)
                survival_plus = np.array([self.bio_model.survivor(0, a, self.options.age_start, self.options.wolverine) 
                                        for a in ages])
                
                setattr(self.bio_params, self.options.param, original_value - h)
                survival_minus = np.array([self.bio_model.survivor(0, a, self.options.age_start, self.options.wolverine) 
                                         for a in ages])
                
                setattr(self.bio_params, self.options.param, original_value)
                survival_grad = (survival_plus - survival_minus) / (2 * h)
                
                # Wage gradient
                wage_plus = self.econ_model.wage(ages, health + h * health_grad, self.options.prod_age)
                wage_minus = self.econ_model.wage(ages, health - h * health_grad, self.options.prod_age)
                wage_grad = (wage_plus - wage_minus) / (2 * h)
                
            except Exception as e:
                raise ValueError(f"Could not compute gradients: {e}")
        
        # Compute willingness to pay
        wtp_s, wtp_h, wtp_total = self.econ_model.compute_willingness_to_pay(
            ages, health, household_vars['consumption'], 
            household_vars['leisure'], survival, health_grad, survival_grad
        )
        
        # Create DataFrame
        df = pd.DataFrame({
            'age': ages,
            'frailty': frailty,
            'disability': disability,
            'health': health,
            'mortality': mortality,
            'survival': survival,
            'wage': wages,
            'consumption': household_vars['consumption'],
            'leisure': household_vars['leisure'],
            'income': household_vars['income'],
            'savings': household_vars['savings'],
            'assets': household_vars['assets'],
            'vsl': vsl,
            'health_grad': health_grad,
            'survival_grad': survival_grad,
            'wage_grad': wage_grad,
            'wtp_s': wtp_s,
            'wtp_h': wtp_h,
            'wtp_total': wtp_total
        })
        
        # Cache result
        self._cache[cache_key] = df
        return df
    
    def compute_life_expectancy(self) -> float:
        """Compute life expectancy"""
        return self.bio_model.life_expectancy(
            self.options.age_start, self.options.age_start, self.options.wolverine
        )
    
    def compute_healthy_life_expectancy(self) -> float:
        """Compute healthy life expectancy"""
        return self.bio_model.healthy_life_expectancy(
            self.options.age_start, self.options.age_start, self.options.wolverine
        )
    
    def run_scenario(self, scenario_config: Dict[str, Any]) -> pd.DataFrame:
        """
        Run a specific scenario with given configuration
        
        Args:
            scenario_config: Dictionary with scenario parameters
            
        Returns:
            DataFrame with scenario results
        """
        # Update model parameters based on scenario
        for key, value in scenario_config.items():
            if hasattr(self.bio_params, key):
                setattr(self.bio_params, key, value)
            elif hasattr(self.econ_params, key):
                setattr(self.econ_params, key, value)
            elif hasattr(self.options, key):
                setattr(self.options, key, value)
        
        # Clear cache
        self._cache.clear()
        
        # Recompute life cycle variables
        return self.compute_life_cycle_variables()
    
    def compute_parameter_sensitivity(self, param_name: str, param_values: np.ndarray) -> pd.DataFrame:
        """
        Compute sensitivity to parameter changes
        
        Args:
            param_name: Name of parameter to vary
            param_values: Array of parameter values to test
            
        Returns:
            DataFrame with sensitivity results
        """
        results = []
        
        for value in param_values:
            # Create scenario with this parameter value
            scenario_config = {param_name: value}
            df = self.run_scenario(scenario_config)
            
            # Compute summary statistics
            le = self.compute_life_expectancy()
            hle = self.compute_healthy_life_expectancy()
            
            results.append({
                'param_value': value,
                'life_expectancy': le,
                'healthy_life_expectancy': hle,
                'wtp_at_birth': df['wtp_total'].iloc[0] if len(df) > 0 else 0,
                'total_wtp': np.sum(df['wtp_total'] * df['survival']) if len(df) > 0 else 0
            })
        
        return pd.DataFrame(results)
    
    def compute_3d_surface(self, param1: str, param2: str, 
                         le_range: np.ndarray, hle_range: np.ndarray) -> pd.DataFrame:
        """
        Compute 3D surface for parameter combinations
        
        Args:
            param1: First parameter name
            param2: Second parameter name
            le_range: Range of life expectancy values
            hle_range: Range of healthy life expectancy values
            
        Returns:
            DataFrame with surface data
        """
        results = []
        
        for le in le_range:
            for hle in hle_range:
                if hle >= le:
                    continue  # Skip invalid combinations
                
                try:
                    # Find parameter values that achieve target LE and HLE
                    def objective(params):
                        p1, p2 = params
                        setattr(self.bio_params, param1, p1)
                        setattr(self.bio_params, param2, p2)
                        
                        current_le = self.compute_life_expectancy()
                        current_hle = self.compute_healthy_life_expectancy()
                        
                        return [current_le - le, current_hle - hle]
                    
                    # Solve for parameter values
                    initial_guess = [getattr(self.bio_params, param1), getattr(self.bio_params, param2)]
                    solution = fsolve(objective, initial_guess)
                    
                    if solution[0] > 0 and solution[1] > 0:  # Valid solution
                        # Compute WTP for this combination
                        df = self.compute_life_cycle_variables()
                        wtp = df['wtp_total'].iloc[0] if len(df) > 0 else 0
                        
                        results.append({
                            'le': le,
                            'hle': hle,
                            f'{param1}': solution[0],
                            f'{param2}': solution[1],
                            'wtp': wtp
                        })
                
                except Exception as e:
                    warnings.warn(f"Failed to solve for LE={le}, HLE={hle}: {e}")
                    continue
        
        return pd.DataFrame(results)
    
    def export_results(self, df: pd.DataFrame, filename: str):
        """Export results to CSV file"""
        df.to_csv(filename, index=False)
        print(f"Results exported to {filename}")
    
    def get_summary_statistics(self, df: pd.DataFrame) -> Dict[str, float]:
        """Get summary statistics from life cycle variables"""
        return {
            'life_expectancy': self.compute_life_expectancy(),
            'healthy_life_expectancy': self.compute_healthy_life_expectancy(),
            'total_wtp': np.sum(df['wtp_total'] * df['survival']),
            'wtp_at_birth': df['wtp_total'].iloc[0] if len(df) > 0 else 0,
            'max_wtp': df['wtp_total'].max(),
            'avg_health': df['health'].mean(),
            'avg_wage': df['wage'].mean()
        }
