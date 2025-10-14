#!/usr/bin/env python3
"""Model Configuration and Parameter Documentation.

This module centralizes all model parameters with their justifications,
sources, and default values using Pydantic for validation and documentation.

Key features:
- Type validation and coercion
- Immutable configuration (frozen=True)
- Rich metadata with sources and literature references
- Automatic JSON schema generation
- Programmatic access to documentation

Parameters are organized by category:
- Economic: Preferences and market parameters
- Lifecycle: Age-specific parameters  
- VSL Calibration: Reference VSL and income elasticity
- Calibration: Optimization settings
- Interpolation: Data processing parameters

Usage:
    >>> from config import ModelConfig
    >>> config = ModelConfig()
    >>> print(config.economic.rho)  # 0.02
    >>> print(config.economic.model_fields['rho'].description)
    >>> config.economic.describe('rho')  # Print full documentation
    >>> model = LifeCycleModel(**config.economic.model_dump())
"""

from pydantic import BaseModel, Field, field_validator
from typing import Dict, Any


# ============================================================================
# Economic Parameters (Preferences and Markets)
# ============================================================================

class EconomicParameters(BaseModel):
    """Economic parameters for lifecycle model.
    
    All parameters include validation, documentation, and literature sources.
    """
    
    model_config = {'frozen': True, 'extra': 'forbid'}
    
    # === Time Preference ===
    
    rho: float = Field(
        default=0.02,
        ge=0.0,
        le=0.1,
        description="Pure time preference rate. Households discount future utility at 2% per year, separate from survival probability discounting.",
        json_schema_extra={
            'units': 'rate per year',
            'source': 'Standard value in lifecycle models',
            'interpretation': 'Measures impatience - how much households prefer current over future utility',
            'literature': ['Gourinchas & Parker (2002)', 'Carroll (1997)'],
        }
    )
    
    r: float = Field(
        default=0.02,
        ge=0.0,
        le=0.1,
        description="Real interest rate. Used to discount monetary values and compute present values.",
        json_schema_extra={
            'units': 'rate per year',
            'source': 'Historical average real return on safe assets',
            'interpretation': 'Real return on savings, determines intertemporal budget constraint',
        }
    )
    
    # === Utility Function Parameters ===
    
    sigma: float = Field(
        default=1 / 1.5,  # ≈ 0.667
        gt=0.0,
        le=5.0,
        description="Elasticity of intertemporal substitution (IES = 1/σ = 1.5). A 1% increase in future consumption relative to current consumption leads to a 1.5% change in marginal utility ratio.",
        json_schema_extra={
            'units': 'dimensionless',
            'source': 'Murphy & Topel (2006)',
            'interpretation': 'Willingness to substitute consumption over time. Higher σ → less willing to substitute',
            'literature': [
                'Murphy & Topel (2006): σ = 2/3 ≈ 0.667',
                'Gourinchas & Parker (2002): IES ≈ 1.5-2.0 for younger households'
            ],
        }
    )
    
    eta: float = Field(
        default=1.509,
        gt=0.0,
        le=10.0,
        description="Elasticity of substitution between consumption and leisure. Measures how easily households substitute between consumption and leisure.",
        json_schema_extra={
            'units': 'dimensionless',
            'source': 'Calibrated to match labor supply elasticity of ~0.5',
            'interpretation': 'Higher η → easier substitution between consumption and leisure',
            'literature': [
                'Chetty et al. (2011): labor supply elasticity ≈ 0.5 (intensive)',
                'MaCurdy (1981): intertemporal elasticity ≈ 0.1-0.4'
            ],
        }
    )
    
    phi: float = Field(
        default=0.224,
        gt=0.0,
        lt=1.0,
        description="Weight on consumption in utility composite. φ = 0.224 implies leisure weight = 0.776, suggesting leisure is more valuable than consumption in utility.",
        json_schema_extra={
            'units': 'dimensionless',
            'source': 'Calibrated to match consumption/leisure ratios',
            'interpretation': 'Higher φ → relatively more consumption (less leisure)',
            'notes': 'Reduced-form parameter combining preferences and constraints',
        }
    )
    
    # === Productivity Parameters ===
    
    A: float = Field(
        default=50.0,
        gt=0.0,
        description="Total factor productivity (TFP) in wage function. Scaling factor for wage = A × experience^ζ1 × health^ζ2.",
        json_schema_extra={
            'units': 'USD per year (real terms)',
            'source': 'Calibrated to match average wage levels',
            'interpretation': 'Overall productivity level in the economy',
        }
    )
    
    zeta1: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Experience elasticity in productivity. A 1% increase in experience increases wages by 0.5%.",
        json_schema_extra={
            'units': 'dimensionless',
            'source': 'Standard Mincer equation coefficient',
            'interpretation': 'Returns to experience in wage function',
            'literature': [
                'Mincer (1974): returns to experience ≈ 5-10% per year when young',
                'ζ1 = 0.5 → experience^0.5 profile peaking at 40-50 years'
            ],
        }
    )
    
    zeta2: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Health elasticity in productivity. A 1% increase in health increases wages by 0.5%.",
        json_schema_extra={
            'units': 'dimensionless',
            'source': 'Health economics literature',
            'interpretation': 'Returns to health in wage function',
            'literature': [
                'Currie & Madrian (1999): significant health-productivity effects',
                'Strauss & Thomas (1998): health elasticity ≈ 0.3-0.6'
            ],
        }
    )
    
    # === Subsistence Consumption ===
    
    z_z0: float = Field(
        default=0.1,
        gt=0.0,
        lt=1.0,
        description="Ratio of utility composite to subsistence at age 50. At age 50, actual consumption composite is 10% above subsistence level.",
        json_schema_extra={
            'units': 'dimensionless',
            'source': 'Calibration parameter, Murphy & Topel (2006) show 0.05, 0.10, and 0.20',
            'interpretation': 'Creates realistic curvature in lifecycle profiles',
        }
    )
    
    z0: float = Field(
        default=600.0,
        gt=0.0,
        description="Subsistence level of consumption-leisure composite. Below this level, utility approaches negative infinity.",
        json_schema_extra={
            'units': 'USD per year (real terms)',
            'source': 'Calibrated to match lifecycle consumption patterns',
            'interpretation': 'Minimum consumption-leisure bundle needed for survival',
        }
    )
    
    def describe(self, param_name: str) -> None:
        """Print comprehensive documentation for a parameter.
        
        Args:
            param_name: Name of the parameter to describe
        """
        if param_name not in self.model_fields:
            raise ValueError(f"Unknown parameter: {param_name}")
        
        field_info = self.model_fields[param_name]
        value = getattr(self, param_name)
        extra = field_info.json_schema_extra or {}
        
        print(f"\n{'=' * 70}")
        print(f"Parameter: {param_name}")
        print(f"{'=' * 70}")
        print(f"Value: {value}")
        if 'units' in extra:
            print(f"Units: {extra['units']}")
        print(f"\nDescription:")
        print(f"  {field_info.description}")
        if 'source' in extra:
            print(f"\nSource:")
            print(f"  {extra['source']}")
        if 'interpretation' in extra:
            print(f"\nInterpretation:")
            print(f"  {extra['interpretation']}")
        if 'literature' in extra:
            print(f"\nLiterature:")
            for ref in extra['literature']:
                print(f"  • {ref}")
        if 'notes' in extra:
            print(f"\nNotes:")
            print(f"  {extra['notes']}")
        print(f"{'=' * 70}\n")


# ============================================================================
# Lifecycle Parameters (Age-Specific)
# ============================================================================

class LifecycleParameters(BaseModel):
    """Lifecycle parameters for age-specific transitions."""
    
    model_config = {'frozen': True, 'extra': 'forbid'}
    
    WageChild: float = Field(
        default=6.975478,
        gt=0.0,
        description="Child wage (before graduation). Calibrated for each country-year based on GDP per capita and target VSL.",
        json_schema_extra={
            'units': 'thousands USD per year',
            'source': 'Calibrated to match country-specific VSL at age 50',
            'interpretation': 'Wage earned before graduation, much lower than adult wages',
            'notes': 'Re-calibrated for each country-year. Default value is U.S. 2019 reference.',
        }
    )
    
    AgeGrad: int = Field(
        default=20,
        ge=16,
        le=30,
        description="Graduation age (end of education). Before this age, individuals earn child wage. After, wages increase with experience.",
        json_schema_extra={
            'units': 'years',
            'source': 'Typical age of completing higher education',
            'literature': ['OECD: average tertiary completion age = 22-26 years'],
        }
    )
    
    AgeRetire: int = Field(
        default=65,
        ge=55,
        le=75,
        description="Retirement age. After this age, individuals face retirement penalty on wages but can adjust labor supply.",
        json_schema_extra={
            'units': 'years',
            'source': 'Statutory retirement age in OECD countries',
            'literature': ['OECD: effective retirement age = 63-66 (men), 61-64 (women)'],
        }
    )
    
    MaxHours: float = Field(
        default=4000.0,
        gt=0.0,
        le=8760.0,  # 365 * 24
        description="Maximum hours available per year. Total time endowment for labor/leisure choice.",
        json_schema_extra={
            'units': 'hours per year',
            'source': 'Calibrated to match labor supply patterns',
            'interpretation': 'Actual hours worked = MaxHours - Leisure',
            'notes': '365 days × 16 waking hours ≈ 5,840. MaxHours = 4,000 accounts for non-discretionary time.',
        }
    )
    
    def describe(self, param_name: str) -> None:
        """Print comprehensive documentation for a parameter."""
        if param_name not in self.model_fields:
            raise ValueError(f"Unknown parameter: {param_name}")
        
        field_info = self.model_fields[param_name]
        value = getattr(self, param_name)
        extra = field_info.json_schema_extra or {}
        
        print(f"\n{'=' * 70}")
        print(f"Parameter: {param_name}")
        print(f"{'=' * 70}")
        print(f"Value: {value}")
        if 'units' in extra:
            print(f"Units: {extra['units']}")
        print(f"\nDescription:")
        print(f"  {field_info.description}")
        if 'source' in extra:
            print(f"\nSource:")
            print(f"  {extra['source']}")
        if 'interpretation' in extra:
            print(f"\nInterpretation:")
            print(f"  {extra['interpretation']}")
        if 'literature' in extra:
            print(f"\nLiterature:")
            for ref in extra['literature']:
                print(f"  • {ref}")
        if 'notes' in extra:
            print(f"\nNotes:")
            print(f"  {extra['notes']}")
        print(f"{'=' * 70}\n")


# ============================================================================
# VSL Calibration Parameters
# ============================================================================

class VSLCalibrationParameters(BaseModel):
    """Parameters for Value of Statistical Life calibration across countries."""
    
    model_config = {'frozen': True, 'extra': 'forbid'}
    
    VSL_ref: float = Field(
        default=11.5e6,
        gt=0.0,
        description="Reference Value of Statistical Life (U.S. 2019). Amount individuals are willing to pay for marginal reduction in mortality risk.",
        json_schema_extra={
            'units': 'USD (2019 dollars)',
            'source': 'EPA recommended VSL for policy analysis',
            'literature': [
                'EPA (2019): $11.5 million',
                'Viscusi & Aldy (2003): $7-9 million (2000 dollars)',
                'Adjusted for inflation: $10-12 million'
            ],
        }
    )
    
    GDP_pc_ref: float = Field(
        default=65349.36,
        gt=0.0,
        description="Reference GDP per capita (U.S. 2019). Used to scale VSL across countries: VSL_i = VSL_ref × (GDP_pc_i / GDP_pc_ref)^η.",
        json_schema_extra={
            'units': 'USD per person per year (2019)',
            'source': 'World Bank data for U.S. 2019',
        }
    )
    
    VSL_eta: float = Field(
        default=1.0,
        ge=0.0,
        le=2.0,
        description="Income elasticity of VSL. A 1% increase in GDP per capita leads to 1% increase in VSL (proportional relationship).",
        json_schema_extra={
            'units': 'dimensionless',
            'source': 'Meta-analysis of VSL studies',
            'interpretation': 'How VSL scales with income across countries',
            'literature': [
                'Viscusi & Aldy (2003): η ≈ 0.5-0.6',
                'OECD (2012): η ≈ 0.8-1.0',
                'Robinson et al. (2019): η ≈ 1.0-1.2'
            ],
            'notes': 'η = 1.0 implies VSL grows proportionally with income',
        }
    )
    
    def describe(self, param_name: str) -> None:
        """Print comprehensive documentation for a parameter."""
        if param_name not in self.model_fields:
            raise ValueError(f"Unknown parameter: {param_name}")
        
        field_info = self.model_fields[param_name]
        value = getattr(self, param_name)
        extra = field_info.json_schema_extra or {}
        
        print(f"\n{'=' * 70}")
        print(f"Parameter: {param_name}")
        print(f"{'=' * 70}")
        print(f"Value: {value}")
        if 'units' in extra:
            print(f"Units: {extra['units']}")
        print(f"\nDescription:")
        print(f"  {field_info.description}")
        if 'source' in extra:
            print(f"\nSource:")
            print(f"  {extra['source']}")
        if 'interpretation' in extra:
            print(f"\nInterpretation:")
            print(f"  {extra['interpretation']}")
        if 'literature' in extra:
            print(f"\nLiterature:")
            for ref in extra['literature']:
                print(f"  • {ref}")
        if 'notes' in extra:
            print(f"\nNotes:")
            print(f"  {extra['notes']}")
        print(f"{'=' * 70}\n")


# ============================================================================
# Calibration Settings
# ============================================================================

class CalibrationParameters(BaseModel):
    """Parameters controlling model calibration behavior."""
    
    model_config = {'frozen': True, 'extra': 'forbid'}
    
    target_vsl_age: int = Field(
        default=50,
        ge=30,
        le=70,
        description="Age at which to match target VSL during calibration. Age 50 is representative of working population with robust VSL estimates.",
        json_schema_extra={
            'units': 'years',
            'source': 'Standard practice in VSL literature. Murphy & Topel (2006) talk about 50.',
            'interpretation': 'Prime working age, less sensitive to edge effects',
        }
    )
    
    tolerance: float = Field(
        default=0.01,
        gt=0.0,
        lt=0.1,
        description="Relative tolerance for calibration convergence (1%). Calibration succeeds if |VSL_model - VSL_target| / VSL_target < 1%.",
        json_schema_extra={
            'units': 'dimensionless (fraction)',
            'interpretation': 'Tighter tolerance increases computation time, looser may yield inaccurate results',
        }
    )
    
    fallback_tolerance: float = Field(
        default=0.10,
        gt=0.0,
        lt=1.0,
        description="Fallback tolerance for accepting imperfect calibration when strict tolerance cannot be met.",
        json_schema_extra={
            'units': 'dimensionless (fraction)',
            'interpretation': 'If strict tolerance fails, accept solution within this threshold',
            'notes': 'Prevents spurious failures on difficult-to-calibrate country-years',
        }
    )
    
    max_time: float = Field(
        default=5.0,
        gt=0.0,
        le=60.0,
        description="Maximum time for calibration (seconds). Prevents infinite loops while allowing sufficient time for convergence.",
        json_schema_extra={
            'units': 'seconds',
            'interpretation': 'If not converged, accept best solution or raise exception',
        }
    )
    
    wage_child_bounds: tuple[float, float] = Field(
        default=(0.1, 100.0),
        description="Bounds for WageChild search during calibration. Prevents unrealistic parameter values.",
        json_schema_extra={
            'units': 'thousands USD per year',
            'interpretation': 'Min and max reasonable values for child wage parameter',
        }
    )
    
    verbose: bool = Field(
        default=False,
        description="Whether to print calibration progress. Set to True for debugging or monitoring.",
    )
    
    @field_validator('wage_child_bounds')
    @classmethod
    def validate_bounds(cls, v):
        """Ensure bounds are valid (min < max)."""
        if v[0] >= v[1]:
            raise ValueError(f"wage_child_bounds must have min < max, got {v}")
        if v[0] <= 0:
            raise ValueError(f"wage_child_bounds min must be > 0, got {v[0]}")
        return v
    
    def describe(self, param_name: str) -> None:
        """Print comprehensive documentation for a parameter."""
        if param_name not in self.model_fields:
            raise ValueError(f"Unknown parameter: {param_name}")
        
        field_info = self.model_fields[param_name]
        value = getattr(self, param_name)
        extra = field_info.json_schema_extra or {}
        
        print(f"\n{'=' * 70}")
        print(f"Parameter: {param_name}")
        print(f"{'=' * 70}")
        print(f"Value: {value}")
        if 'units' in extra:
            print(f"Units: {extra['units']}")
        print(f"\nDescription:")
        print(f"  {field_info.description}")
        if 'source' in extra:
            print(f"\nSource:")
            print(f"  {extra['source']}")
        if 'interpretation' in extra:
            print(f"\nInterpretation:")
            print(f"  {extra['interpretation']}")
        if 'notes' in extra:
            print(f"\nNotes:")
            print(f"  {extra['notes']}")
        print(f"{'=' * 70}\n")


# ============================================================================
# Data Processing Parameters
# ============================================================================

class InterpolationParameters(BaseModel):
    """Parameters for data interpolation and extrapolation."""
    
    model_config = {'frozen': True, 'extra': 'forbid'}
    
    max_age: int = Field(
        default=240,
        ge=100,
        le=500,
        description="Maximum age for interpolation. Extend health and survival curves beyond observed data to ensure survival reaches zero.",
        json_schema_extra={
            'units': 'years',
            'source': 'Chosen to ensure complete lifecycle',
            'interpretation': 'GBD data extends to 95+, extrapolate to this age',
        }
    )
    
    health_floor: float = Field(
        default=0.1,
        gt=0.0,
        lt=1.0,
        description="Minimum health level (10% of perfect health). Prevents division by zero in wage function.",
        json_schema_extra={
            'units': 'fraction of perfect health',
            'source': 'Numerical stability requirement',
            'interpretation': 'Health cannot fall below this level',
            'notes': 'Floor of 0.1 → wages at oldest ages are ≥ 31.6% (= 0.1^0.5) of prime-age wages',
        }
    )
    
    old_age: int = Field(
        default=60,
        ge=50,
        le=80,
        description="Age at which to start health trend estimation for extrapolation. Health trends are most linear after this age.",
        json_schema_extra={
            'units': 'years',
            'source': 'Empirical observation',
            'interpretation': 'Fit linear trend to ages 60-95 for extrapolation',
        }
    )
    
    min_trend_size: int = Field(
        default=20,
        ge=5,
        le=50,
        description="Minimum number of age points needed to reliably estimate linear health decline trend.",
        json_schema_extra={
            'units': 'number of ages',
            'interpretation': 'Ensures robust trend estimation',
        }
    )
    
    def describe(self, param_name: str) -> None:
        """Print comprehensive documentation for a parameter."""
        if param_name not in self.model_fields:
            raise ValueError(f"Unknown parameter: {param_name}")
        
        field_info = self.model_fields[param_name]
        value = getattr(self, param_name)
        extra = field_info.json_schema_extra or {}
        
        print(f"\n{'=' * 70}")
        print(f"Parameter: {param_name}")
        print(f"{'=' * 70}")
        print(f"Value: {value}")
        if 'units' in extra:
            print(f"Units: {extra['units']}")
        print(f"\nDescription:")
        print(f"  {field_info.description}")
        if 'source' in extra:
            print(f"\nSource:")
            print(f"  {extra['source']}")
        if 'interpretation' in extra:
            print(f"\nInterpretation:")
            print(f"  {extra['interpretation']}")
        if 'notes' in extra:
            print(f"\nNotes:")
            print(f"  {extra['notes']}")
        print(f"{'=' * 70}\n")


# ============================================================================
# Main Configuration Class
# ============================================================================

class ModelConfig(BaseModel):
    """Complete model configuration with all parameter categories.
    
    This is the main entry point for accessing all model parameters.
    Parameters are organized into logical categories and include full
    documentation with sources and literature references.
    
    Usage:
        >>> config = ModelConfig()
        >>> config.economic.rho  # Access parameter value
        >>> config.economic.describe('rho')  # Print documentation
        >>> config.to_dict()  # Get all parameters as nested dict
        >>> model = LifeCycleModel(**config.economic.model_dump())
    """
    
    model_config = {'frozen': True}
    
    economic: EconomicParameters = Field(
        default_factory=EconomicParameters,
        description="Economic parameters (preferences and markets)"
    )
    
    lifecycle: LifecycleParameters = Field(
        default_factory=LifecycleParameters,
        description="Lifecycle parameters (age-specific)"
    )
    
    vsl_calibration: VSLCalibrationParameters = Field(
        default_factory=VSLCalibrationParameters,
        description="VSL calibration parameters"
    )
    
    calibration: CalibrationParameters = Field(
        default_factory=CalibrationParameters,
        description="Calibration settings"
    )
    
    interpolation: InterpolationParameters = Field(
        default_factory=InterpolationParameters,
        description="Data interpolation parameters"
    )
    
    def to_dict(self) -> Dict[str, Dict[str, Any]]:
        """Export all parameters as nested dictionary.
        
        Returns:
            Dictionary with parameter categories as keys, each containing
            parameter name-value pairs.
        """
        return {
            'economic': self.economic.model_dump(),
            'lifecycle': self.lifecycle.model_dump(),
            'vsl_calibration': self.vsl_calibration.model_dump(),
            'calibration': self.calibration.model_dump(),
            'interpolation': self.interpolation.model_dump(),
        }
    
    def describe_all(self) -> None:
        """Print documentation for all parameters in all categories."""
        for category_name in ['economic', 'lifecycle', 'vsl_calibration', 'calibration', 'interpolation']:
            category = getattr(self, category_name)
            print(f"\n{'#' * 70}")
            print(f"# {category_name.upper().replace('_', ' ')}")
            print(f"{'#' * 70}")
            for param_name in category.model_fields.keys():
                category.describe(param_name)


# ============================================================================
# Backward Compatibility (Old Dictionary Interface)
# ============================================================================

# Create default configuration instance
_default_config = ModelConfig()

# Export old-style dictionaries for backward compatibility
ECONOMIC_PARAMS = _default_config.economic.model_dump()
LIFECYCLE_PARAMS = _default_config.lifecycle.model_dump()
VSL_CALIBRATION = _default_config.vsl_calibration.model_dump()
CALIBRATION_PARAMS = _default_config.calibration.model_dump()
INTERPOLATION_PARAMS = _default_config.interpolation.model_dump()


def get_all_params() -> Dict[str, Dict[str, Any]]:
    """Get all model parameters as a single dictionary (legacy interface).
    
    Returns:
        Dictionary with all parameter categories combined.
    """
    return _default_config.to_dict()


# ============================================================================
# Command-line Interface
# ============================================================================

if __name__ == "__main__":
    """Print all parameters when run as script."""
    config = ModelConfig()
    
    print("=" * 80)
    print("MODEL PARAMETERS")
    print("=" * 80)
    print("\nUsing Pydantic for validation and documentation")
    print("All parameters include type checking, bounds validation, and metadata\n")
    
    for category_name, category in [
        ('ECONOMIC', config.economic),
        ('LIFECYCLE', config.lifecycle),
        ('VSL CALIBRATION', config.vsl_calibration),
        ('CALIBRATION', config.calibration),
        ('INTERPOLATION', config.interpolation),
    ]:
        print(f"\n{category_name}")
        print("-" * 80)
        for param_name, value in category.model_dump().items():
            print(f"  {param_name:20s} = {value}")
    
    print("\n" + "=" * 80)
    print("\nFor detailed documentation, use:")
    print("  >>> from config import ModelConfig")
    print("  >>> config = ModelConfig()")
    print("  >>> config.economic.describe('rho')")
    print("  >>> config.describe_all()  # All parameters")
    print("=" * 80)
