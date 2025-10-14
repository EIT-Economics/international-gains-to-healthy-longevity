#!/usr/bin/env python3
"""
preprocess.py

This script processes raw data from `data` folder containing:

1. Global Burden of Disease (GBD)
2. UN World Population Prospects (WPP)  
3. World Bank (WB) GDP data

transforming it into `intermediate` data that is used by the other scripts.
"""

import pandas as pd
import numpy as np
from re import match
from pathlib import Path
from typing import Dict, List, Tuple
from plot import create_exploratory_plots

from paths import (
    DATA_DIR, GBD_DATA_DIR, WPP_DATA_DIR, WB_DATA_DIR,
    INTERMEDIATE_DIR, INTERMEDIATE_GBD_DIR, 
    INTERMEDIATE_WPP_DIR, INTERMEDIATE_WB_DIR
)


class DataProcessor:
    """Main class for processing health and economic data."""
        
    def process_gbd_data(self) -> Dict[str, pd.DataFrame]:
        """
        Process Global Burden of Disease data using specified filepaths.

        Specifically processes the following files in the GBD directory:
            "International_GBD_1990-2019_DATA_1.csv"
            "International_GBD_1990-2019_DATA_2.csv" 
            "International_GBD_1990-2019_DATA_3.csv"
            "GBD_population.csv"

        Writes the following files to the intermediate GBD directory:
            "mortality_rates.csv"
            "morbidity_rates.csv"

        Returns:
            Dict[str, pd.DataFrame]: Processed datasets
        """
        # Assert input data directory exists
        assert GBD_DATA_DIR.exists(), f"GBD directory {GBD_DATA_DIR} not found"
        print(f"Processing GBD data from {GBD_DATA_DIR}...")
        

        # Import and combine GBD data files
        gbd_files = [
            "International_GBD_1990-2019_DATA_1.csv",
            "International_GBD_1990-2019_DATA_2.csv", 
            "International_GBD_1990-2019_DATA_3.csv"
        ]
        
        gbd_dfs = []
        for file in gbd_files:
            file_path = GBD_DATA_DIR / file
            assert file_path.exists(), f"File {file_path} not found"
            df = pd.read_csv(file_path)
            gbd_dfs.append(df)
            
        # Combine all GBD data
        gbd_full = pd.concat(gbd_dfs, ignore_index=True)
        
        # Select relevant columns
        columns_to_keep = [
            "location_name", "measure_name", "age_name", "metric_name",
            "cause_name", "year", "val"
        ]
        gbd_full = gbd_full[columns_to_keep]
        
        # Clean age name variable
        gbd_full.loc[gbd_full['age_name'] == "<1 year", 'age_name'] = "0 to 0"
        gbd_full.loc[gbd_full['age_name'] == "95 plus", 'age_name'] = "95 to 99"
        
        # Shorten measure names
        gbd_full.loc[gbd_full['measure_name'] == "YLDs (Years Lived with Disability)", 'measure_name'] = "YLD"
        
        # Create numerical age variables from age bands
        age_split = gbd_full['age_name'].str.split(' to ', expand=True)
        gbd_full['age_low'] = pd.to_numeric(age_split[0])
        gbd_full['age_high'] = pd.to_numeric(age_split[1])
        
        # Import and merge population data
        pop_file = GBD_DATA_DIR / "GBD_population.csv"
        assert pop_file.exists(), f"GBD population file {pop_file} not found"
        pop_df = pd.read_csv(pop_file)
        gbd_merged = pd.merge(gbd_full, pop_df, on=['location_name', 'year', 'age_name'], how='left')
        
        # Process mortality data
        mort_df = gbd_merged[gbd_merged['measure_name'] == 'Deaths'].copy()
        mort_df['val'] = mort_df['val'] / 100000  # Convert to individual-level probability
        
        # Get mortality causes (exclude "All causes")
        mort_causes = [col for col in mort_df['cause_name'].unique() if col != "All causes"]
        
        # Pivot mortality data to wide format
        mort_wide = mort_df.pivot_table(
            index=['location_name', 'age_name', 'age_low', 'age_high', 'year', 'population'],
            columns='cause_name',
            values='val',
            fill_value=0
        ).reset_index()
        
        # Calculate total mortality
        mort_wide['Total'] = mort_wide[mort_causes].sum(axis=1)
        
        # Remove "All causes" column if it exists
        if 'All causes' in mort_wide.columns:
            mort_wide = mort_wide.drop('All causes', axis=1)
        
        # Process health/disability data
        health_df = gbd_merged[gbd_merged['measure_name'] == 'YLD'].copy()
        health_df['val'] = health_df['val'] / 100000  # Convert to individual-level probability
        
        # Get health causes
        health_causes = [col for col in health_df['cause_name'].unique() if col != "All causes"]
        
        # Pivot health data to wide format
        health_wide = health_df.pivot_table(
            index=['location_name', 'age_name', 'age_low', 'age_high', 'year', 'population'],
            columns='cause_name',
            values='val',
            fill_value=0
        ).reset_index()
        
        # Calculate total disability
        health_wide['Total'] = health_wide[health_causes].sum(axis=1)

        # Remove "All causes" column if it exists
        if 'All causes' in health_wide.columns:
            health_wide = health_wide.drop('All causes', axis=1)
        
        # Export processed data
        mort_wide.to_csv(INTERMEDIATE_GBD_DIR / "mortality_rates.csv", index=False)
        health_wide.to_csv(INTERMEDIATE_GBD_DIR / "morbidity_rates.csv", index=False)
        
        print(f"Processed GBD data: {len(mort_wide)} mortality records, {len(health_wide)} health records")
        return {'mortality': mort_wide, 'health': health_wide, 'raw_gbd': gbd_merged}
    
    @staticmethod
    def _process_wpp_sheet(file_path: Path, sheet_name: str, 
                          value_col: str, id_vars: List[str],
                          band_col: str, skiprows: int = 16) -> pd.DataFrame:
        """
        Helper method to process WPP Excel sheets with common patterns.

        @TODO: Look into handling 100+ age group
        
        Args:
            file_path: Path to Excel file
            sheet_name: Name of sheet to read
            value_col: Name for the value column after melting
            id_vars: ID variables for melting
            band_col: Column name for band values (e.g., 'Age', 'Year')
            skiprows: Number of rows to skip when reading Excel
            
        Returns:
            Processed DataFrame in long format
        """
        df = pd.read_excel(file_path, sheet_name=sheet_name, skiprows=skiprows)
        df = df[df['Type'] == 'Country/Area'].copy()
        df['Country'] = df['Region, subregion, country or area *']

        if 'Reference date (as of 1 July)' in df.columns:
            assert 'Year' not in df.columns, "Year column already exists"
            df['Year'] = df['Reference date (as of 1 July)'].astype(int)
        
        # Handle 100+ age group for population data
        if '100+' in df.columns and 'Age' in band_col:
            df['100-104'] = df['100+']
        
        # Select band columns (age or year)
        if 'Age' in band_col:
            band_pattern = r'^\d+-\d+$' # e.g. 0-4, 5-9, 10-14, etc.
        else:  # Year columns
            band_pattern = r'^\d{4}-\d{4}$'
        
        band_cols = [col for col in df.columns if match(band_pattern, col)]
        band_cols = ['Country', 'Variant'] + band_cols
        if 'Year' in df.columns:
            band_cols = ['Year'] + band_cols
        
        df = df[band_cols]
        
        # Convert to long format
        long_df = pd.melt(
            df,
            id_vars=id_vars,
            var_name=band_col,
            value_name=value_col
        )
        
        # Process band variables (from bands to single values)
        band_split = long_df[band_col].str.split('-', expand=True)
        long_df[f'{band_col}_low'] = pd.to_numeric(band_split[0])
        long_df[f'{band_col}_high'] = pd.to_numeric(band_split[1])
        long_df[f'{band_col}_mid'] = (long_df[f'{band_col}_low'] + long_df[f'{band_col}_high']) / 2
        
        return long_df

    def process_wpp_data(self) -> Dict[str, pd.DataFrame]:
        """
        Process UN World Population Prospects data using specified filepaths.

        Specifically processes the following files in the WPP directory:
            "UN_WPP2019_popbyage.xlsx"
            "UN_WPP2019_lifeexpectancy.xlsx"
            "UN_WPP2019_fertility.xlsx"

        Writes the following files to the intermediate WPP directory:
            "population.csv" (combined estimates and projections)
            "life_expectancy.csv" (combined estimates and projections)
            "fertility.csv" (combined estimates and projections)

        Returns:
            Dict[str, pd.DataFrame]: Processed WPP datasets
        """
        # Assert input data directory exists
        assert WPP_DATA_DIR.exists(), f"WPP directory {WPP_DATA_DIR} not found"
        print(f"Processing WPP data from {WPP_DATA_DIR}...")
        
        # Initialize processed data
        processed_data = {}
        
        ### Process population data (age-based) ###
        pop_file = WPP_DATA_DIR / 'UN_WPP2019_popbyage.xlsx'
        assert pop_file.exists(), f"Population estimates file {pop_file} not found"
        
        pop_estimates = pd.concat([
            self._process_wpp_sheet( # Historical estimates
                pop_file, 'ESTIMATES', 'Population', 
                ['Country', 'Variant', 'Year'], 'Age'
            ),
            self._process_wpp_sheet( # Future projections
                pop_file, 'MEDIUM VARIANT', 'Population', 
                ['Country', 'Variant', 'Year'], 'Age'
            )
        ], ignore_index=True)
        
        pop_estimates.to_csv(INTERMEDIATE_WPP_DIR / "population.csv", index=False)
        processed_data['population'] = pop_estimates
        

        ### Process life expectancy data (year-based) ###
        le_file = WPP_DATA_DIR / 'UN_WPP2019_lifeexpectancy.xlsx'
        assert le_file.exists(), f"Life expectancy file {le_file} not found"
        
        le_estimates = pd.concat([
            self._process_wpp_sheet(    # Historical estimates
                le_file, 'ESTIMATES', 'Life_Expectancy',
                ['Country', 'Variant'], 'Year'
                ),
            self._process_wpp_sheet(    # Future projections
                le_file, 'MEDIUM VARIANT', 'Life_Expectancy',
                ['Country', 'Variant'], 'Year'
            )
        ], ignore_index=True)
        
        le_estimates.to_csv(INTERMEDIATE_WPP_DIR / "life_expectancy.csv", index=False)
        processed_data['life_expectancy'] = le_estimates
        

        ### Process fertility data (year-based) ###
        fert_file = WPP_DATA_DIR / 'UN_WPP2019_fertility.xlsx'
        assert fert_file.exists(), f"Fertility file {fert_file} not found"
        
        fert_estimates = pd.concat([
            self._process_wpp_sheet(    # Historical estimates
                fert_file, 'ESTIMATES', 'Fertility_Rate',
                ['Country', 'Variant'], 'Year'
            ),
            self._process_wpp_sheet(    # Future projections
                fert_file, 'MEDIUM VARIANT', 'Fertility_Rate',
                ['Country', 'Variant'], 'Year'
            )
        ], ignore_index=True)
        
        fert_estimates.to_csv(INTERMEDIATE_WPP_DIR / "fertility.csv", index=False)
        processed_data['fertility'] = fert_estimates
        
        print("Processed input WPP data...")
        return processed_data
    
    def process_gdp_data(self) -> pd.DataFrame:
        """
        Process World Bank GDP data using specified filepaths.

        Specifically processes the following files in the WB directory:
            "GDP_constant_USD.csv"
            "GDP_constant_PPP.csv"

        Writes the following file to the intermediate WB directory:
            "real_gdp_data.csv"
        """
        assert WB_DATA_DIR.exists(), f"WB directory {WB_DATA_DIR} not found"
        print(f"Processing GDP data from {WB_DATA_DIR}...")
        
        # Import GDP data files
        gdp_files = {
            'usd': 'GDP_constant_USD.csv',
            'ppp': 'GDP_constant_PPP.csv'
        }
        
        gdp_data = []
        for key, filename in gdp_files.items():
            file_path = WB_DATA_DIR / filename
            assert file_path.exists(), f"File {file_path} not found"
            df = pd.read_csv(file_path, skiprows=4)
            df['location_name'] = df['Country Name']
                
            # Remove unnecessary columns
            cols_to_drop = ['Country Name', 'Country Code', 'Indicator Name', 'Indicator Code']
            df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
            
            # Convert to long format
            year_cols = [col for col in df.columns if col.startswith('20') or col.startswith('19')]
            df_long = pd.melt(
                df,
                id_vars=['location_name'],
                value_vars=year_cols,
                var_name='year',
                value_name=f'real_gdp_{key}'
            )
            
            df_long['year'] = pd.to_numeric(df_long['year'])
            gdp_data.append(df_long)
        
        # Merge USD and PPP data
        gdp_merged = pd.merge(
            gdp_data[0], gdp_data[1], 
            on=['location_name', 'year'], 
            how='outer'
        )
        
        # Standardize country names
        country_mapping = {
            'United States': 'United States of America',
            'World': 'Global',
            'Czech Republic': 'Czechia',
            'Iran, Islamic Rep.': 'Iran (Islamic Republic of)'
        }
        
        for old_name, new_name in country_mapping.items():
            gdp_merged.loc[gdp_merged['location_name'] == old_name, 'location_name'] = new_name
        
        # Export
        gdp_merged.to_csv(INTERMEDIATE_WB_DIR / "real_gdp_data.csv", index=False)
        print(f"Processed GDP data: {len(gdp_merged)} records")
        return gdp_merged

    def process_consumption_data(self) -> pd.DataFrame:
        """
        Process US consumption data for pre-graduation period.
        
        Equivalent to Julia: USc = Float64.(XLSX.readdata("data/USCData2018.xlsx","Sheet1","B19:B145"))

        Specifically processes the following files in the data directory:
            "USCData2018.xlsx"

        Writes the following file to the intermediate WB directory:
            "consumption_USA.csv"
        
        Returns:
            numpy array of consumption values for ages 0-126 (only ages 0-20 are used)
        """
        print(f"Processing consumption data from {DATA_DIR}...")

        consumption_file = DATA_DIR / "USCData2018.xlsx"
        assert consumption_file.exists(), f"Consumption file {consumption_file} not found"
            
        # Read Excel file, Sheet1, age and consumption columns
        consumption_df = pd.read_excel(consumption_file, sheet_name="Sheet1", header=None, 
                            usecols="A,B", skiprows=17, nrows=128)

        # Give column names
        consumption_df.columns = ['age', 'consumption']
        
        # Convert to numpy array and ensure float64
        assert consumption_df['age'].dtype == 'int64', "Age column is not int64"
        assert consumption_df['consumption'].dtype == 'float64', "Consumption column is not float64"
        
        # Handle any NaN values (replace with 1.0)
        assert not consumption_df.isna().any().any(), "NaN values found in consumption data"
        
        # Export
        consumption_df.to_csv(INTERMEDIATE_DIR / "consumption_USA.csv", index=False)

        print(f"Processed US consumption data for {len(consumption_df)} ages")
        return consumption_df

    @staticmethod
    def ogive_interpolation(df: pd.DataFrame, col_name: str, 
                max_age: int, min_value: float = 0.0, 
                max_value: float = None) -> np.ndarray:
        """
        Vectorized ogive interpolation from bands to single-year ages 0 to max_age.

        Args:
            df: DataFrame with columns: age_low, age_high, col_name
            col_name: Name of column to interpolate
            max_age: Maximum age to interpolate to
            min_value: Minimum value to constrain to
            max_value: Maximum value to constrain to

        Returns: 
            np.ndarray of length max_age+1
        """
        # 1) Build unique band edges (include [0, max_age+1] to anchor interpolation)
        edges = np.array(sorted(
            set(df['age_low']).union(df['age_high'] + 1).union({0, max_age + 1})
        ), dtype=int) # Given our high max age, this looks artifically stretched

        # 2) Vectorize band params and map bands to edge indices
        a0 = df['age_low'].to_numpy()
        a1 = df['age_high'].to_numpy()

        data = df[col_name].to_numpy(dtype=float)

        # Handle YLD to health conversion
        if col_name == 'health_rate':
            data = 1 - data

        width = (a1 - a0 + 1).astype(float)              # band width in years
        slope = data / width                              # CDF slope inside the band
        i0 = np.searchsorted(edges, a0)                  # start edge index
        i1 = np.searchsorted(edges, a1 + 1)              # end edge index (exclusive)

        # 3) Difference-array accumulation of slopes across edge intervals
        slope_delta = np.zeros(edges.size, dtype=float)
        np.add.at(slope_delta, i0,  slope)               # +slope starting at i0
        np.add.at(slope_delta, i1, -slope)               # -slope after i1-1
        slopes = np.cumsum(slope_delta)[:-1]             # slope per interval [edge_i, edge_{i+1})

        # 4) Integrate slopes to get CDF at edges, then interpolate at integer boundaries
        cdf_edges = np.concatenate(([0.0], np.cumsum(slopes * np.diff(edges))))
        boundaries = np.arange(0, max_age + 2)           # 0,1,...,max_age+1
        cdf_b = np.interp(boundaries, edges, cdf_edges)

        # 5) Single-year counts are first differences of the CDF
        per_age = np.diff(cdf_b)
        return np.clip(per_age, min_value, max_value)

    def piecewise_constant_interpolation(self, df: pd.DataFrame, 
                max_age: int, 
                health_floor: float = 0.1,
                old_age: int = 60,
                min_trend_size: int = 20) -> Tuple[np.ndarray, np.ndarray]:
        """
        Constant interpolation from age bands to single-year ages 0 to max_age.
        
        Args:
            df: DataFrame with columns: age_low, age_high, mortality_rate, health_rate
            max_age: Maximum age to interpolate to
            health_floor: Minimum health rate to constrain to
            old_age: Age at which to start the trend
            min_trend_size: Minimum size of trend to use for extrapolation
        
        Returns:
            survival: Survival array
            health: Health array
        """
        # Create age array
        ages = np.arange(0, max_age + 1)
        max_observed_age = int(df['age_high'].max())

        # Extract band boundaries and rates as arrays
        age_lows = df['age_low'].values
        age_highs = df['age_high'].values
        mortality_rates = df['mortality_rate'].values
        yld_rates = df['health_rate'].values
        
        # Initialize output arrays
        mortality = np.ones(max_age + 1)        # Default to 1.0 (certain death)
        health = np.full(max_age + 1, np.nan)   # Mark all as needing assignment
        
        # Vectorized band assignment: for each age, find which band it belongs to
        for j in range(len(df)):
            # Create mask for ages in this band
            in_band = (ages >= age_lows[j]) & (ages <= age_highs[j])
            
            # Assign mortality rate for this band
            mortality[in_band] = mortality_rates[j]
            
            # Assign health rate (YLD to health conversion)
            health_rate = 1.0 - yld_rates[j]
            health[in_band] = np.maximum(health_rate, health_floor)
        
        # Compute survival using cumulative product: S(a) = ∏(1 - μ(t)) from t=0 to a-1
        # S[0] = 1, S[a] = S[a-1] * (1 - mortality[a-1])
        survival = np.cumprod(1 - np.concatenate([[0], mortality[:-1]]))
        survival[-1] = 0.0  # Ensure last period survival is zero
        
        # Second pass: extrapolate health for ages beyond observed data
        # Use linear trend based on decline in older ages 
        if max_observed_age < max_age:
            # Find reference ages for trend estimation (older ages, typically 60+)
            trend_start_age = min(old_age, max_observed_age - min_trend_size)
            
            # Vectorized: select ages in trend range with valid health values
            trend_mask = (ages >= trend_start_age) & (ages <= max_observed_age) & \
                            (~np.isnan(health)) & (health > health_floor)
            trend_ages = ages[trend_mask]
            trend_health_vals = health[trend_mask]
            
            if len(trend_ages) >= 2:
                # Fit linear trend: H(a) = H_0 + slope * a
                slope = np.polyfit(trend_ages, trend_health_vals, 1)[0]
                
                # Vectorized extrapolation for all ages beyond observed
                last_observed_health = health[max_observed_age]
                extrapolation_ages = ages[max_observed_age + 1:]
                age_diffs = extrapolation_ages - max_observed_age
                
                # Linear extrapolation with floor
                extrapolated = last_observed_health + slope * age_diffs
                health[max_observed_age + 1:] = np.maximum(extrapolated, health_floor)
            else:
                # Fallback: gradual linear decay to floor
                last_health = health[max_observed_age]
                decay_rate = (last_health - health_floor) / (max_age - max_observed_age)
                
                # Vectorized decay
                extrapolation_ages = ages[max_observed_age + 1:]
                age_diffs = extrapolation_ages - max_observed_age
                decayed = last_health - decay_rate * age_diffs
                health[max_observed_age + 1:] = np.maximum(decayed, health_floor)
        
        return survival, health

    def merge_health_and_gdp(self, max_age: int = 240) -> pd.DataFrame:
        """
        Merge health and GDP data and create country-year level DataFrame with interpolated data.
        
        This function efficiently processes the entire dataset using vectorized operations:
        1. Loads mortality, health, and GDP data
        2. Expands age-banded data to country-year-age level
        3. Interpolates to integer ages for each country-year
        4. Computes aggregate measures (life expectancy, healthy life expectancy, etc.)
        5. Returns a DataFrame with one row per country-year
        
        Args:
            max_age: Maximum age for interpolation (default 240)
            
        Returns:
            DataFrame with country-year level data including interpolated health measures
        """
        print("Merging mortality, health, and GDP data...")
        
        # Load mortality data
        mort_file = INTERMEDIATE_GBD_DIR / "mortality_rates.csv"
        assert mort_file.exists(), f"Mortality data file {mort_file} not found"
        mortality_df = pd.read_csv(mort_file)
        
        # Load health data  
        health_file = INTERMEDIATE_GBD_DIR / "morbidity_rates.csv"
        assert health_file.exists(), f"Health data file {health_file} not found"
        health_df = pd.read_csv(health_file)
        
        # Load GDP data
        gdp_file = INTERMEDIATE_WB_DIR / "real_gdp_data.csv"
        assert gdp_file.exists(), f"GDP data file {gdp_file} not found"
        gdp_df = pd.read_csv(gdp_file)
                
        # Merge mortality and health data on location_name, year, age_name
        merged_df = pd.merge(
            mortality_df[['location_name', 'year', 'age_low', 'age_high', 'population', 'Total']].rename(columns={'Total': 'mortality_rate'}),
            health_df[['location_name', 'year', 'age_low', 'age_high', 'population', 'Total']].rename(columns={'Total': 'health_rate'}),
            on=['location_name', 'year', 'age_low', 'age_high', 'population'],
            how='outer',
            validate='one_to_one'
        )
        assert len(merged_df) == len(mortality_df) == len(health_df)
        
        # Merge with GDP data
        merged_df = pd.merge(
            merged_df,
            gdp_df,
            on=['location_name', 'year'],
            how='left'
        )

        # Load WHO life expectancy and health expectancy data
        WHO_df = pd.read_csv(DATA_DIR / "WHO_HLE_data.csv")
        WHO_df['LE_WHO'] = WHO_df['Life expectancy at birth (years)']
        WHO_df['HLE_WHO'] = WHO_df['Healthy life expectancy (HALE) at birth (years)']
        WHO_df['year'] = WHO_df['Year']
        # Rename country to "United Kingdom" where necessary
        WHO_df['location_name'] = WHO_df['Country'].replace(
            "United Kingdom of Great Britain and Northern Ireland", "United Kingdom"
        )

        # Merge with WHO data
        merged_df = pd.merge(
            merged_df,
            WHO_df[['location_name', 'year', 'LE_WHO', 'HLE_WHO']],
            on=['location_name', 'year'],
            how='left'
        )

        # Load fertility data
        fert_df = pd.read_csv(INTERMEDIATE_WPP_DIR / "fertility.csv")
        fert_df['year_diff'] = fert_df['Year_high'] - fert_df['Year_low']
        assert len(fert_df['year_diff'].unique()) == 1, "Different age ranges in fertility data"
        
        fert_df['year'] = fert_df.apply(
            lambda row: list(range(int(row['Year_low']), int(row['Year_high']))), 
            axis=1
        )
    
        # Explode the year ranges
        fert_df = fert_df.explode('year').reset_index(drop=True)
    
        # Rename and calculate annual values
        fert_df['location_name'] = fert_df['Country']
        fert_df['births'] = fert_df['Fertility_Rate'] / fert_df['year_diff'].iloc[0] * 1000

        fert_df.to_csv(INTERMEDIATE_DIR / "fertility_expanded.csv", index=False)
    
        # Select final columns and merge
        merged_df = pd.merge(
            merged_df,
            fert_df[['location_name', 'year', 'births']],
            on=['location_name', 'year'],
            how='left'
        )
        
        
        def process_country_year(group) -> Tuple[Dict[str, float], pd.DataFrame]:
            """Process a single country-year group to compute interpolated measures to integer ages."""
            # Population is extensive: use ogive interpolation to distribute across ages
            population = self.ogive_interpolation(group, 'population', max_age)
            assert abs(np.sum(population) - group['population'].sum()) < 1

            # Health and mortality are intensive (rates): use piecewise *CONSTANT* interpolation
            survival, health = self.piecewise_constant_interpolation(group, max_age)
            
            # Create result dictionary of aggregates measures
            total_pop = np.sum(population)
            gdp_total = group['real_gdp_usd'].iloc[0] if not group['real_gdp_usd'].isna().all() else np.nan
            country_year_summary = {
                'country': group['location_name'].iloc[0],
                'year': group['year'].iloc[0],
                'total_population': total_pop,
                'real_gdp_usd': gdp_total,
                'real_gdp_pc': gdp_total / total_pop if total_pop > 0 and not np.isnan(gdp_total) else np.nan,
                'le': np.sum(survival),
                'hle': np.sum(survival * health),
                'le_WHO': group['LE_WHO'].iloc[0],
                'hle_WHO': group['HLE_WHO'].iloc[0],
                'births': group['births'].iloc[0],
                'avg_health': np.sum(health * population) / total_pop if total_pop > 0 else np.mean(health),
                'avg_survival': np.sum(survival * population) / total_pop if total_pop > 0 else np.mean(survival)
            }

            # Create age-specific DataFrame
            age_specific_data = pd.DataFrame({
                'country': group['location_name'].iloc[0],
                'year': group['year'].iloc[0],
                'real_gdp_usd': group['real_gdp_usd'].iloc[0],
                'age': np.arange(0, max_age + 1),
                'population': population,
                'survival': survival,
                'health': health
            })
            
            return country_year_summary, age_specific_data
        
        # Iterate over unique country-year combinations
        country_years = merged_df[['location_name', 'year']].drop_duplicates()
        print(f"Processing {len(country_years)} country-year combinations...")
        summaries, results_df = [], pd.DataFrame()
        for _, group in merged_df.groupby(['location_name', 'year']):
            if group['real_gdp_usd'].isna().all():
                print(f"Warning: No GDP data for {group['location_name'].iloc[0]} in {group['year'].iloc[0]}, skipping...")
                continue
            summ, res = process_country_year(group)
            summaries.append(summ)
            results_df = pd.concat([results_df, res])
        summaries_df = pd.DataFrame(summaries)
        
        print(f"Created country-year level dataset with {len(results_df)} observations")
        print(f"Countries: {results_df['country'].nunique()}")
        print(f"Years: {results_df['year'].min()} - {results_df['year'].max()}")

        # Check WHO and GBD data align
        full_le_df = summaries_df.dropna(subset=['le_WHO', 'le'])
        for col in ['le', 'hle']:
            pct_diff_mean = np.mean(np.abs(full_le_df[f'{col}_WHO'] - full_le_df[col]) / full_le_df[f'{col}_WHO']) * 100
            assert pct_diff_mean < 1, f"WHO and GBD {col} data have mean percent difference = {pct_diff_mean:.2f}% > 1%!"
            pct_diff_max = np.max(np.abs(full_le_df[f'{col}_WHO'] - full_le_df[col]) / full_le_df[f'{col}_WHO']) * 100
            assert pct_diff_max < 5, f"WHO and GBD {col} data have max percent difference = {pct_diff_max:.2f}% > 5%!"
            correlation = np.corrcoef(full_le_df[f'{col}_WHO'], full_le_df[col])[0, 1]
            assert correlation > 0.99, f"WHO and GBD {col} data have correlation = {correlation:.2f} < 0.99!"
            print(f"WHO and GBD data align: {pct_diff_mean:.2f}% mean difference, {pct_diff_max:.2f}% max difference, {correlation:.2f} correlation")
        
        # Save the results
        output_file = INTERMEDIATE_DIR / "merged.csv"
        results_df.to_csv(output_file, index=False)
        print(f"Saved results to {output_file}")

        # Save the summaries
        summaries_file = INTERMEDIATE_DIR / "merged_summaries.csv"
        summaries_df.to_csv(summaries_file, index=False)
        print(f"Saved summaries to {summaries_file}")
        
        return results_df

    def run_all_processing(self):
        """Run all data processing steps."""
        print("Starting comprehensive data processing...")
        processed_data = {}

        # Process GBD data
        gbd_results = self.process_gbd_data()
        processed_data.update(gbd_results)
        
        # Process WPP data
        wpp_results = self.process_wpp_data()
        processed_data.update(wpp_results)
        
        # Process GDP data
        gdp_results = self.process_gdp_data()
        processed_data['gdp'] = gdp_results

        # Process consumption data
        usc_results = self.process_consumption_data()

        # Merge and interpolate health and GDP data
        merged_results = self.merge_health_and_gdp()
        processed_data.update(merged_results)
        
        print("Data processing completed successfully!") 
        return processed_data


def main(plot: bool = True):
    """
    Main function to run all data processing
    
    Args:
        plot: Whether to plot the data
    """
    dp = DataProcessor()
    dp.run_all_processing()

    # Plot (US) mortality and health trends
    if plot:
        create_exploratory_plots()


if __name__ == "__main__":
    main()
