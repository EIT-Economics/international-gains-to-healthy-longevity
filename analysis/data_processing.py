#!/usr/bin/env python3
"""
This script processes raw data from:
1. Global Burden of Disease (GBD)
2. UN World Population Prospects (WPP)  
3. World Bank (WB) GDP data

transforming it into `intermediate` data that is used by the other scripts.
"""

import pandas as pd
import re
from pathlib import Path
from typing import Dict, List
import sys 

# Set up paths based on repository structure
BASE_DIR = Path(__file__).parent.parent # Repository root
DATA_DIR = BASE_DIR / "data"  # Raw input data
INTERMEDIATE_DIR = BASE_DIR / "intermediate"  # Processed data
INTERMEDIATE_DIR.mkdir(parents=True, exist_ok=True)

sys.path.append(str(BASE_DIR))
from visualization.plotting import create_exploratory_plots


class DataProcessor:
    """Main class for processing health and economic data."""
        
    def process_gbd_data(self, save_countries: bool = False, save_ages: bool = False) -> Dict[str, pd.DataFrame]:
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
            "countries.csv" if save_countries is True
            "ages.csv" if save_ages is True

        Args:
            save_countries: Whether to save countries for reference
            save_ages: Whether to save ages for reference

        Returns:
            Dict[str, pd.DataFrame]: Processed datasets
        """
        # Assert input data directory exists
        GBD_DIR = DATA_DIR / "GBD"
        assert GBD_DIR.exists(), f"GBD directory {GBD_DIR} not found"
        print(f"Processing GBD data from {GBD_DIR}...")
        
        # Create GBD output directory if it doesn't exist
        OUT_DIR = INTERMEDIATE_DIR / "GBD"
        OUT_DIR.mkdir(parents=True, exist_ok=True)

        # Import and combine GBD data files
        gbd_files = [
            "International_GBD_1990-2019_DATA_1.csv",
            "International_GBD_1990-2019_DATA_2.csv", 
            "International_GBD_1990-2019_DATA_3.csv"
        ]
        
        gbd_dfs = []
        for file in gbd_files:
            file_path = GBD_DIR / file
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
        gbd_full['age'] = (gbd_full['age_low'] + gbd_full['age_high']) / 2
        gbd_full['age'] = gbd_full['age'].round().astype(int)
        
        # Save countries for reference
        if save_countries:
            countries = gbd_full['location_name'].value_counts().reset_index()
            countries.columns = ['location_name', 'count']
            countries.to_csv(OUT_DIR / "countries.csv", index=False)
        
        # Save ages for reference
        if save_ages:  
            ages = gbd_full['age_name'].value_counts().reset_index()
            ages.columns = ['age_name', 'count']
            ages.to_csv(OUT_DIR / "ages.csv", index=False)
        
        # Import and merge population data
        pop_file = GBD_DIR / "GBD_population.csv"
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
            index=['location_name', 'age_name', 'age', 'year', 'population'],
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
            index=['location_name', 'age_name', 'age', 'year', 'population'],
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
        mort_wide.to_csv(OUT_DIR / "mortality_rates.csv", index=False)
        health_wide.to_csv(OUT_DIR / "morbidity_rates.csv", index=False)
        
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
        
        band_cols = [col for col in df.columns if re.match(band_pattern, col)]
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
        WPP_DIR = DATA_DIR / "WPP"
        assert WPP_DIR.exists(), f"WPP directory {WPP_DIR} not found"
        print(f"Processing WPP data from {WPP_DIR}...")

        # Create WPP output directory if it doesn't exist
        OUT_DIR = INTERMEDIATE_DIR / "WPP"
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        
        # Initialize processed data
        processed_data = {}
        
        ### Process population data (age-based) ###
        pop_file = WPP_DIR / 'UN_WPP2019_popbyage.xlsx'
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
        
        pop_estimates.to_csv(OUT_DIR / "population.csv", index=False)
        processed_data['population'] = pop_estimates
        

        ### Process life expectancy data (year-based) ###
        le_file = WPP_DIR / 'UN_WPP2019_lifeexpectancy.xlsx'
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
        
        le_estimates.to_csv(OUT_DIR / "life_expectancy.csv", index=False)
        processed_data['life_expectancy'] = le_estimates
        

        ### Process fertility data (year-based) ###
        fert_file = WPP_DIR / 'UN_WPP2019_fertility.xlsx'
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
        
        fert_estimates.to_csv(OUT_DIR / "fertility.csv", index=False)
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
        WB_DIR = DATA_DIR / "WB"
        assert WB_DIR.exists(), f"WB directory {WB_DIR} not found"
        print(f"Processing GDP data from {WB_DIR}...")

        # Create WB output directory if it doesn't exist
        OUT_DIR = INTERMEDIATE_DIR / "WB"
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        
        # Import GDP data files
        gdp_files = {
            'usd': 'GDP_constant_USD.csv',
            'ppp': 'GDP_constant_PPP.csv'
        }
        
        gdp_data = []
        for key, filename in gdp_files.items():
            file_path = WB_DIR / filename
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
        gdp_merged.to_csv(OUT_DIR / "real_gdp_data.csv", index=False)
        print(f"Processed GDP data: {len(gdp_merged)} records")
        return gdp_merged
    
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
        
        print("Data processing completed successfully!") 
        return processed_data
        

def main(plot: bool = True):
    """
    Main function to run all data processing
    
    Args:
        plot: Whether to plot the data
    """
    DataProcessor().run_all_processing()

    # Plot (US) mortality and health trends
    if plot:
        create_exploratory_plots()


if __name__ == "__main__":
    main()
