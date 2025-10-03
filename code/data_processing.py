#!/usr/bin/env python3
"""
Python equivalent of R data processing scripts for International Gains to Healthy Longevity
Emulates functionality from GBD_data.R, GBD_data_USdeepdive.R, and WPP_data.R

This script processes:
1. Global Burden of Disease (GBD) data
2. UN World Population Prospects (WPP) data  
3. World Bank GDP data
4. US-specific detailed analysis
"""

import pandas as pd
import numpy as np
import re
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict

# Set up paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"  # Raw input data
INTERMEDIATE_DIR = BASE_DIR / "intermediate"  # Processed data
FIGURES_DIR = BASE_DIR / "figures" # Generated plots and visualizations

# Create directories if they don't exist
for dir_path in [INTERMEDIATE_DIR, FIGURES_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

class DataProcessor:
    """Main class for processing health and economic data"""
    def __init__(self):
        self.gbd_data = None
        self.wpp_data = None
        self.gdp_data = None
        self.processed_data = {}
        
    def process_gbd_data(self) -> Dict[str, pd.DataFrame]:
        """
        Process Global Burden of Disease data (equivalent to former GBD_data.R)

        Specifically processes the following files in the GBD directory:
            "International_GBD_1990-2019_DATA_1.csv"
            "International_GBD_1990-2019_DATA_2.csv" 
            "International_GBD_1990-2019_DATA_3.csv"
            "GBD_population.csv"

        Writes the following files to the intermediate GBD directory:
            "mortality_data.csv"
            "health_data.csv"
            "countries.csv"
            "ages.csv"
        """
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
        
        # Create numerical age variables
        age_split = gbd_full['age_name'].str.split(' to ', expand=True)
        gbd_full['age_low'] = pd.to_numeric(age_split[0])
        gbd_full['age_high'] = pd.to_numeric(age_split[1])
        gbd_full['age'] = (gbd_full['age_low'] + gbd_full['age_high']) / 2
        gbd_full['age'] = gbd_full['age'].round().astype(int)
        
        # Save country and age lists
        countries = gbd_full['location_name'].value_counts().reset_index()
        countries.columns = ['location_name', 'count']
        countries.to_csv(OUT_DIR / "countries.csv", index=False)
        
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
        mort_df['val'] = mort_df['val'] / 100000  # Convert to probability
        
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
        health_df['val'] = health_df['val'] / 100000  # Convert to probability
        
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
        mort_wide.to_csv(OUT_DIR / "mortality_data.csv", index=False)
        health_wide.to_csv(OUT_DIR / "health_data.csv", index=False)
        
        print(f"Processed GBD data: {len(mort_wide)} mortality records, {len(health_wide)} health records")
        
        return {
            'mortality': mort_wide,
            'health': health_wide,
            'raw_gbd': gbd_merged
        }
    
    def process_wpp_data(self) -> Dict[str, pd.DataFrame]:
        """
        Process UN World Population Prospects data (equivalent to former WPP_data.R)

        Specifically processes the following files in the WPP directory:
            "UN_WPP2019_popbyage.xlsx"
            "UN_WPP2019_lifeexpectancy.xlsx"
            "UN_WPP2019_fertility.xlsx"

        Writes the following files to the intermediate WPP directory:
            "estimates.csv"
            "projections.csv"
            "LE_estimates.csv"
            "LE_projections.csv"
            "fertility_estimates.csv"
            "fertility_projections.csv"
        """
        WPP_DIR = DATA_DIR / "WPP"
        assert WPP_DIR.exists(), f"WPP directory {WPP_DIR} not found"
        print(f"Processing WPP data from {WPP_DIR}...")

        # Create WPP output directory if it doesn't exist
        OUT_DIR = INTERMEDIATE_DIR / "WPP"
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        
        wpp_files = {
            'population_estimates': 'UN_WPP2019_popbyage.xlsx',
            'life_expectancy': 'UN_WPP2019_lifeexpectancy.xlsx',
            'fertility': 'UN_WPP2019_fertility.xlsx'
        }
        
        processed_data = {}
        
        # Process population data
        pop_file = WPP_DIR / wpp_files['population_estimates']
        assert pop_file.exists(), f"Population estimates file {pop_file} not found"
        # Historical estimates
        estimates_df = pd.read_excel(pop_file, sheet_name='ESTIMATES', skiprows=16)
        estimates_df = estimates_df[estimates_df['Type'] == 'Country/Area'].copy()
        estimates_df['Country'] = estimates_df['Region, subregion, country or area *']
        estimates_df['Year'] = estimates_df['Reference date (as of 1 July)']
        
        # Handle 100+ age group
        if '100+' in estimates_df.columns:
            estimates_df['100-104'] = estimates_df['100+']
        
        # Select age columns
        age_cols = [col for col in estimates_df.columns if re.match(r'^\d+-\d+$', col)]
        age_cols = ['Country', 'Variant', 'Year'] + age_cols
        
        estimates_df = estimates_df[age_cols]
        
        # Convert to long format
        estimates_long = pd.melt(
            estimates_df,
            id_vars=['Country', 'Variant', 'Year'],
            var_name='Age',
            value_name='Population'
        )
            
        # Process age variables
        age_split = estimates_long['Age'].str.split('-', expand=True)
        estimates_long['Age_low'] = pd.to_numeric(age_split[0])
        estimates_long['Age_high'] = pd.to_numeric(age_split[1])
        estimates_long['Age_mid'] = (estimates_long['Age_low'] + estimates_long['Age_high']) / 2
        
        estimates_long.to_csv(OUT_DIR / "estimates.csv", index=False)
        processed_data['population_estimates'] = estimates_long
        
        # Future projections
        proj_df = pd.read_excel(pop_file, sheet_name='MEDIUM VARIANT', skiprows=16)
        proj_df = proj_df[proj_df['Type'] == 'Country/Area'].copy()
        proj_df['Country'] = proj_df['Region, subregion, country or area *']
        proj_df['Year'] = proj_df['Reference date (as of 1 July)']
        
        if '100+' in proj_df.columns:
            proj_df['100-104'] = proj_df['100+']
        
        proj_df = proj_df[age_cols]
        
        # Convert to long format
        proj_long = pd.melt(
            proj_df,
            id_vars=['Country', 'Variant', 'Year'],
            var_name='Age',
            value_name='Population'
        )
            
        # Process age variables
        age_split = proj_long['Age'].str.split('-', expand=True)
        proj_long['Age_low'] = pd.to_numeric(age_split[0])
        proj_long['Age_high'] = pd.to_numeric(age_split[1])
        proj_long['Age_mid'] = (proj_long['Age_low'] + proj_long['Age_high']) / 2
        
        proj_long.to_csv(OUT_DIR / "projections.csv", index=False)
        processed_data['population_projections'] = proj_long
        
        # Process life expectancy data
        le_file = WPP_DIR / wpp_files['life_expectancy']
        assert le_file.exists(), f"Life expectancy file {le_file} not found"
        # Historical estimates
        le_est = pd.read_excel(le_file, sheet_name='ESTIMATES', skiprows=16)
        le_est = le_est[le_est['Type'] == 'Country/Area'].copy()
        le_est['Country'] = le_est['Region, subregion, country or area *']
        
        # Select year columns
        year_cols = [col for col in le_est.columns if re.match(r'^\d{4}-\d{4}$', col)]
        year_cols = ['Country', 'Variant'] + year_cols
        le_est = le_est[year_cols]
        
        # Convert to long format
        le_est_long = pd.melt(
            le_est,
            id_vars=['Country', 'Variant'],
            var_name='Year',
            value_name='Life_Expectancy'
        )
            
        # Process year variables
        year_split = le_est_long['Year'].str.split('-', expand=True)
        le_est_long['Year_low'] = pd.to_numeric(year_split[0])
        le_est_long['Year_high'] = pd.to_numeric(year_split[1])
        le_est_long['Year_mid'] = (le_est_long['Year_low'] + le_est_long['Year_high']) / 2
        
        le_est_long.to_csv(OUT_DIR / "LE_estimates.csv", index=False)
        processed_data['le_estimates'] = le_est_long
        
        # Future projections
        le_proj = pd.read_excel(le_file, sheet_name='MEDIUM VARIANT', skiprows=16)
        le_proj = le_proj[le_proj['Type'] == 'Country/Area'].copy()
        le_proj['Country'] = le_proj['Region, subregion, country or area *']
        
        # Select future year columns
        future_year_cols = [col for col in le_proj.columns if re.match(r'^\d{4}-\d{4}$', col)]
        future_year_cols = ['Country', 'Variant'] + future_year_cols
        le_proj = le_proj[future_year_cols]
        
        # Convert to long format
        le_proj_long = pd.melt(
            le_proj,
            id_vars=['Country', 'Variant'],
            var_name='Year',
            value_name='Life_Expectancy'
        )
        
        # Process year variables
        year_split = le_proj_long['Year'].str.split('-', expand=True)
        le_proj_long['Year_low'] = pd.to_numeric(year_split[0])
        le_proj_long['Year_high'] = pd.to_numeric(year_split[1])
        le_proj_long['Year_mid'] = (le_proj_long['Year_low'] + le_proj_long['Year_high']) / 2
        
        le_proj_long.to_csv(OUT_DIR / "LE_projections.csv", index=False)
        processed_data['le_projections'] = le_proj_long
        
        # Process fertility data
        fert_file = WPP_DIR / wpp_files['fertility']
        assert fert_file.exists(), f"Fertility file {fert_file} not found"
        # Historical estimates
        fert_est = pd.read_excel(fert_file, sheet_name='ESTIMATES', skiprows=16)
        fert_est = fert_est[fert_est['Type'] == 'Country/Area'].copy()
        fert_est['Country'] = fert_est['Region, subregion, country or area *']
        
        # Select year columns
        year_cols = [col for col in fert_est.columns if re.match(r'^\d{4}-\d{4}$', col)]
        year_cols = ['Country', 'Variant'] + year_cols
        fert_est = fert_est[year_cols]
        
        # Convert to long format
        fert_est_long = pd.melt(
            fert_est,
            id_vars=['Country', 'Variant'],
            var_name='Year',
            value_name='Fertility_Rate'
        )
            
        # Process year variables
        year_split = fert_est_long['Year'].str.split('-', expand=True)
        fert_est_long['Year_low'] = pd.to_numeric(year_split[0])
        fert_est_long['Year_high'] = pd.to_numeric(year_split[1])
        fert_est_long['Year_mid'] = (fert_est_long['Year_low'] + fert_est_long['Year_high']) / 2
        
        fert_est_long.to_csv(OUT_DIR / "fertility_estimates.csv", index=False)
        processed_data['fertility_estimates'] = fert_est_long
        
        # Future projections
        fert_proj = pd.read_excel(fert_file, sheet_name='MEDIUM VARIANT', skiprows=16)
        fert_proj = fert_proj[fert_proj['Type'] == 'Country/Area'].copy()
        fert_proj['Country'] = fert_proj['Region, subregion, country or area *']
        
        # Select future year columns
        future_year_cols = [col for col in fert_proj.columns if re.match(r'^\d{4}-\d{4}$', col)]
        future_year_cols = ['Country', 'Variant'] + future_year_cols
        fert_proj = fert_proj[future_year_cols]
        
        # Convert to long format
        fert_proj_long = pd.melt(
            fert_proj,
            id_vars=['Country', 'Variant'],
            var_name='Year',
            value_name='Fertility_Rate'
        )
            
        # Process year variables
        year_split = fert_proj_long['Year'].str.split('-', expand=True)
        fert_proj_long['Year_low'] = pd.to_numeric(year_split[0])
        fert_proj_long['Year_high'] = pd.to_numeric(year_split[1])
        fert_proj_long['Year_mid'] = (fert_proj_long['Year_low'] + fert_proj_long['Year_high']) / 2
        
        fert_proj_long.to_csv(OUT_DIR / "fertility_projections.csv", index=False)
        processed_data['fertility_projections'] = fert_proj_long
        
        print(f"Processed WPP data: {len(processed_data)} datasets")
        return processed_data
    
    def process_gdp_data(self) -> pd.DataFrame:
        """
        Process World Bank GDP data (equivalent to GDP processing in former GBD_data.R)

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
        
        gdp_data = {}
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
            df_long = df_long[df_long['year'] >= 1990]
            df_long = df_long[df_long['year'] < 2020]
            
            gdp_data[key] = df_long
        
        # Merge USD and PPP data
        if 'usd' in gdp_data and 'ppp' in gdp_data:
            gdp_merged = pd.merge(
                gdp_data['usd'], 
                gdp_data['ppp'], 
                on=['location_name', 'year'], 
                how='outer'
            )
        elif 'usd' in gdp_data:
            gdp_merged = gdp_data['usd']
        elif 'ppp' in gdp_data:
            gdp_merged = gdp_data['ppp']
        else:
            print(f"No GDP data files found in {WB_DIR}")
            return pd.DataFrame()
        
        # Standardize country names
        country_mapping = {
            'United States': 'United States of America',
            'World': 'Global',
            'Czech Republic': 'Czechia',
            'Iran, Islamic Rep.': 'Iran (Islamic Republic of)'
        }
        
        for old_name, new_name in country_mapping.items():
            gdp_merged.loc[gdp_merged['location_name'] == old_name, 'location_name'] = new_name
        
        # Add population data if available
        gdp_merged['population'] = np.nan
        gdp_merged['real_gdp_usd'] = np.nan
        
        # Export
        gdp_merged.to_csv(OUT_DIR / "real_gdp_data.csv", index=False)
        print(f"Processed GDP data: {len(gdp_merged)} records")
        
        return gdp_merged
    
    def process_us_detailed_analysis(self) -> Dict[str, pd.DataFrame]:
        """
        Process US-specific detailed analysis (equivalent to former GBD_data_USdeepdive.R)

        Specifically processes the following file in the GBD directory:
            "US_GBD_1990-2019_DATA.csv"

        Writes the following files to the intermediate GBD directory:
            "GBD_mortality_data_US.csv"
            "GBD_health_data_US.csv"
        """
        GBD_DIR = DATA_DIR / "GBD"
        assert GBD_DIR.exists(), f"GBD directory {GBD_DIR} not found"
        print("Processing US detailed analysis...")
        
        # Create GBD output directory if it doesn't exist
        OUT_DIR = INTERMEDIATE_DIR / "GBD"
        OUT_DIR.mkdir(parents=True, exist_ok=True)

        # Load US-specific GBD data
        us_file = GBD_DIR / "US_GBD_1990-2019_DATA.csv"
        assert us_file.exists(), f"US GBD data file {us_file} not found"
        
        us_df = pd.read_csv(us_file)
        
        # Select relevant columns
        columns_to_keep = [
            'measure_id', 'measure_name', 'age_id', 'age_name', 'cause_id',
            'cause_name', 'metric_name', 'year', 'val'
        ]
        us_df = us_df[columns_to_keep]
        
        # Clean age variable
        us_df.loc[us_df['age_name'] == "<1 year", 'age_name'] = "0 to 0"
        us_df.loc[us_df['age_name'] == "95 plus", 'age_name'] = "95 to 99"
        
        # Shorten measure names
        measure_mapping = {
            'YLLs (Years of Life Lost)': 'YLL',
            'DALYs (Disability-Adjusted Life Years)': 'DALY',
            'YLDs (Years Lived with Disability)': 'YLD'
        }
        us_df['measure_name'] = us_df['measure_name'].replace(measure_mapping)
        
        # Create numerical age variables
        age_split = us_df['age_name'].str.split(' to ', expand=True)
        us_df['age_low'] = pd.to_numeric(age_split[0])
        us_df['age_high'] = pd.to_numeric(age_split[1])
        us_df['age_mid'] = (us_df['age_low'] + us_df['age_high']) / 2
        
        # Process mortality data
        mort_df = us_df[us_df['measure_name'] == 'Deaths'].copy()
        mort_df = mort_df[mort_df['metric_name'] == 'Rate'].copy()
        mort_df['val'] = mort_df['val'] / 100000  # Convert to probability
        
        mort_causes = [col for col in mort_df['cause_name'].unique() if col != "All causes"]
        
        # Pivot to wide format
        mort_wide = mort_df.pivot_table(
            index=['age_name', 'age_mid', 'year'],
            columns='cause_name',
            values='val',
            fill_value=0
        ).reset_index()
        
        mort_wide['Total'] = mort_wide[mort_causes].sum(axis=1)
        
        # Process health data
        health_df = us_df[us_df['measure_name'] == 'YLD'].copy()
        health_df = health_df[health_df['metric_name'] == 'Rate'].copy()
        health_df['val'] = health_df['val'] / 100000  # Convert to probability
        
        health_causes = [col for col in health_df['cause_name'].unique() if col != "All causes"]
        
        # Pivot to wide format
        health_wide = health_df.pivot_table(
            index=['age_name', 'age_mid', 'year'],
            columns='cause_name',
            values='val',
            fill_value=0
        ).reset_index()
        
        health_wide['Total'] = health_wide[health_causes].sum(axis=1)
        
        # Export
        mort_wide.to_csv(OUT_DIR / "mortality_data_US.csv", index=False)
        health_wide.to_csv(OUT_DIR / "health_data_US.csv", index=False)
        
        print(f"Processed US detailed analysis: {len(mort_wide)} mortality records, {len(health_wide)} health records")
        
        return {
            'mortality': mort_wide,
            'health': health_wide
        }
    
    def create_exploratory_plots(self, data_dict: Dict[str, pd.DataFrame]):
        """
        Create exploratory plots.

        Saves the following outputs to the figures directory:
            "GBD_mortality_trends.pdf"
            "GBD_health_trends.pdf"
        """
        print("Creating exploratory plots...")
        
        # Set up plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Plot mortality trends if data available
        if 'mortality' in data_dict:
            mort_df = data_dict['mortality']
            
            # Plot mortality by age for different years
            if 'Total' in mort_df.columns:
                plt.figure(figsize=(12, 8))
                for year in mort_df['year'].unique()[:5]:  # Plot first 5 years
                    year_data = mort_df[mort_df['year'] == year]
                    plt.plot(year_data['age_mid'], year_data['Total'], 
                            label=f'Year {year}', alpha=0.7)
                
                plt.xlabel('Age')
                plt.ylabel('Mortality Rate')
                plt.title('Mortality Rates by Age and Year')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(FIGURES_DIR / "mortality_trends.pdf", dpi=300, bbox_inches='tight')
                plt.close()
        
        # Plot health trends if data available
        if 'health' in data_dict:
            health_df = data_dict['health']
            
            # Plot health by age for different years
            if 'Total' in health_df.columns:
                plt.figure(figsize=(12, 8))
                for year in health_df['year'].unique()[:5]:  # Plot first 5 years
                    year_data = health_df[health_df['year'] == year]
                    plt.plot(year_data['age_mid'], year_data['Total'], 
                            label=f'Year {year}', alpha=0.7)
                
                plt.xlabel('Age')
                plt.ylabel('Disability Rate (YLD)')
                plt.title('Disability Rates by Age and Year')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(FIGURES_DIR / "health_trends.pdf", dpi=300, bbox_inches='tight')
                plt.close()
        
        print(f"Exploratory plots saved to {FIGURES_DIR}")
    
    def run_all_processing(self):
        """Run all data processing steps."""
        print("Starting comprehensive data processing...")
        
        # Process GBD data
        gbd_results = self.process_gbd_data()
        self.processed_data.update(gbd_results)
        
        # Process WPP data
        wpp_results = self.process_wpp_data()
        self.processed_data.update(wpp_results)
        
        # Process GDP data
        gdp_results = self.process_gdp_data()
        if not gdp_results.empty:
            self.processed_data['gdp'] = gdp_results
        else:
            print("Warning: No GDP data found")
        
        # Process US detailed analysis
        us_results = self.process_us_detailed_analysis()
        self.processed_data.update(us_results)
        
        # Create exploratory plots
        self.create_exploratory_plots(self.processed_data)
        
        print("Data processing completed successfully!")        
        return self.processed_data

def main(verbose: bool = False):
    """
    Main function to run all data processing
    """
    processor = DataProcessor()
    results = processor.run_all_processing()
    
    # Print summary
    if verbose:
        print("\n" + "="*50)
        print("PROCESSING SUMMARY")
        print("="*50)
        for key, data in results.items():
            if isinstance(data, pd.DataFrame):
                print(f"{key}: {len(data)} records, {len(data.columns)} columns")
            else:
                print(f"{key}: {type(data).__name__}")
        
        print("\nAll data processing completed successfully!")

if __name__ == "__main__":
    main()
