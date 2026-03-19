import pandas as pd
import numpy as np
from datetime import datetime
import json
import os

class AgroNovaModel:
    def __init__(self, data_dir='./data/maharashtra/'):
        self.data_dir = data_dir
        self.load_data()
        self.initialize_models()
    
    def load_data(self):
        """Load Maharashtra data files"""
        try:
            # Load district data
            self.rainfall_df = pd.read_csv(f'{self.data_dir}/rainfall_data.csv')
            self.soil_texture_df = pd.read_csv(f'{self.data_dir}/soil_texture_data.csv')
            self.nutrients_df = pd.read_csv(f'{self.data_dir}/soil_nutrients_data.csv')
            self.temperature_df = pd.read_csv(f'{self.data_dir}/temperature_data.csv')
            
            # Load crop mappings
            self.crop_calendar = pd.read_csv(f'{self.data_dir}/crop_calendar_mh.csv')
            self.intercrop_matrix = pd.read_csv(f'{self.data_dir}/intercrop_matrix_mh.csv')
            self.soil_degradation = pd.read_csv(f'{self.data_dir}/soil_degradation_thresholds.csv')
            
            print("✅ AgroNova data loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading AgroNova data: {e}")
    
    def initialize_models(self):
        """Initialize prediction models"""
        self.month_to_season = {
            'Jan': 'Rabi', 'Feb': 'Rabi', 'Mar': 'Summer',
            'Apr': 'Summer', 'May': 'Summer', 'Jun': 'Kharif',
            'Jul': 'Kharif', 'Aug': 'Kharif', 'Sep': 'Kharif',
            'Oct': 'Rabi', 'Nov': 'Rabi', 'Dec': 'Rabi'
        }
        
        self.crop_seasons = {
            'Kharif': ['Jun', 'Jul', 'Aug', 'Sep', 'Oct'],
            'Rabi': ['Nov', 'Dec', 'Jan', 'Feb', 'Mar'],
            'Summer': ['Mar', 'Apr', 'May', 'Jun']
        }
    
    def get_location_data(self, district, taluka):
        """Get all data for a specific location"""
        data = {
            'district': district,
            'taluka': taluka,
            'rainfall': self.get_rainfall(district, taluka),
            'temperature': self.get_temperature(district),
            'soil_texture': self.get_soil_texture(district, taluka),
            'nutrients': self.get_soil_nutrients(district, taluka)
        }
        return data
    
    def get_rainfall(self, district, taluka):
        """Get rainfall data for district-taluka"""
        try:
            row = self.rainfall_df[
                (self.rainfall_df['District'] == district) & 
                (self.rainfall_df['Taluka'] == taluka)
            ]
            if not row.empty:
                return row.iloc[0].to_dict()
        except:
            pass
        return {}
    
    def get_temperature(self, district):
        """Get temperature data for district"""
        try:
            row = self.temperature_df[self.temperature_df['District'] == district]
            if not row.empty:
                return row.iloc[0].to_dict()
        except:
            pass
        return {}
    
    def get_soil_texture(self, district, taluka):
        """Get soil texture data"""
        try:
            row = self.soil_texture_df[
                (self.soil_texture_df['District'] == district) & 
                (self.soil_texture_df['Taluka'] == taluka)
            ]
            if not row.empty:
                return row.iloc[0].to_dict()
        except:
            pass
        return {}
    
    def get_soil_nutrients(self, district, taluka):
        """Get soil nutrient data"""
        try:
            row = self.nutrients_df[
                (self.nutrients_df['District'] == district) & 
                (self.nutrients_df['Taluka'] == taluka)
            ]
            if not row.empty:
                return row.iloc[0].to_dict()
        except:
            pass
        return {}
    
    def predict_soil_degradation(self, current_nutrients, crop_history):
        """Predict soil degradation based on crop history"""
        degradation_score = 0
        issues = []
        
        # Calculate nutrient depletion
        n_depletion = 0
        p_depletion = 0
        k_depletion = 0
        
        for crop in crop_history:
            crop_data = self.get_crop_nutrient_extraction(crop['name'])
            n_depletion += crop_data.get('N_extraction', 0)
            p_depletion += crop_data.get('P_extraction', 0)
            k_depletion += crop_data.get('K_extraction', 0)
        
        # Predict future nutrient levels
        predicted_n = current_nutrients.get('N (kg/ha)', 0) - n_depletion
        predicted_p = current_nutrients.get('P (kg/ha)', 0) - p_depletion
        predicted_k = current_nutrients.get('K (kg/ha)', 0) - k_depletion
        
        # Check against thresholds
        if predicted_n < 200:
            degradation_score += 0.3
            issues.append(f"Nitrogen will drop to {predicted_n:.1f} kg/ha (below optimal 250-350)")
        
        if predicted_p < 15:
            degradation_score += 0.25
            issues.append(f"Phosphorus will drop to {predicted_p:.1f} kg/ha (below optimal 15-25)")
        
        if predicted_k < 150:
            degradation_score += 0.2
            issues.append(f"Potassium will drop to {predicted_k:.1f} kg/ha (below optimal 200-300)")
        
        # Check pH
        current_ph = current_nutrients.get('pH', 7.0)
        if current_ph < 5.5 or current_ph > 8.5:
            degradation_score += 0.25
            issues.append(f"pH is {current_ph} (optimal is 6.0-7.5)")
        
        return {
            'degradation_score': min(1.0, degradation_score),
            'degradation_level': self.get_degradation_level(degradation_score),
            'predicted_nutrients': {
                'N': predicted_n,
                'P': predicted_p,
                'K': predicted_k
            },
            'issues': issues[:5],
            'recommendations': self.get_degradation_recommendations(degradation_score, predicted_n, predicted_p, predicted_k)
        }
    
    def get_crop_nutrient_extraction(self, crop_name):
        """Get nutrient extraction for a crop"""
        # Simplified extraction values (kg/ha)
        extraction_map = {
            'Soybean': {'N_extraction': 50, 'P_extraction': 30, 'K_extraction': 40},
            'Cotton': {'N_extraction': 120, 'P_extraction': 50, 'K_extraction': 80},
            'Sorghum': {'N_extraction': 100, 'P_extraction': 50, 'K_extraction': 40},
            'Maize': {'N_extraction': 120, 'P_extraction': 60, 'K_extraction': 60},
            'Wheat': {'N_extraction': 100, 'P_extraction': 50, 'K_extraction': 40},
            'Chickpea': {'N_extraction': 20, 'P_extraction': 40, 'K_extraction': 30},
            'Groundnut': {'N_extraction': 40, 'P_extraction': 60, 'K_extraction': 40},
            'Onion': {'N_extraction': 60, 'P_extraction': 30, 'K_extraction': 80},
            'Tomato': {'N_extraction': 100, 'P_extraction': 50, 'K_extraction': 150},
            'Potato': {'N_extraction': 100, 'P_extraction': 40, 'K_extraction': 150}
        }
        return extraction_map.get(crop_name, {'N_extraction': 50, 'P_extraction': 30, 'K_extraction': 40})
    
    def get_degradation_level(self, score):
        """Get degradation level from score"""
        if score < 0.3:
            return 'Normal'
        elif score < 0.5:
            return 'Warning'
        elif score < 0.7:
            return 'Critical'
        else:
            return 'Severe'
    
    def get_degradation_recommendations(self, score, n, p, k):
        """Get recommendations based on degradation"""
        recs = []
        
        if n < 200:
            recs.append("Add legume intercrop to fix nitrogen naturally")
            recs.append("Apply 50-100 kg/ha urea before next sowing")
        
        if p < 15:
            recs.append("Apply DAP fertilizer (50-75 kg/ha)")
            recs.append("Incorporate phosphate-solubilizing bacteria")
        
        if k < 150:
            recs.append("Apply MOP fertilizer (40-60 kg/ha)")
            recs.append("Add organic matter to improve potassium retention")
        
        if score > 0.5:
            recs.append("Consider crop rotation with legumes")
            recs.append("Add green manure cover crop")
        
        return recs[:3]
    
    def recommend_intercrops(self, location_data, current_crops, season):
        """Recommend intercrops for the location and season"""
        recommendations = []
        
        # Get suitable crops for the season
        suitable_crops = self.get_seasonal_crops(season, location_data)
        
        for crop in suitable_crops:
            # Find compatible intercrops
            intercrops = self.find_compatible_intercrops(crop, location_data)
            
            for intercrop in intercrops:
                score = self.calculate_intercrop_score(crop, intercrop, location_data)
                profit = self.calculate_profit(crop, intercrop, location_data)
                
                if score > 7.0:
                    recommendations.append({
                        'primary_crop': crop,
                        'intercrop': intercrop,
                        'score': score,
                        'profit_increase': profit,
                        'planting_pattern': self.get_planting_pattern(crop, intercrop),
                        'benefits': self.get_intercrop_benefits(crop, intercrop)
                    })
        
        # Sort by score and profit
        recommendations.sort(key=lambda x: (x['score'], x['profit_increase']), reverse=True)
        return recommendations[:5]
    
    def get_seasonal_crops(self, season, location_data):
        """Get crops suitable for the season and location"""
        rainfall = location_data['rainfall']
        soil_texture = location_data['soil_texture']
        
        # Define crop suitability
        crop_suitability = {
            'Kharif': ['Soybean', 'Cotton', 'Maize', 'Groundnut', 'Sorghum', 'Pigeon Pea'],
            'Rabi': ['Wheat', 'Chickpea', 'Onion', 'Tomato', 'Potato', 'Mustard'],
            'Summer': ['Vegetables', 'Pulses', 'Oilseeds']
        }
        
        # Filter based on rainfall
        annual_rainfall = sum([v for k, v in rainfall.items() if k not in ['District', 'Taluka']])
        
        if annual_rainfall < 600:
            # Low rainfall areas
            suitable = ['Sorghum', 'Pearl Millet', 'Groundnut', 'Chickpea']
        elif annual_rainfall > 1000:
            # High rainfall areas
            suitable = ['Rice', 'Sugarcane', 'Cotton', 'Soybean']
        else:
            # Medium rainfall
            suitable = crop_suitability.get(season, [])
        
        return suitable[:5]
    
    def find_compatible_intercrops(self, primary_crop, location_data):
        """Find compatible intercrops for primary crop"""
        # This should come from your intercrop_matrix.csv
        compatibility_map = {
            'Cotton': ['Green Gram', 'Soybean', 'Cowpea', 'Coriander'],
            'Soybean': ['Pigeon Pea', 'Sorghum', 'Maize'],
            'Maize': ['Soybean', 'Black Gram', 'Potato'],
            'Wheat': ['Mustard', 'Chickpea', 'Linseed'],
            'Chickpea': ['Linseed', 'Mustard', 'Safflower'],
            'Groundnut': ['Pearl Millet', 'Sorghum'],
            'Onion': ['Radish', 'Carrot', 'Spinach'],
            'Tomato': ['Marigold', 'Basil', 'Onion']
        }
        return compatibility_map.get(primary_crop, [])
    
    def calculate_intercrop_score(self, crop1, crop2, location_data):
        """Calculate compatibility score (0-10)"""
        score = 6.0  # Base score
        
        # Nutrient complementarity
        if self.are_nutrients_complementary(crop1, crop2):
            score += 1.5
        
        # Root depth complementarity
        if self.are_roots_complementary(crop1, crop2):
            score += 1.5
        
        # Pest/disease protection
        if self.provides_pest_protection(crop1, crop2):
            score += 1.0
        
        # Water use compatibility
        water_score = self.check_water_compatibility(crop1, crop2, location_data['rainfall'])
        score += water_score
        
        return min(10.0, score)
    
    def are_nutrients_complementary(self, crop1, crop2):
        """Check if crops have complementary nutrient needs"""
        legume_crops = ['Soybean', 'Chickpea', 'Green Gram', 'Black Gram', 'Cowpea', 'Pigeon Pea']
        
        if crop1 in legume_crops and crop2 not in legume_crops:
            return True
        if crop2 in legume_crops and crop1 not in legume_crops:
            return True
        return False
    
    def are_roots_complementary(self, crop1, crop2):
        """Check if crops have complementary root systems"""
        # Deep rooted crops
        deep_rooted = ['Sorghum', 'Cotton', 'Sunflower', 'Pigeon Pea', 'Pearl Millet']
        # Shallow rooted crops
        shallow_rooted = ['Onion', 'Potato', 'Groundnut', 'Vegetables', 'Pulses']
        
        # Check if one is deep and other is shallow
        if (crop1 in deep_rooted and crop2 in shallow_rooted) or \
           (crop2 in deep_rooted and crop1 in shallow_rooted):
            return True
        return False
    
    def provides_pest_protection(self, crop1, crop2):
        """Check if one crop provides pest protection to another"""
        # Pest repellent crops
        pest_repellent = ['Marigold', 'Basil', 'Onion', 'Garlic', 'Mint', 'Coriander']
        
        # Crops that need protection
        needs_protection = ['Tomato', 'Cabbage', 'Cauliflower', 'Cotton', 'Potato']
        
        if (crop1 in pest_repellent and crop2 in needs_protection) or \
           (crop2 in pest_repellent and crop1 in needs_protection):
            return True
        return False
    
    def check_water_compatibility(self, crop1, crop2, rainfall_data):
        """Check if crops are compatible in water use"""
        # Water requirement categories
        high_water = ['Rice', 'Sugarcane', 'Banana', 'Tomato', 'Potato']
        medium_water = ['Cotton', 'Maize', 'Wheat', 'Soybean']
        low_water = ['Sorghum', 'Pearl Millet', 'Chickpea', 'Groundnut']
        
        # Calculate annual rainfall
        if isinstance(rainfall_data, dict):
            annual_rainfall = sum([v for k, v in rainfall_data.items() 
                                  if k not in ['District', 'Taluka'] and isinstance(v, (int, float))])
        else:
            annual_rainfall = 700  # Default
        
        # For low rainfall areas, avoid two high water crops
        if annual_rainfall < 600:
            if crop1 in high_water and crop2 in high_water:
                return -2.0  # Penalty
            elif (crop1 in low_water and crop2 in low_water) or \
                 (crop1 in low_water and crop2 in medium_water) or \
                 (crop1 in medium_water and crop2 in low_water):
                return 1.0  # Bonus
        
        return 0.0  # Neutral
    
    def get_planting_pattern(self, primary_crop, intercrop):
        """Get planting pattern for intercrop system"""
        patterns = {
            'Cotton+Green Gram': '6:2 border cropping',
            'Cotton+Soybean': '4:2 paired rows',
            'Soybean+Pigeon Pea': '4:1 strip cropping',
            'Sorghum+Pigeon Pea': '3:1 alternate rows',
            'Maize+Soybean': '2:1 paired rows',
            'Wheat+Mustard': '8:1 strip cropping',
            'Chickpea+Linseed': '8:2 strip cropping',
            'Groundnut+Pearl Millet': '6:2 strip cropping',
            'Tomato+Marigold': '5:1 border rows',
            'Onion+Radish': '4:1 alternate rows'
        }
        
        key = f"{primary_crop}+{intercrop}"
        return patterns.get(key, 'Alternate rows')
    
    def get_intercrop_benefits(self, primary_crop, intercrop):
        """Get benefits of the intercrop combination"""
        benefits = {
            'Cotton+Green Gram': 'Early income, fixes nitrogen, reduces bollworm',
            'Cotton+Soybean': 'Nitrogen fixation, canopy complementarity',
            'Soybean+Pigeon Pea': 'Different maturity, deep nutrient mining',
            'Sorghum+Pigeon Pea': 'Drought escape, fixes nitrogen',
            'Maize+Soybean': 'Nitrogen fixation, different canopy height',
            'Wheat+Mustard': 'Phosphorus mobilization, extra oil income',
            'Chickpea+Linseed': 'Disease break, oil+protein combination',
            'Groundnut+Pearl Millet': 'Drought tolerance, different root depth',
            'Tomato+Marigold': 'Nematode control, pest repellent',
            'Onion+Radish': 'Space utilization, different harvest times'
        }
        
        key = f"{primary_crop}+{intercrop}"
        return benefits.get(key, 'Improves soil health and increases yield')
    
    def calculate_intercrop_score(self, crop1, crop2, location_data):
        """Calculate compatibility score (0-10) - FIXED VERSION"""
        score = 6.0  # Base score
        
        # Nutrient complementarity
        if self.are_nutrients_complementary(crop1, crop2):
            score += 1.5
        
        # Root depth complementarity
        if self.are_roots_complementary(crop1, crop2):
            score += 1.5
        
        # Pest/disease protection
        if self.provides_pest_protection(crop1, crop2):
            score += 1.0
        
        # Water use compatibility
        water_score = self.check_water_compatibility(crop1, crop2, location_data['rainfall'])
        score += water_score
        
        return min(10.0, score)
    
    def get_seasonal_crops(self, season, location_data):
        """Get crops suitable for the season and location - IMPROVED VERSION"""
        rainfall = location_data['rainfall']
        soil_texture = location_data['soil_texture']
        
        # Define crop suitability with more crops
        crop_suitability = {
            'Kharif': ['Soybean', 'Cotton', 'Maize', 'Groundnut', 'Sorghum', 
                      'Pigeon Pea', 'Rice', 'Sugarcane', 'Black Gram', 'Green Gram'],
            'Rabi': ['Wheat', 'Chickpea', 'Onion', 'Tomato', 'Potato', 
                    'Mustard', 'Sunflower', 'Garlic', 'Cabbage', 'Cauliflower'],
            'Summer': ['Vegetables', 'Pulses', 'Oilseeds', 'Moong', 'Urad', 'Cucumber']
        }
        
        # Calculate annual rainfall
        annual_rainfall = 0
        if isinstance(rainfall, dict):
            for month in ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                         'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']:
                if month in rainfall:
                    try:
                        annual_rainfall += float(rainfall[month])
                    except:
                        pass
        
        # Filter based on rainfall
        suitable = []
        if annual_rainfall < 600:
            # Low rainfall areas
            suitable = ['Sorghum', 'Pearl Millet', 'Groundnut', 'Chickpea', 
                       'Moong', 'Urad', 'Mustard', 'Sunflower']
        elif annual_rainfall > 1000:
            # High rainfall areas
            suitable = ['Rice', 'Sugarcane', 'Cotton', 'Soybean', 'Maize', 
                       'Vegetables', 'Pigeon Pea']
        else:
            # Medium rainfall
            suitable = crop_suitability.get(season, [])
        
        # Filter based on soil texture if available
        if soil_texture and 'Texture Classification' in soil_texture:
            soil_type = soil_texture['Texture Classification']
            
            if 'Clay' in soil_type:
                # Clay soils are good for rice, sugarcane, wheat
                suitable = [c for c in suitable if c in ['Rice', 'Sugarcane', 'Wheat', 'Cotton']]
            elif 'Sandy' in soil_type:
                # Sandy soils are good for groundnut, pulses
                suitable = [c for c in suitable if c in ['Groundnut', 'Moong', 'Urad', 'Pearl Millet']]
        
        return suitable[:8]  # Return top 8 suitable crops
    
    def find_compatible_intercrops(self, primary_crop, location_data):
        """Find compatible intercrops for primary crop - EXPANDED VERSION"""
        compatibility_map = {
            'Cotton': ['Green Gram', 'Soybean', 'Cowpea', 'Coriander', 'Onion'],
            'Soybean': ['Pigeon Pea', 'Sorghum', 'Maize', 'Pearl Millet'],
            'Maize': ['Soybean', 'Black Gram', 'Potato', 'Cowpea'],
            'Wheat': ['Mustard', 'Chickpea', 'Linseed', 'Safflower'],
            'Chickpea': ['Linseed', 'Mustard', 'Safflower', 'Wheat'],
            'Groundnut': ['Pearl Millet', 'Sorghum', 'Moong'],
            'Onion': ['Radish', 'Carrot', 'Spinach', 'Coriander'],
            'Tomato': ['Marigold', 'Basil', 'Onion', 'Garlic'],
            'Potato': ['Beans', 'Peas', 'Cabbage'],
            'Sorghum': ['Pigeon Pea', 'Cowpea', 'Moong'],
            'Rice': ['Azolla', 'Fish', 'Duck'],
            'Sugarcane': ['Onion', 'Potato', 'Garlic']
        }
        
        return compatibility_map.get(primary_crop, ['Legumes', 'Vegetables', 'Oilseeds'])
    
    def predict_soil_degradation(self, current_nutrients, crop_history):
        """Predict soil degradation based on crop history - IMPROVED VERSION"""
        degradation_score = 0
        issues = []
        
        # If no nutrients data, return default
        if not current_nutrients:
            return {
                'degradation_score': 0.3,
                'degradation_level': 'Normal',
                'predicted_nutrients': {'N': 250, 'P': 20, 'K': 250},
                'issues': ['No soil nutrient data available'],
                'recommendations': ['Conduct soil test for accurate analysis']
            }
        
        # Calculate nutrient depletion
        n_depletion = 0
        p_depletion = 0
        k_depletion = 0
        
        for crop in crop_history:
            crop_data = self.get_crop_nutrient_extraction(crop['name'])
            n_depletion += crop_data.get('N_extraction', 0)
            p_depletion += crop_data.get('P_extraction', 0)
            k_depletion += crop_data.get('K_extraction', 0)
        
        # Get current nutrient values, handle missing keys
        try:
            current_n = float(current_nutrients.get('N (kg/ha)', 250))
            current_p = float(current_nutrients.get('P (kg/ha)', 20))
            current_k = float(current_nutrients.get('K (kg/ha)', 250))
            current_ph = float(current_nutrients.get('pH', 7.0))
        except (ValueError, TypeError):
            current_n = 250
            current_p = 20
            current_k = 250
            current_ph = 7.0
        
        # Predict future nutrient levels
        predicted_n = current_n - n_depletion
        predicted_p = current_p - p_depletion
        predicted_k = current_k - k_depletion
        
        # Check against thresholds
        if predicted_n < 200:
            degradation_score += 0.3
            issues.append(f"Nitrogen will drop to {predicted_n:.1f} kg/ha (below optimal 250-350)")
        
        if predicted_p < 15:
            degradation_score += 0.25
            issues.append(f"Phosphorus will drop to {predicted_p:.1f} kg/ha (below optimal 15-25)")
        
        if predicted_k < 150:
            degradation_score += 0.2
            issues.append(f"Potassium will drop to {predicted_k:.1f} kg/ha (below optimal 200-300)")
        
        # Check pH
        if current_ph < 5.5 or current_ph > 8.5:
            degradation_score += 0.25
            issues.append(f"pH is {current_ph} (optimal is 6.0-7.5)")
        
        # Check organic carbon if available
        if 'OC (%)' in current_nutrients:
            try:
                oc = float(current_nutrients['OC (%)'])
                if oc < 0.5:
                    degradation_score += 0.2
                    issues.append(f"Organic carbon is low: {oc}% (optimal >0.8%)")
            except:
                pass
        
        return {
            'degradation_score': min(1.0, degradation_score),
            'degradation_level': self.get_degradation_level(degradation_score),
            'predicted_nutrients': {
                'N': predicted_n,
                'P': predicted_p,
                'K': predicted_k
            },
            'issues': issues[:5],
            'recommendations': self.get_degradation_recommendations(degradation_score, predicted_n, predicted_p, predicted_k)
        }
    
    def calculate_profit(self, primary_crop, intercrop, location_data):
        """Calculate expected profit increase"""
        base_profits = {
            'Cotton': 80000,
            'Soybean': 60000,
            'Maize': 55000,
            'Wheat': 50000,
            'Chickpea': 45000,
            'Groundnut': 40000,
            'Onion': 90000,
            'Tomato': 100000
        }
        
        base = base_profits.get(primary_crop, 40000)
        
        # Intercrop profit addition
        intercrop_profits = {
            'Green Gram': 15000,
            'Soybean': 20000,
            'Mustard': 10000,
            'Chickpea': 12000,
            'Marigold': 5000,
            'Basil': 8000
        }
        
        addition = intercrop_profits.get(intercrop, 8000)
        
        return base + addition
    
    def analyze_farmer_data(self, district, taluka, crop_history, future_plan):
        """Main analysis function for farmer data"""
        # Get location data
        location_data = self.get_location_data(district, taluka)
        
        # Get current season
        current_month = datetime.now().strftime('%b')
        current_season = self.month_to_season.get(current_month, 'Kharif')
        
        # Predict soil degradation
        soil_analysis = self.predict_soil_degradation(
            location_data['nutrients'], 
            crop_history
        )
        
        # Get intercrop recommendations
        intercrop_recs = self.recommend_intercrops(
            location_data, 
            [c['name'] for c in crop_history] if crop_history else [],
            current_season
        )
        
        # Calculate profit projections
        profit_analysis = self.calculate_profit_analysis(intercrop_recs)
        
        # Generate final report
        return {
            'location': {
                'district': district,
                'taluka': taluka,
                'current_season': current_season,
                'current_month': current_month
            },
            'soil_health': soil_analysis,
            'intercrop_recommendations': intercrop_recs,
            'profit_analysis': profit_analysis,
            'alerts': self.generate_alerts(soil_analysis),
            'best_option': self.select_best_option(intercrop_recs) if intercrop_recs else None
        }
    
    def calculate_profit_analysis(self, intercrop_recs):
        """Calculate profit analysis"""
        if not intercrop_recs:
            return {}
        
        profits = [rec['profit_increase'] for rec in intercrop_recs]
        
        return {
            'max_profit': max(profits) if profits else 0,
            'min_profit': min(profits) if profits else 0,
            'avg_profit': sum(profits) / len(profits) if profits else 0,
            'total_options': len(intercrop_recs)
        }
    
    def generate_alerts(self, soil_analysis):
        """Generate alerts based on soil analysis"""
        alerts = []
        
        if soil_analysis['degradation_level'] == 'Critical':
            alerts.append({
                'level': 'critical',
                'message': 'Soil degradation detected! Immediate action required.',
                'action': 'Change cropping pattern and add organic matter'
            })
        
        if soil_analysis['degradation_level'] == 'Warning':
            alerts.append({
                'level': 'warning',
                'message': 'Soil health is declining.',
                'action': 'Adjust fertilization and consider intercrops'
            })
        
        return alerts
    
    def select_best_option(self, intercrop_recs):
        """Select the best intercrop option"""
        if not intercrop_recs:
            return None
        
        # Score based on 70% profit, 30% compatibility
        for rec in intercrop_recs:
            rec['final_score'] = (0.7 * (rec['profit_increase'] / 100000)) + (0.3 * (rec['score'] / 10))
        
        best = max(intercrop_recs, key=lambda x: x['final_score'])
        return best