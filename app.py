from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import json
import numpy as np
import pandas as pd
from agro_nova_model import AgroNovaModel
from datetime import datetime
import traceback
import math

app = Flask(__name__)
CORS(app)

# Load ML model and components
try:
    model = joblib.load('./plant_health_model.pkl')
    scaler = joblib.load('./scaler.pkl')
    
    with open('./optimal_ranges.json', 'r') as f:
        optimal_ranges = json.load(f)
    
    with open('./features.json', 'r') as f:
        features = json.load(f)
    
    # Load training data
    df = pd.read_csv('./plant_health_data.csv')
    
    print("✅ Model and components loaded successfully!")
    
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None
    scaler = None
    optimal_ranges = {}
    features = []
    df = pd.DataFrame()

try:
    agro_nova_model = AgroNovaModel(data_dir='./data/maharashtra/')
    print("✅ AgroNova model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading AgroNova model: {e}")
    agro_nova_model = None

# ============================================
# HELPER FUNCTIONS
# ============================================

def get_health_status(score):
    """Get health status category"""
    if score >= 85: return 'Excellent'
    if score >= 70: return 'Good'
    if score >= 55: return 'Fair'
    if score >= 40: return 'Poor'
    return 'Critical'

def categorize_health(score, prediction):
    """Categorize health for quick assessment"""
    if prediction == 'Healthy' and score >= 70:
        return 'Optimal'
    elif prediction == 'Moderate Stress' and score >= 60:
        return 'Manageable'
    elif prediction == 'High Stress' or score < 40:
        return 'Critical'
    else:
        return 'Needs Attention'

def get_key_issues(penalties, recommendations):
    """Extract key issues from penalties and recommendations"""
    issues = []
    
    # Add from penalties
    for feat, pen in penalties[:2]:
        if pen > 10:
            issues.append(f'{feat} (Major impact: -{pen} pts)')
        elif pen > 5:
            issues.append(f'{feat} (Moderate impact: -{pen} pts)')
    
    # Add from critical recommendations
    for rec in recommendations:
        if rec.get('severity') == 'Critical':
            issues.append(rec.get('issue', 'Critical issue'))
            if len(issues) >= 3:
                break
    
    return issues[:3]  # Return top 3 issues

# ============================================
# HEALTH SCORE CALCULATION
# ============================================

def calculate_plant_health_score(features_dict, ranges, ml_confidence=None):
    """
    Optimized health score calculation with feature-specific logic
    """
    score = 100
    penalty_factors = []
    
    # Feature weights based on importance
    weights = {
        'Soil_Moisture': 0.18,
        'Nitrogen_Level': 0.15,
        'Chlorophyll_Content': 0.12,
        'Electrochemical_Signal': 0.12,
        'Soil_pH': 0.10,
        'Light_Intensity': 0.09,
        'Ambient_Temperature': 0.08,
        'Soil_Temperature': 0.06,
        'Humidity': 0.05,
        'Potassium_Level': 0.03,
        'Phosphorus_Level': 0.02
    }
    
    for feature, range_info in ranges.items():
        if feature in features_dict:
            value = features_dict[feature]
            weight = weights.get(feature, 0.05)
            
            if value is None or pd.isna(value):
                penalty = 20 * weight
                penalty_factors.append((feature, round(penalty, 1)))
                score -= penalty
                continue
                
            min_val = range_info.get('min', 0)
            max_val = range_info.get('max', 100)
            
            # Special handling for Light Intensity
            if feature == 'Light_Intensity' and value < min_val:
                # Exponential penalty for low light
                deviation_ratio = (min_val - value) / min_val
                penalty = min(25 * weight, (deviation_ratio ** 0.7) * 100 * weight * 2)
                penalty_factors.append((feature, round(penalty, 1)))
                score -= penalty
                continue
                
            if value < min_val:
                deviation = (min_val - value) / min_val if min_val > 0 else 1
                critical_low = range_info.get('critical', min_val * 0.5)
                
                if value <= critical_low:
                    penalty = min(25 * weight, deviation * 100 * weight * 1.8)
                else:
                    penalty = min(15 * weight, deviation * 100 * weight)
                penalty_factors.append((feature, round(penalty, 1)))
                score -= penalty
                
            elif value > max_val:
                deviation = (value - max_val) / max_val if max_val > 0 else 1
                critical_high = range_info.get('critical_high', max_val * 1.5)
                
                if value >= critical_high:
                    penalty = min(25 * weight, deviation * 100 * weight * 1.8)
                else:
                    penalty = min(15 * weight, deviation * 100 * weight)
                penalty_factors.append((feature, round(penalty, 1)))
                score -= penalty
    
    # ML confidence adjustment
    if ml_confidence:
        if ml_confidence.get('high_stress', 0) > 80:
            score *= 0.65  # Reduce by 35% for high confidence in stress
        elif ml_confidence.get('high_stress', 0) > 60:
            score *= 0.8   # Reduce by 20%
        elif ml_confidence.get('moderate_stress', 0) > 80:
            score *= 0.85  # Reduce by 15%
        elif ml_confidence.get('healthy', 0) > 90:
            score = min(100, score * 1.05)  # Boost by 5%
    
    score = max(0, min(100, round(score, 1)))
    return score, penalty_factors

# ============================================
# PREDICTION CONSENSUS LOGIC
# ============================================

def get_final_prediction(ml_prediction, health_score, ml_confidence):
    """
    Smart consensus between ML model and health score
    """
    # Extract confidence values
    healthy_conf = ml_confidence.get('healthy', 0)
    moderate_conf = ml_confidence.get('moderate_stress', 0)
    high_conf = ml_confidence.get('high_stress', 0)
    
    # Rule 1: Very confident predictions take priority
    if high_conf > 85:
        return 'High Stress'
    elif healthy_conf > 85:
        return 'Healthy'
    
    # Rule 2: Health score based decision with ML adjustment
    if health_score >= 85:
        # Excellent health score overrides moderate ML predictions
        if ml_prediction == 'Moderate Stress' and moderate_conf < 80:
            return 'Healthy'
        else:
            return 'Healthy'
    elif health_score >= 70:
        # Good health - check ML confidence
        if ml_prediction == 'High Stress' and high_conf > 70:
            return 'Moderate Stress'  # Downgrade from High to Moderate
        elif ml_prediction == 'Healthy' and healthy_conf > 60:
            return 'Healthy'
        else:
            return 'Moderate Stress'
    elif health_score >= 50:
        return 'Moderate Stress'
    elif health_score >= 30:
        # Poor health - ML can upgrade to High Stress if confident
        if ml_prediction == 'High Stress' and high_conf > 60:
            return 'High Stress'
        else:
            return 'Moderate Stress'
    else:
        return 'High Stress'

# ============================================
# RECOMMENDATION GENERATION
# ============================================

def generate_recommendations(input_features, final_prediction, ml_prediction, health_score, ml_confidence):
    """Generate recommendations based on feature values and predictions"""
    recommendations = []
    
    # Add ML insight if there's discrepancy
    if final_prediction != ml_prediction:
        if final_prediction == 'Healthy' and ml_prediction != 'Healthy':
            recommendations.append({
                'issue': 'Optimistic Assessment',
                'severity': 'Info',
                'action': f'Health score ({health_score}/100) suggests better condition than ML prediction ({ml_prediction})',
                'priority': 3,
                'feature': 'consensus'
            })
        elif final_prediction == 'High Stress' and ml_prediction != 'High Stress':
            recommendations.append({
                'issue': 'Severe Condition Detected',
                'severity': 'Critical',
                'action': f'Combined analysis indicates severe stress despite ML prediction ({ml_prediction})',
                'priority': 1,
                'feature': 'consensus'
            })
    
    # Add parameter-based recommendations
    feature_names = {
        'Soil_Moisture': 'Soil Moisture',
        'Light_Intensity': 'Light',
        'Nitrogen_Level': 'Nitrogen',
        'Soil_pH': 'Soil pH',
        'Electrochemical_Signal': 'Stress Signal',
        'Ambient_Temperature': 'Temperature',
        'Soil_Temperature': 'Soil Temperature',
        'Humidity': 'Humidity',
        'Phosphorus_Level': 'Phosphorus',
        'Potassium_Level': 'Potassium',
        'Chlorophyll_Content': 'Chlorophyll'
    }
    
    units = {
        'Soil_Moisture': '%',
        'Light_Intensity': 'lux',
        'Nitrogen_Level': 'ppm',
        'Soil_pH': 'pH',
        'Electrochemical_Signal': 'mV',
        'Ambient_Temperature': '°C',
        'Soil_Temperature': '°C',
        'Humidity': '%',
        'Phosphorus_Level': 'ppm',
        'Potassium_Level': 'ppm',
        'Chlorophyll_Content': 'SPAD'
    }
    
    for feature, value in input_features.items():
        if feature in optimal_ranges and not pd.isna(value):
            range_info = optimal_ranges[feature]
            min_val = range_info.get('min', 0)
            max_val = range_info.get('max', 100)
            
            if value < min_val * 0.8:  # Very low
                severity = 'Critical'
                priority = 1
            elif value < min_val:  # Low
                severity = 'Warning'
                priority = 2
            elif value > max_val * 1.2:  # Very high
                severity = 'Critical'
                priority = 1
            elif value > max_val:  # High
                severity = 'Warning'
                priority = 2
            else:
                continue
            
            name = feature_names.get(feature, feature)
            unit = units.get(feature, '')
            
            if value < min_val:
                action = f'Increase {name.lower()}. Current: {value}{unit}, Target: {min_val}-{max_val}{unit}'
                issue = f'Low {name}'
            else:
                action = f'Reduce {name.lower()}. Current: {value}{unit}, Target: {min_val}-{max_val}{unit}'
                issue = f'High {name}'
            
            recommendations.append({
                'issue': issue,
                'severity': severity,
                'action': action,
                'priority': priority,
                'feature': feature
            })
    
    # Sort recommendations by priority and severity
    severity_order = {'Critical': 0, 'Warning': 1, 'Info': 2}
    recommendations.sort(key=lambda x: (x['priority'], severity_order.get(x['severity'], 3)))
    
    # Limit to top 5 recommendations
    return recommendations[:5]

# ============================================
# PREDICTION PROCESSING FUNCTION
# ============================================

def process_prediction(input_features):
    """Process a single prediction with all calculations"""
    if model is None or scaler is None:
        return None, "Model not loaded"
    
    try:
        # Scale features
        input_df = pd.DataFrame([input_features])
        input_scaled = scaler.transform(input_df[features])
        
        # Make prediction
        prediction = model.predict(input_scaled)[0]
        probabilities = model.predict_proba(input_scaled)[0]
        
        # Get ML prediction
        health_labels = {0: 'Healthy', 1: 'Moderate Stress', 2: 'High Stress'}
        ml_prediction = health_labels.get(prediction, 'Unknown')
        
        # Calculate confidence
        ml_confidence = {
            'healthy': round(probabilities[0] * 100, 1),
            'moderate_stress': round(probabilities[1] * 100, 1),
            'high_stress': round(probabilities[2] * 100, 1)
        }
        
        # Calculate health score WITH ML confidence
        health_score, penalty_factors = calculate_plant_health_score(
            input_features, 
            optimal_ranges,
            ml_confidence
        )
        
        # Get final consensus prediction
        final_prediction = get_final_prediction(
            ml_prediction, 
            health_score, 
            ml_confidence
        )
        
        # Generate recommendations
        recommendations = generate_recommendations(
            input_features, 
            final_prediction, 
            ml_prediction, 
            health_score, 
            ml_confidence
        )
        
        # Prepare result
        result = {
            'prediction': final_prediction,
            'ml_prediction': ml_prediction,
            'ml_confidence': ml_confidence,
            'health_score': health_score,
            'health_status': get_health_status(health_score),
            'health_category': categorize_health(health_score, final_prediction),
            'penalty_factors': [
                {'feature': feat, 'penalty': pen} 
                for feat, pen in penalty_factors[:3]
            ] if penalty_factors else [],
            'recommendations': recommendations,
            'nutrient_balance': {
                'nitrogen': input_features.get('Nitrogen_Level', 0),
                'phosphorus': input_features.get('Phosphorus_Level', 0),
                'potassium': input_features.get('Potassium_Level', 0),
                'ratio': f"{input_features.get('Nitrogen_Level', 0):.1f}:{input_features.get('Phosphorus_Level', 0):.1f}:{input_features.get('Potassium_Level', 0):.1f}"
            },
            'key_issues': get_key_issues(penalty_factors, recommendations),
            'timestamp': datetime.now().isoformat()
        }
        
        return result, None
        
    except Exception as e:
        return None, str(e)

# ============================================
# API ENDPOINTS
# ============================================

@app.route('/')
def home():
    """Home endpoint - API status"""
    return jsonify({
        'message': 'Plant Health Prediction API',
        'status': 'active',
        'model_loaded': model is not None,
        'version': '1.0.0',
        'endpoints': {
            '/': 'API status',
            '/api/health': 'Health check',
            '/api/features': 'Get feature information',
            '/api/predict': 'Single plant prediction (POST)',
            '/api/batch-predict': 'Batch prediction (POST)',
            '/api/analyze': 'Analyze with optional parameters (POST)',
            '/api/stats': 'Model statistics (GET)'
        }
    })

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy' if model is not None else 'unhealthy',
        'model_loaded': model is not None,
        'scaler_loaded': scaler is not None,
        'features_loaded': len(features) > 0,
        'optimal_ranges_loaded': len(optimal_ranges) > 0,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/features', methods=['GET'])
def get_features_info():
    """Get feature information and optimal ranges"""
    return jsonify({
        'features': features,
        'optimal_ranges': optimal_ranges,
        'feature_descriptions': {
            'Soil_Moisture': 'Water content in soil (Volumetric Water Content)',
            'Ambient_Temperature': 'Air temperature around the plant',
            'Soil_Temperature': 'Temperature at root zone depth',
            'Humidity': 'Relative air humidity',
            'Light_Intensity': 'Amount of light received (Photosynthetically Active Radiation)',
            'Soil_pH': 'Soil acidity/alkalinity level',
            'Nitrogen_Level': 'Nitrogen concentration in soil',
            'Phosphorus_Level': 'Phosphorus concentration in soil',
            'Potassium_Level': 'Potassium concentration in soil',
            'Chlorophyll_Content': 'Leaf chlorophyll measurement (SPAD units)',
            'Electrochemical_Signal': 'Plant electrical activity (stress indicator)'
        }
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    """Single plant prediction endpoint"""
    try:
        if model is None or scaler is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        data = request.json
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Prepare input features
        input_features = {}
        missing_features = []
        
        for feature in features:
            value = data.get(feature)
            if value is None:
                # Use median from training data for missing features
                if not df.empty and feature in df.columns:
                    value = df[feature].median()
                    missing_features.append(feature)
                else:
                    value = 0
            try:
                input_features[feature] = float(value)
            except (ValueError, TypeError):
                return jsonify({'error': f'Invalid value for {feature}'}), 400
        
        # Process prediction
        result, error = process_prediction(input_features)
        
        if error:
            return jsonify({'error': f'Prediction failed: {error}'}), 500
        
        # Add metadata
        result['missing_features_used'] = missing_features if missing_features else None
        result['input_features'] = input_features
        
        return jsonify(result)
        
    except Exception as e:
        print(f"Prediction error: {traceback.format_exc()}")
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/api/batch-predict', methods=['POST'])
def batch_predict():
    """Batch prediction endpoint"""
    try:
        if model is None or scaler is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        data = request.json
        
        if not data or 'plants' not in data:
            return jsonify({'error': 'No plants data provided'}), 400
        
        plants = data['plants']
        if not isinstance(plants, list) or len(plants) == 0:
            return jsonify({'error': 'Plants must be a non-empty list'}), 400
        
        results = []
        successful = 0
        failed = 0
        
        for idx, plant_data in enumerate(plants):
            try:
                # Prepare features for this plant
                input_features = {}
                for feature in features:
                    value = plant_data.get(feature)
                    if value is None and not df.empty and feature in df.columns:
                        value = df[feature].median()
                    elif value is None:
                        value = 0
                    
                    try:
                        input_features[feature] = float(value)
                    except:
                        input_features[feature] = 0.0
                
                # Process prediction
                result, error = process_prediction(input_features)
                
                if error:
                    failed += 1
                    results.append({
                        'plant_id': plant_data.get('plant_id', f'plant_{idx+1}'),
                        'error': error,
                        'prediction': 'Error',
                        'health_score': 0
                    })
                else:
                    successful += 1
                    result['plant_id'] = plant_data.get('plant_id', f'plant_{idx+1}')
                    results.append(result)
                    
            except Exception as e:
                failed += 1
                print(f"Error processing plant {idx}: {e}")
                results.append({
                    'plant_id': plant_data.get('plant_id', f'plant_{idx+1}'),
                    'error': str(e),
                    'prediction': 'Error',
                    'health_score': 0
                })
        
        # Calculate statistics
        if successful == 0:
            return jsonify({'error': 'All plants failed processing'}), 400
        
        healthy_count = sum(1 for r in results if r.get('prediction') == 'Healthy')
        moderate_count = sum(1 for r in results if r.get('prediction') == 'Moderate Stress')
        high_count = sum(1 for r in results if r.get('prediction') == 'High Stress')
        
        valid_scores = [r.get('health_score', 0) for r in results if isinstance(r.get('health_score'), (int, float))]
        avg_score = round(sum(valid_scores) / len(valid_scores), 1) if valid_scores else 0
        
        stats = {
            'total_plants': len(results),
            'successful': successful,
            'failed': failed,
            'healthy_count': healthy_count,
            'moderate_stress_count': moderate_count,
            'high_stress_count': high_count,
            'average_health_score': avg_score,
            'min_score': min(valid_scores) if valid_scores else 0,
            'max_score': max(valid_scores) if valid_scores else 0
        }
        
        return jsonify({
            'summary': stats,
            'results': results
        })
        
    except Exception as e:
        print(f"Batch prediction error: {traceback.format_exc()}")
        return jsonify({'error': f'Batch prediction failed: {str(e)}'}), 500

@app.route('/api/analyze', methods=['POST'])
def analyze():
    """Advanced analysis with optional parameters"""
    try:
        data = request.json
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Check if batch or single prediction
        if 'plants' in data:
            # Batch analysis
            return batch_predict()
        else:
            # Single analysis with optional parameters
            analysis_type = data.get('analysis_type', 'basic')
            
            if analysis_type == 'trend':
                # Time series analysis
                time_series = data.get('time_series', [])
                if not time_series:
                    return jsonify({'error': 'No time series data provided'}), 400
                
                predictions = []
                for ts_data in time_series:
                    # Prepare features
                    input_features = {}
                    for feature in features:
                        value = ts_data.get(feature)
                        if value is None and not df.empty and feature in df.columns:
                            value = df[feature].median()
                        elif value is None:
                            value = 0
                        
                        try:
                            input_features[feature] = float(value)
                        except:
                            input_features[feature] = 0.0
                    
                    result, error = process_prediction(input_features)
                    if result:
                        result['timestamp'] = ts_data.get('timestamp', datetime.now().isoformat())
                        predictions.append(result)
                
                # Calculate trend
                if len(predictions) >= 2:
                    first_score = predictions[0].get('health_score', 0)
                    last_score = predictions[-1].get('health_score', 0)
                    trend = 'improving' if last_score > first_score else 'declining' if last_score < first_score else 'stable'
                else:
                    trend = 'insufficient_data'
                
                return jsonify({
                    'analysis_type': 'trend',
                    'total_readings': len(predictions),
                    'trend': trend,
                    'predictions': predictions
                })
            
            else:
                # Basic single prediction
                return predict()
                
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get model statistics and information"""
    return jsonify({
        'model_type': 'Random Forest Classifier',
        'model_status': 'loaded' if model is not None else 'not loaded',
        'features_count': len(features),
        'features_list': features,
        'training_samples': len(df) if not df.empty else 0,
        'classes': ['Healthy', 'Moderate Stress', 'High Stress'],
        'health_score_ranges': {
            'Excellent': '85-100',
            'Good': '70-84',
            'Fair': '55-69',
            'Poor': '40-54',
            'Critical': '0-39'
        },
        'health_categories': {
            'Optimal': 'Healthy with score >= 70',
            'Manageable': 'Moderate Stress with score >= 60',
            'Critical': 'High Stress or score < 40',
            'Needs Attention': 'All other cases'
        },
        'optimal_ranges_configured': len(optimal_ranges) > 0,
        'api_version': '1.0.0',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/sample-data', methods=['GET'])
def get_sample_data():
    """Get sample data for testing"""
    samples = {
        'healthy_sample': {
            'Soil_Moisture': 28.0,
            'Ambient_Temperature': 25.0,
            'Soil_Temperature': 22.0,
            'Humidity': 65.0,
            'Light_Intensity': 35000.0,
            'Soil_pH': 6.5,
            'Nitrogen_Level': 35.0,
            'Phosphorus_Level': 20.0,
            'Potassium_Level': 45.0,
            'Chlorophyll_Content': 42.0,
            'Electrochemical_Signal': 0.3
        },
        'stressed_sample': {
            'Soil_Moisture': 14.8,
            'Ambient_Temperature': 28.9,
            'Soil_Temperature': 21.9,
            'Humidity': 55.3,
            'Light_Intensity': 556.2,
            'Soil_pH': 5.58,
            'Nitrogen_Level': 10.0,
            'Phosphorus_Level': 45.8,
            'Potassium_Level': 39.1,
            'Chlorophyll_Content': 35.7,
            'Electrochemical_Signal': 0.94
        },
        'moderate_sample': {
            'Soil_Moisture': 33.3,
            'Ambient_Temperature': 20.2,
            'Soil_Temperature': 16.3,
            'Humidity': 56.3,
            'Light_Intensity': 455.7,
            'Soil_pH': 6.11,
            'Nitrogen_Level': 15.9,
            'Phosphorus_Level': 10.5,
            'Potassium_Level': 24.9,
            'Chlorophyll_Content': 33.9,
            'Electrochemical_Signal': 1.68
        }
    }
    
    return jsonify({
        'samples': samples,
        'description': 'Sample data for testing the prediction model',
        'usage': 'Use these samples in POST requests to /api/predict'
    })

@app.route('/api/validate', methods=['POST'])
def validate_input():
    """Validate input data against optimal ranges"""
    try:
        data = request.json
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        validation_results = {}
        warnings = []
        critical_issues = []
        
        for feature in features:
            value = data.get(feature)
            
            if value is None:
                validation_results[feature] = {
                    'status': 'missing',
                    'message': 'Feature value is missing',
                    'severity': 'warning'
                }
                warnings.append(f'{feature} is missing')
                continue
            
            try:
                value = float(value)
            except (ValueError, TypeError):
                validation_results[feature] = {
                    'status': 'invalid',
                    'message': 'Value must be a number',
                    'severity': 'critical'
                }
                critical_issues.append(f'{feature} has invalid value')
                continue
            
            if feature in optimal_ranges:
                range_info = optimal_ranges[feature]
                min_val = range_info.get('min', 0)
                max_val = range_info.get('max', 100)
                critical_low = range_info.get('critical', min_val * 0.5)
                critical_high = range_info.get('critical_high', max_val * 1.5)
                
                if value < min_val:
                    if value <= critical_low:
                        status = 'critical_low'
                        severity = 'critical'
                        critical_issues.append(f'{feature} is critically low')
                    else:
                        status = 'low'
                        severity = 'warning'
                        warnings.append(f'{feature} is below optimal range')
                elif value > max_val:
                    if value >= critical_high:
                        status = 'critical_high'
                        severity = 'critical'
                        critical_issues.append(f'{feature} is critically high')
                    else:
                        status = 'high'
                        severity = 'warning'
                        warnings.append(f'{feature} is above optimal range')
                else:
                    status = 'optimal'
                    severity = 'good'
                
                validation_results[feature] = {
                    'status': status,
                    'severity': severity,
                    'value': value,
                    'optimal_range': f'{min_val}-{max_val}',
                    'message': f'Value: {value}, Optimal: {min_val}-{max_val}'
                }
            else:
                validation_results[feature] = {
                    'status': 'unknown',
                    'message': 'No optimal range defined for this feature',
                    'severity': 'info'
                }
        
        overall_status = 'valid'
        if critical_issues:
            overall_status = 'critical'
        elif warnings:
            overall_status = 'warning'
        
        return jsonify({
            'overall_status': overall_status,
            'critical_issues': critical_issues,
            'warnings': warnings,
            'validation_results': validation_results,
            'summary': {
                'total_features': len(features),
                'valid_features': sum(1 for v in validation_results.values() if v['status'] == 'optimal'),
                'warning_features': len(warnings),
                'critical_features': len(critical_issues),
                'missing_features': sum(1 for v in validation_results.values() if v['status'] == 'missing')
            }
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/agronova/districts', methods=['GET'])
def get_districts():
    """Get list of districts in Maharashtra"""
    try:
        if agro_nova_model is None:
            return jsonify({'error': 'AgroNova model not loaded'}), 500
        
        # Extract unique districts from rainfall data
        districts = sorted(agro_nova_model.rainfall_df['District'].unique().tolist())
        
        return jsonify({
            'districts': districts,
            'count': len(districts)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/agronova/talukas', methods=['POST'])
def get_talukas():
    """Get talukas for a district"""
    try:
        if agro_nova_model is None:
            return jsonify({'error': 'AgroNova model not loaded'}), 500
        
        data = request.json
        district = data.get('district')
        
        if not district:
            return jsonify({'error': 'District not provided'}), 400
        
        # Get talukas for district
        talukas = agro_nova_model.rainfall_df[
            agro_nova_model.rainfall_df['District'] == district
        ]['Taluka'].unique().tolist()
        
        return jsonify({
            'district': district,
            'talukas': sorted(talukas),
            'count': len(talukas)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/agronova/location-data', methods=['POST'])
def get_location_data():
    """Get all data for a district-taluka"""
    try:
        if agro_nova_model is None:
            return jsonify({'error': 'AgroNova model not loaded'}), 500
        
        data = request.json
        district = data.get('district')
        taluka = data.get('taluka')
        
        if not district or not taluka:
            return jsonify({'error': 'District and taluka required'}), 400
        
        location_data = agro_nova_model.get_location_data(district, taluka)
        
        if not location_data['rainfall']:
            return jsonify({'error': 'Location not found'}), 404
        
        return jsonify({
            'location': {
                'district': district,
                'taluka': taluka
            },
            'rainfall': location_data['rainfall'],
            'temperature': location_data['temperature'],
            'soil_texture': location_data['soil_texture'],
            'nutrients': location_data['nutrients']
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/agronova/analyze', methods=['POST'])
def agronova_analyze():
    """Main AgroNova analysis endpoint"""
    try:
        if agro_nova_model is None:
            return jsonify({'error': 'AgroNova model not loaded'}), 500
        
        # Check content type
        if not request.is_json:
            return jsonify({'error': 'Content-Type must be application/json'}), 415
        
        data = request.get_json(silent=True)
        
        if data is None:
            return jsonify({'error': 'Invalid JSON data'}), 400
        
        # Debug: Print received data
        print("AgroNova received data:", data)
        
        # Required fields
        district = data.get('district')
        taluka = data.get('taluka')
        crop_history = data.get('crop_history', [])  # Array of {name, season, yield}
        future_plan = data.get('future_plan', '')  # What farmer wants to plant
        
        print(f"District: {district}, Taluka: {taluka}")
        
        if not district or not taluka:
            return jsonify({'error': 'District and taluka required'}), 400
        
        # Get location data
        location_data = agro_nova_model.get_location_data(district, taluka)
        
        if not location_data['rainfall']:
            return jsonify({
                'error': f'Location not found: {district} - {taluka}',
                'available_districts': list(agro_nova_model.rainfall_df['District'].unique())[:10]
            }), 404
        
        # Analyze farmer data
        analysis = agro_nova_model.analyze_farmer_data(
            district, taluka, crop_history, future_plan
        )
        
        # Add metadata
        analysis['timestamp'] = datetime.now().isoformat()
        analysis['model_version'] = 'AgroNova Pro v1.0'
        analysis['status'] = 'success'
        
        print(f"Analysis complete for {district} - {taluka}")
        
        return jsonify(analysis)
        
    except Exception as e:
        print(f"AgroNova analysis error: {traceback.format_exc()}")
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc() if app.debug else None
        }), 500

@app.route('/api/agronova/crops', methods=['GET'])
def get_available_crops():
    """Get list of available crops for Maharashtra"""
    try:
        crops = [
            'Soybean', 'Cotton', 'Maize', 'Wheat', 'Sorghum',
            'Chickpea', 'Groundnut', 'Onion', 'Tomato', 'Potato',
            'Mustard', 'Sunflower', 'Pigeon Pea', 'Green Gram',
            'Black Gram', 'Pearl Millet', 'Rice', 'Sugarcane'
        ]
        
        return jsonify({
            'crops': sorted(crops),
            'seasons': ['Kharif', 'Rabi', 'Summer'],
            'count': len(crops)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/agronova/sample-data', methods=['GET'])
def get_agronova_sample():
    """Get sample data for AgroNova"""
    return jsonify({
        'sample_input': {
            'district': 'Pune',
            'taluka': 'Haveli',
            'crop_history': [
                {'name': 'Soybean', 'season': 'Kharif 2023', 'yield': '25 quintals/ha'},
                {'name': 'Wheat', 'season': 'Rabi 2023-24', 'yield': '30 quintals/ha'}
            ],
            'future_plan': 'Cotton for next Kharif',
            'resources': {
                'irrigation': True,
                'organic_manure': False,
                'budget': 'medium'
            }
        },
        'description': 'Sample input for AgroNova Pro analysis'
    })

# ============================================
# ERROR HANDLERS
# ============================================

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify({'error': 'Method not allowed'}), 405

@app.errorhandler(500)
def internal_server_error(error):
    return jsonify({'error': 'Internal server error'}), 500

# ============================================
# MAIN EXECUTION
# ============================================

if __name__ == '__main__':
    print("🚀 Plant Health Prediction API Started")
    print("=" * 50)
    print(f"📊 Features: {len(features)}")
    print(f"🤖 Model: {'Loaded' if model else 'Not loaded'}")
    print(f"📈 Training Samples: {len(df) if not df.empty else 0}")
    print(f"🎯 Optimal Ranges: {len(optimal_ranges)} features configured")
    print("=" * 50)
    print("📋 Available Endpoints:")
    print("  GET  /              - API status")
    print("  GET  /api/health    - Health check")
    print("  GET  /api/features  - Feature information")
    print("  POST /api/predict   - Single prediction")
    print("  POST /api/batch-predict - Batch prediction")
    print("  POST /api/analyze   - Advanced analysis")
    print("  GET  /api/stats     - Model statistics")
    print("  GET  /api/sample-data - Sample test data")
    print("  POST /api/validate  - Input validation")
    print("=" * 50)
    app.run(debug=True, port=5000, host='0.0.0.0')