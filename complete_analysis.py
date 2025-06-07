#!/usr/bin/env python3
"""
Complete analysis of the actual public_cases.json data
Comprehensive reverse engineering with the authentic dataset
"""

import json
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

def load_authentic_data():
    """Load the authentic public_cases.json data"""
    try:
        with open('public_cases.json', 'r') as f:
            data = json.load(f)
        
        df = pd.DataFrame(data)
        print(f"Loaded authentic dataset: {len(df)} cases")
        
        # Verify data structure
        expected_columns = ['trip_duration_days', 'miles_traveled', 'total_receipts_amount', 'reimbursement_amount']
        if not all(col in df.columns for col in expected_columns):
            print("Warning: Data structure doesn't match expected format")
            print(f"Available columns: {df.columns.tolist()}")
        
        # Clean and validate
        df_clean = df[expected_columns].copy()
        df_clean = df_clean.dropna()
        
        print(f"Clean dataset: {len(df_clean)} cases")
        print(f"Data range:")
        print(f"  Trip duration: {df_clean['trip_duration_days'].min()}-{df_clean['trip_duration_days'].max()} days")
        print(f"  Miles: {df_clean['miles_traveled'].min()}-{df_clean['miles_traveled'].max()}")
        print(f"  Receipts: ${df_clean['total_receipts_amount'].min():.2f}-${df_clean['total_receipts_amount'].max():.2f}")
        print(f"  Reimbursement: ${df_clean['reimbursement_amount'].min():.2f}-${df_clean['reimbursement_amount'].max():.2f}")
        
        return df_clean
        
    except FileNotFoundError:
        print("Error: public_cases.json not found")
        return None
    except json.JSONDecodeError:
        print("Error: Invalid JSON format")
        return None

def comprehensive_pattern_analysis(df):
    """Perform comprehensive pattern analysis on the data"""
    
    print("\n=== COMPREHENSIVE PATTERN ANALYSIS ===")
    
    # Basic correlation analysis
    corr_matrix = df.corr()
    print("\nCorrelation with reimbursement_amount:")
    for col in ['trip_duration_days', 'miles_traveled', 'total_receipts_amount']:
        corr = corr_matrix.loc[col, 'reimbursement_amount']
        print(f"  {col}: {corr:.4f}")
    
    # Linear regression baseline
    X = df[['trip_duration_days', 'miles_traveled', 'total_receipts_amount']]
    y = df['reimbursement_amount']
    
    lr = LinearRegression()
    lr.fit(X, y)
    r2_linear = lr.score(X, y)
    
    print(f"\nLinear regression R²: {r2_linear:.6f}")
    print("Linear coefficients:")
    for i, col in enumerate(X.columns):
        print(f"  {col}: {lr.coef_[i]:.4f}")
    print(f"  Intercept: {lr.intercept_:.4f}")
    
    # Test for non-linear patterns
    poly_features = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly_features.fit_transform(X)
    
    lr_poly = LinearRegression()
    lr_poly.fit(X_poly, y)
    r2_poly = lr_poly.score(X_poly, y)
    
    print(f"Polynomial (degree 2) R²: {r2_poly:.6f}")
    print(f"Improvement over linear: {r2_poly - r2_linear:.6f}")
    
    # Random Forest for feature importance
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X, y)
    r2_rf = rf.score(X, y)
    
    print(f"Random Forest R²: {r2_rf:.6f}")
    print("Feature importance:")
    for i, col in enumerate(X.columns):
        print(f"  {col}: {rf.feature_importances_[i]:.4f}")
    
    # Analyze specific patterns
    analyze_threshold_patterns(df)
    analyze_ratio_patterns(df)
    
    return lr, lr_poly, rf, poly_features

def analyze_threshold_patterns(df):
    """Analyze threshold-based patterns in the data"""
    
    print("\n=== THRESHOLD PATTERN ANALYSIS ===")
    
    # Trip duration thresholds
    print("\nTrip duration patterns:")
    for threshold in [1, 2, 3, 5, 7, 10, 14]:
        below = df[df['trip_duration_days'] <= threshold]
        above = df[df['trip_duration_days'] > threshold]
        
        if len(below) >= 5 and len(above) >= 5:
            below_avg_daily = (below['reimbursement_amount'] / below['trip_duration_days']).mean()
            above_avg_daily = (above['reimbursement_amount'] / above['trip_duration_days']).mean()
            
            if abs(above_avg_daily - below_avg_daily) > 5:
                print(f"  {threshold} days: ≤{threshold}: ${below_avg_daily:.2f}/day, >{threshold}: ${above_avg_daily:.2f}/day")
    
    # Miles traveled thresholds
    print("\nMiles traveled patterns:")
    for threshold in [50, 100, 200, 300, 500, 1000]:
        below = df[df['miles_traveled'] <= threshold]
        above = df[df['miles_traveled'] > threshold]
        
        if len(below) >= 5 and len(above) >= 5:
            below_avg_mile = (below['reimbursement_amount'] / below['miles_traveled']).mean()
            above_avg_mile = (above['reimbursement_amount'] / above['miles_traveled']).mean()
            
            if abs(above_avg_mile - below_avg_mile) > 0.1:
                print(f"  {threshold} miles: ≤{threshold}: ${below_avg_mile:.3f}/mile, >{threshold}: ${above_avg_mile:.3f}/mile")
    
    # Receipt amount thresholds
    print("\nReceipt amount patterns:")
    for threshold in [25, 50, 100, 150, 200, 300]:
        below = df[df['total_receipts_amount'] <= threshold]
        above = df[df['total_receipts_amount'] > threshold]
        
        if len(below) >= 5 and len(above) >= 5:
            below_ratio = (below['reimbursement_amount'] / below['total_receipts_amount']).mean()
            above_ratio = (above['reimbursement_amount'] / above['total_receipts_amount']).mean()
            
            if abs(above_ratio - below_ratio) > 0.2:
                print(f"  ${threshold}: ≤${threshold}: {below_ratio:.2f}x, >${threshold}: {above_ratio:.2f}x")

def analyze_ratio_patterns(df):
    """Analyze ratio-based patterns"""
    
    print("\n=== RATIO PATTERN ANALYSIS ===")
    
    # Per-day analysis
    df['daily_rate'] = df['reimbursement_amount'] / df['trip_duration_days']
    print(f"Daily rate: ${df['daily_rate'].mean():.2f} ± ${df['daily_rate'].std():.2f}")
    print(f"Range: ${df['daily_rate'].min():.2f} - ${df['daily_rate'].max():.2f}")
    
    # Per-mile analysis
    df['mile_rate'] = df['reimbursement_amount'] / df['miles_traveled']
    df['mile_rate'] = df['mile_rate'].replace([np.inf, -np.inf], np.nan)
    print(f"Mile rate: ${df['mile_rate'].mean():.3f} ± ${df['mile_rate'].std():.3f}")
    print(f"Range: ${df['mile_rate'].min():.3f} - ${df['mile_rate'].max():.3f}")
    
    # Receipt multiplier analysis
    df['receipt_multiplier'] = df['reimbursement_amount'] / df['total_receipts_amount']
    df['receipt_multiplier'] = df['receipt_multiplier'].replace([np.inf, -np.inf], np.nan)
    print(f"Receipt multiplier: {df['receipt_multiplier'].mean():.2f} ± {df['receipt_multiplier'].std():.2f}")
    print(f"Range: {df['receipt_multiplier'].min():.2f} - {df['receipt_multiplier'].max():.2f}")

def build_optimized_algorithms(df, models):
    """Build and test multiple optimized algorithms"""
    
    print("\n=== BUILDING OPTIMIZED ALGORITHMS ===")
    
    lr, lr_poly, rf, poly_features = models
    
    # Algorithm 1: Enhanced Linear with Manual Adjustments
    def algorithm_enhanced_linear(trip_duration_days, miles_traveled, total_receipts_amount):
        # Base linear prediction
        base = lr.predict([[trip_duration_days, miles_traveled, total_receipts_amount]])[0]
        
        # Manual adjustments based on threshold analysis
        adjustments = 0
        
        # Trip duration adjustments
        if trip_duration_days == 1:
            adjustments += 10.0
        elif trip_duration_days >= 10:
            adjustments += 15.0
        
        # Mileage adjustments
        if miles_traveled <= 100:
            adjustments += 8.0
        elif miles_traveled >= 500:
            adjustments += 5.0
        
        return round(base + adjustments, 2)
    
    # Algorithm 2: Polynomial Model
    def algorithm_polynomial(trip_duration_days, miles_traveled, total_receipts_amount):
        X_input = poly_features.transform([[trip_duration_days, miles_traveled, total_receipts_amount]])
        prediction = lr_poly.predict(X_input)[0]
        return round(prediction, 2)
    
    # Algorithm 3: Segmented Linear Model
    def algorithm_segmented(trip_duration_days, miles_traveled, total_receipts_amount):
        # Different formulas for different trip types
        
        if trip_duration_days == 1:
            # Single day trips
            return round(75.0 + 0.8 * miles_traveled + 1.5 * total_receipts_amount, 2)
        elif trip_duration_days <= 3:
            # Short trips
            return round(45.0 * trip_duration_days + 0.6 * miles_traveled + 1.2 * total_receipts_amount, 2)
        elif trip_duration_days <= 7:
            # Medium trips
            return round(40.0 * trip_duration_days + 0.55 * miles_traveled + 1.15 * total_receipts_amount, 2)
        else:
            # Long trips
            return round(50.0 * trip_duration_days + 0.65 * miles_traveled + 1.3 * total_receipts_amount, 2)
    
    # Algorithm 4: Business Rules Based
    def algorithm_business_rules(trip_duration_days, miles_traveled, total_receipts_amount):
        # Start with base components
        daily_allowance = 42.0 * trip_duration_days
        
        # Tiered mileage
        if miles_traveled <= 50:
            mileage_allowance = miles_traveled * 1.2
        elif miles_traveled <= 200:
            mileage_allowance = 50 * 1.2 + (miles_traveled - 50) * 0.7
        elif miles_traveled <= 500:
            mileage_allowance = 50 * 1.2 + 150 * 0.7 + (miles_traveled - 200) * 0.6
        else:
            mileage_allowance = 50 * 1.2 + 150 * 0.7 + 300 * 0.6 + (miles_traveled - 500) * 0.65
        
        # Receipt handling
        receipt_allowance = total_receipts_amount * 1.25
        
        # Bonuses
        bonus = 0
        if trip_duration_days >= 7:
            bonus += 25.0  # Extended travel bonus
        if trip_duration_days == 1:
            bonus += 15.0  # Single day premium
        
        total = daily_allowance + mileage_allowance + receipt_allowance + bonus
        return round(total, 2)
    
    algorithms = {
        "Enhanced Linear": algorithm_enhanced_linear,
        "Polynomial": algorithm_polynomial,
        "Segmented": algorithm_segmented,
        "Business Rules": algorithm_business_rules
    }
    
    # Test all algorithms
    results = {}
    for name, func in algorithms.items():
        print(f"\nTesting {name}...")
        errors = []
        exact_matches = 0
        close_matches = 0
        
        for _, row in df.iterrows():
            days = int(row['trip_duration_days'])
            miles = int(row['miles_traveled'])
            receipts = float(row['total_receipts_amount'])
            expected = float(row['reimbursement_amount'])
            
            try:
                predicted = func(days, miles, receipts)
                error = abs(predicted - expected)
                errors.append(error)
                
                if error <= 0.01:
                    exact_matches += 1
                elif error <= 1.00:
                    close_matches += 1
            except Exception as e:
                errors.append(1000)  # Penalty for errors
        
        avg_error = np.mean(errors)
        max_error = np.max(errors)
        
        results[name] = {
            'avg_error': avg_error,
            'max_error': max_error,
            'exact_matches': exact_matches,
            'close_matches': close_matches,
            'accuracy': (exact_matches + close_matches) / len(df) * 100
        }
        
        print(f"  Average Error: ${avg_error:.2f}")
        print(f"  Max Error: ${max_error:.2f}")
        print(f"  Exact Matches: {exact_matches}/{len(df)} ({exact_matches/len(df)*100:.1f}%)")
        print(f"  Close Matches: {close_matches}/{len(df)} ({close_matches/len(df)*100:.1f}%)")
    
    # Find best algorithm
    best_name = min(results.keys(), key=lambda k: results[k]['avg_error'])
    best_func = algorithms[best_name]
    
    print(f"\nBest Algorithm: {best_name}")
    print(f"Performance: ${results[best_name]['avg_error']:.2f} average error")
    
    return best_func, best_name, results

def test_sample_cases(algorithm, name):
    """Test the best algorithm on sample cases"""
    
    print(f"\n=== {name.upper()} - SAMPLE CALCULATIONS ===")
    
    # Load a few sample cases from the actual data
    with open('public_cases.json', 'r') as f:
        data = json.load(f)
    
    # Test first 10 cases
    for i in range(min(10, len(data))):
        case = data[i]
        days = case['trip_duration_days']
        miles = case['miles_traveled']
        receipts = case['total_receipts_amount']
        expected = case['reimbursement_amount']
        
        predicted = algorithm(days, miles, receipts)
        error = abs(predicted - expected)
        accuracy = (1 - error/expected) * 100 if expected > 0 else 0
        
        print(f"Case {i+1}: {days}d, {miles}mi, ${receipts:.2f}")
        print(f"  Predicted: ${predicted:.2f}, Expected: ${expected:.2f}")
        print(f"  Error: ${error:.2f} ({accuracy:.1f}% accurate)")

def save_final_algorithm(algorithm, name, results):
    """Save the final algorithm and results"""
    
    # Create a summary
    summary = {
        "algorithm_name": name,
        "performance": results[name],
        "all_results": results,
        "algorithm_code": f"""
def calculate_reimbursement(trip_duration_days, miles_traveled, total_receipts_amount):
    '''
    Final reverse-engineered travel reimbursement algorithm
    Based on analysis of authentic public_cases.json data
    
    Performance: {results[name]['avg_error']:.2f} average error
    '''
    # Implementation varies based on best performing algorithm
    # See complete_analysis.py for full implementation
    pass
"""
    }
    
    with open('final_algorithm_results.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nResults saved to final_algorithm_results.json")
    print(f"Best algorithm: {name} with ${results[name]['avg_error']:.2f} average error")

def main():
    """Main analysis function"""
    
    print("=== AUTHENTIC DATA REVERSE ENGINEERING ===")
    
    # Load authentic data
    df = load_authentic_data()
    if df is None:
        return
    
    # Comprehensive pattern analysis
    models = comprehensive_pattern_analysis(df)
    
    # Build and test optimized algorithms
    best_algorithm, best_name, results = build_optimized_algorithms(df, models)
    
    # Test sample cases
    test_sample_cases(best_algorithm, best_name)
    
    # Save results
    save_final_algorithm(best_algorithm, best_name, results)
    
    print(f"\n=== ANALYSIS COMPLETE ===")
    print(f"Analyzed {len(df)} authentic cases")
    print(f"Best algorithm: {best_name}")
    print(f"Performance: ${results[best_name]['avg_error']:.2f} average error")

if __name__ == "__main__":
    main()