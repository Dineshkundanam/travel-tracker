#!/usr/bin/env python3
"""
Deep residual analysis to find the missing patterns in the reimbursement formula
"""

import json
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

def load_data():
    with open('public_cases.json', 'r') as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    df_clean = df[['trip_duration_days', 'miles_traveled', 'total_receipts_amount', 'reimbursement_amount']].copy()
    return df_clean.dropna()

def analyze_residuals():
    df = load_data()
    
    # Fit the best linear model
    X = df[['trip_duration_days', 'miles_traveled', 'total_receipts_amount']]
    y = df['reimbursement_amount']
    
    lr = LinearRegression()
    lr.fit(X, y)
    
    # Calculate predictions and residuals
    predicted = lr.predict(X)
    residuals = y - predicted
    
    # Add residuals to dataframe for analysis
    df['predicted'] = predicted
    df['residuals'] = residuals
    df['abs_residuals'] = np.abs(residuals)
    
    print("=== RESIDUAL ANALYSIS ===")
    print(f"Mean residual: ${residuals.mean():.2f}")
    print(f"Std residual: ${residuals.std():.2f}")
    print(f"Max positive residual: ${residuals.max():.2f}")
    print(f"Max negative residual: ${residuals.min():.2f}")
    
    # Look for patterns in residuals
    print("\n=== RESIDUAL PATTERNS BY TRIP DURATION ===")
    for days in sorted(df['trip_duration_days'].unique()):
        day_data = df[df['trip_duration_days'] == days]
        if len(day_data) >= 3:
            mean_residual = day_data['residuals'].mean()
            print(f"{days} days: {len(day_data)} cases, avg residual: ${mean_residual:.2f}")
    
    print("\n=== RESIDUAL PATTERNS BY MILES TRAVELED ===")
    mile_bins = [0, 50, 100, 200, 300, 500, 1000, 2000]
    df['mile_bins'] = pd.cut(df['miles_traveled'], bins=mile_bins, include_lowest=True)
    mile_residuals = df.groupby('mile_bins')['residuals'].agg(['count', 'mean', 'std'])
    print(mile_residuals)
    
    print("\n=== LARGE RESIDUAL CASES ===")
    large_residuals = df[df['abs_residuals'] > 20].sort_values('abs_residuals', ascending=False)
    for _, row in large_residuals.head(10).iterrows():
        print(f"Days: {row['trip_duration_days']}, Miles: {row['miles_traveled']}, Receipts: ${row['total_receipts_amount']:.2f}")
        print(f"  Expected: ${row['reimbursement_amount']:.2f}, Predicted: ${row['predicted']:.2f}, Residual: ${row['residuals']:.2f}")
    
    # Look for multiplicative patterns in the residuals
    print("\n=== TESTING MULTIPLICATIVE CORRECTIONS ===")
    
    # Test if residuals correlate with specific combinations
    df['days_squared'] = df['trip_duration_days'] ** 2
    df['miles_squared'] = df['miles_traveled'] ** 2
    df['receipts_squared'] = df['total_receipts_amount'] ** 2
    df['days_miles'] = df['trip_duration_days'] * df['miles_traveled']
    df['days_receipts'] = df['trip_duration_days'] * df['total_receipts_amount']
    df['miles_receipts'] = df['miles_traveled'] * df['total_receipts_amount']
    
    # Check correlations with residuals
    correlations = {}
    for col in ['days_squared', 'miles_squared', 'receipts_squared', 'days_miles', 'days_receipts', 'miles_receipts']:
        corr = df[col].corr(df['residuals'])
        correlations[col] = corr
        print(f"Correlation between {col} and residuals: {corr:.4f}")
    
    # Test weekend/weekday pattern (if trip duration suggests weekend travel)
    print("\n=== WEEKEND TRAVEL ANALYSIS ===")
    # Trips that likely include weekends (>= 3 days or exactly 2 days)
    df['likely_weekend'] = ((df['trip_duration_days'] >= 3) | 
                           (df['trip_duration_days'] == 2)).astype(int)
    
    weekend_residuals = df.groupby('likely_weekend')['residuals'].agg(['count', 'mean', 'std'])
    print("Residuals by weekend likelihood:")
    print(weekend_residuals)
    
    # Test if there are threshold effects we missed
    print("\n=== THRESHOLD EFFECT ANALYSIS ===")
    
    # Test common business thresholds
    thresholds_to_test = {
        'trip_duration_days': [1, 2, 3, 5, 7, 10, 14],
        'miles_traveled': [50, 100, 200, 300, 500, 1000],
        'total_receipts_amount': [25, 50, 100, 150, 200, 300]
    }
    
    for col, thresholds in thresholds_to_test.items():
        print(f"\nThreshold analysis for {col}:")
        for threshold in thresholds:
            below = df[df[col] <= threshold]
            above = df[df[col] > threshold]
            
            if len(below) >= 10 and len(above) >= 10:
                below_mean = below['residuals'].mean()
                above_mean = above['residuals'].mean()
                diff = above_mean - below_mean
                
                if abs(diff) > 5:  # Only show significant differences
                    print(f"  {threshold}: Below avg residual: ${below_mean:.2f}, Above: ${above_mean:.2f}, Diff: ${diff:.2f}")
    
    return df

def build_improved_algorithm(df):
    """Build an improved algorithm based on residual analysis"""
    
    print("\n=== BUILDING IMPROVED ALGORITHM ===")
    
    # Base linear model coefficients
    X = df[['trip_duration_days', 'miles_traveled', 'total_receipts_amount']]
    y = df['reimbursement_amount']
    lr = LinearRegression()
    lr.fit(X, y)
    
    # Try to model the residuals with additional features
    residuals = y - lr.predict(X)
    
    # Create additional features based on our analysis
    X_extended = df[['trip_duration_days', 'miles_traveled', 'total_receipts_amount']].copy()
    
    # Add quadratic terms
    X_extended['days_squared'] = df['trip_duration_days'] ** 2
    X_extended['miles_squared'] = df['miles_traveled'] ** 2
    
    # Add interaction terms
    X_extended['days_miles'] = df['trip_duration_days'] * df['miles_traveled']
    X_extended['days_receipts'] = df['trip_duration_days'] * df['total_receipts_amount']
    
    # Add threshold indicators
    X_extended['long_trip'] = (df['trip_duration_days'] >= 7).astype(int)
    X_extended['medium_trip'] = ((df['trip_duration_days'] >= 3) & (df['trip_duration_days'] < 7)).astype(int)
    X_extended['high_mileage'] = (df['miles_traveled'] >= 500).astype(int)
    X_extended['low_mileage'] = (df['miles_traveled'] <= 100).astype(int)
    X_extended['high_receipts'] = (df['total_receipts_amount'] >= 200).astype(int)
    
    # Fit extended model
    lr_extended = LinearRegression()
    lr_extended.fit(X_extended, y)
    
    print(f"Original R²: {lr.score(X, y):.6f}")
    print(f"Extended R²: {lr_extended.score(X_extended, y):.6f}")
    
    # Show coefficients
    print("\nExtended model coefficients:")
    for i, col in enumerate(X_extended.columns):
        print(f"  {col}: {lr_extended.coef_[i]:.6f}")
    print(f"  Intercept: {lr_extended.intercept_:.6f}")
    
    # Test the extended model
    predicted_extended = lr_extended.predict(X_extended)
    errors_extended = np.abs(y - predicted_extended)
    
    print(f"\nExtended model performance:")
    print(f"  Average error: ${errors_extended.mean():.2f}")
    print(f"  Max error: ${errors_extended.max():.2f}")
    print(f"  Cases within $0.01: {sum(errors_extended <= 0.01)}")
    print(f"  Cases within $1.00: {sum(errors_extended <= 1.00)}")
    
    return lr_extended, X_extended.columns

def create_final_algorithm(lr_extended, feature_names):
    """Create the final algorithm function"""
    
    def calculate_reimbursement(trip_duration_days, miles_traveled, total_receipts_amount):
        # Create feature vector
        features = np.array([
            trip_duration_days,
            miles_traveled,
            total_receipts_amount,
            trip_duration_days ** 2,
            miles_traveled ** 2,
            trip_duration_days * miles_traveled,
            trip_duration_days * total_receipts_amount,
            1 if trip_duration_days >= 7 else 0,
            1 if 3 <= trip_duration_days < 7 else 0,
            1 if miles_traveled >= 500 else 0,
            1 if miles_traveled <= 100 else 0,
            1 if total_receipts_amount >= 200 else 0
        ]).reshape(1, -1)
        
        prediction = lr_extended.predict(features)[0]
        return round(prediction, 2)
    
    return calculate_reimbursement

def main():
    df = analyze_residuals()
    lr_extended, feature_names = build_improved_algorithm(df)
    final_algorithm = create_final_algorithm(lr_extended, feature_names)
    
    # Test final algorithm
    print("\n=== TESTING FINAL ALGORITHM ===")
    test_cases = [
        (3, 150, 85.50, 289.15),
        (1, 50, 45.25, 141.19),
        (7, 450, 200.00, 777.50),
        (5, 300, 125.75, 501.40),
        (2, 100, 65.80, 223.47)
    ]
    
    for days, miles, receipts, expected in test_cases:
        predicted = final_algorithm(days, miles, receipts)
        error = abs(predicted - expected)
        print(f"  {days}d, {miles}mi, ${receipts:.2f} → Predicted: ${predicted:.2f}, Expected: ${expected:.2f}, Error: ${error:.2f}")

if __name__ == "__main__":
    main()