#!/usr/bin/env python3
"""
Quick analysis script to identify key patterns in the reimbursement data
"""

import json
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
# import matplotlib.pyplot as plt  # Not needed for this analysis

def load_and_analyze():
    # Load the data
    with open('public_cases.json', 'r') as f:
        data = json.load(f)
    
    df = pd.DataFrame(data)
    print("Dataset loaded:", len(df), "cases")
    print("\nColumns:", df.columns.tolist())
    print("\nData types:")
    print(df.dtypes)
    print("\nFirst few rows:")
    print(df.head())
    print("\nCheck for missing values:")
    print(df.isnull().sum())
    
    # Clean the data - only keep the relevant columns and remove problematic ones
    df_clean = df[['trip_duration_days', 'miles_traveled', 'total_receipts_amount', 'reimbursement_amount']].copy()
    df_clean = df_clean.dropna()
    print(f"\nAfter cleaning data: {len(df_clean)} cases")
    
    # Check for linear relationships
    print("\n=== LINEAR RELATIONSHIP ANALYSIS ===")
    X = df_clean[['trip_duration_days', 'miles_traveled', 'total_receipts_amount']]
    y = df_clean['reimbursement_amount']
    
    lr = LinearRegression()
    lr.fit(X, y)
    
    print(f"Linear regression R² score: {lr.score(X, y):.4f}")
    print(f"Coefficients:")
    print(f"  - Trip duration: ${lr.coef_[0]:.2f} per day")
    print(f"  - Miles traveled: ${lr.coef_[1]:.4f} per mile") 
    print(f"  - Receipts: {lr.coef_[2]:.4f}x multiplier")
    print(f"  - Intercept: ${lr.intercept_:.2f}")
    
    # Test simple formula hypothesis from interviews
    print("\n=== TESTING INTERVIEW HYPOTHESES ===")
    
    # Hypothesis 1: $45/day + $0.47/mile + 1.18x receipts
    predicted_h1 = 45 * df_clean['trip_duration_days'] + 0.47 * df_clean['miles_traveled'] + 1.18 * df_clean['total_receipts_amount']
    error_h1 = np.mean(np.abs(predicted_h1 - df_clean['reimbursement_amount']))
    print(f"Hypothesis 1 (45/day + 0.47/mile + 1.18x receipts): Avg error ${error_h1:.2f}")
    
    # Hypothesis 2: Based on linear regression coefficients
    predicted_h2 = lr.predict(X)
    error_h2 = np.mean(np.abs(predicted_h2 - df_clean['reimbursement_amount']))
    print(f"Hypothesis 2 (linear regression): Avg error ${error_h2:.2f}")
    
    # Look for threshold effects
    print("\n=== THRESHOLD ANALYSIS ===")
    
    # Check if there are different patterns for different trip lengths
    for threshold in [3, 5, 7, 10]:
        short_trips = df_clean[df_clean['trip_duration_days'] <= threshold]
        long_trips = df_clean[df_clean['trip_duration_days'] > threshold]
        
        if len(short_trips) > 10 and len(long_trips) > 10:
            short_avg_per_day = (short_trips['reimbursement_amount'] / short_trips['trip_duration_days']).mean()
            long_avg_per_day = (long_trips['reimbursement_amount'] / long_trips['trip_duration_days']).mean()
            
            print(f"Trips ≤{threshold} days: ${short_avg_per_day:.2f}/day avg")
            print(f"Trips >{threshold} days: ${long_avg_per_day:.2f}/day avg")
            print(f"Difference: ${long_avg_per_day - short_avg_per_day:.2f}/day")
            print()
    
    # Check mileage patterns
    print("=== MILEAGE PATTERN ANALYSIS ===")
    for threshold in [100, 300, 500]:
        short_miles = df_clean[df_clean['miles_traveled'] <= threshold]
        long_miles = df_clean[df_clean['miles_traveled'] > threshold]
        
        if len(short_miles) > 10 and len(long_miles) > 10:
            short_rate = (short_miles['reimbursement_amount'] / short_miles['miles_traveled']).mean()
            long_rate = (long_miles['reimbursement_amount'] / long_miles['miles_traveled']).mean()
            
            print(f"Trips ≤{threshold} miles: ${short_rate:.4f}/mile avg")
            print(f"Trips >{threshold} miles: ${long_rate:.4f}/mile avg")
            print()
    
    # Analyze receipt patterns
    print("=== RECEIPT MULTIPLIER ANALYSIS ===")
    receipt_ratios = df_clean['reimbursement_amount'] / df_clean['total_receipts_amount']
    print(f"Receipt multiplier stats:")
    print(f"  Mean: {receipt_ratios.mean():.3f}")
    print(f"  Median: {receipt_ratios.median():.3f}")
    print(f"  Std: {receipt_ratios.std():.3f}")
    
    # Look at a few specific examples
    print("\n=== SAMPLE CASES ANALYSIS ===")
    sample_cases = df_clean.head(10)
    for _, row in sample_cases.iterrows():
        days = row['trip_duration_days']
        miles = row['miles_traveled'] 
        receipts = row['total_receipts_amount']
        reimb = row['reimbursement_amount']
        
        # Try to decompose
        daily_portion = reimb / days
        mile_portion = reimb / miles if miles > 0 else 0
        receipt_ratio = reimb / receipts if receipts > 0 else 0
        
        print(f"{days}d, {miles}mi, ${receipts:.2f} → ${reimb:.2f}")
        print(f"  ${daily_portion:.2f}/day, ${mile_portion:.4f}/mile, {receipt_ratio:.2f}x receipts")
    
    # Decision tree to find rules
    print("\n=== DECISION TREE RULES ===")
    dt = DecisionTreeRegressor(max_depth=4, min_samples_split=20)
    dt.fit(X, y)
    print(f"Decision tree R² score: {dt.score(X, y):.4f}")
    
    # Export key findings
    findings = {
        'linear_regression': {
            'r2_score': lr.score(X, y),
            'daily_rate': lr.coef_[0],
            'mile_rate': lr.coef_[1],
            'receipt_multiplier': lr.coef_[2],
            'intercept': lr.intercept_
        },
        'receipt_analysis': {
            'mean_multiplier': receipt_ratios.mean(),
            'median_multiplier': receipt_ratios.median()
        }
    }
    
    with open('analysis_findings.json', 'w') as f:
        json.dump(findings, f, indent=2)
    
    print(f"\nAnalysis complete. Key findings saved to analysis_findings.json")

if __name__ == "__main__":
    load_and_analyze()