#!/usr/bin/env python3
"""
Final optimized reimbursement algorithm based on comprehensive analysis
"""

import json
import pandas as pd
import numpy as np

def calculate_reimbursement_v1(trip_duration_days, miles_traveled, total_receipts_amount):
    """
    Version 1: Enhanced Linear Model with Threshold Adjustments
    Based on residual analysis showing clear threshold effects
    """
    # Base linear components (from regression analysis)
    base = 21.45  # Base amount
    daily_rate = 2.87 * trip_duration_days
    mileage_rate = 1.50 * miles_traveled  
    receipt_rate = 0.47 * total_receipts_amount
    
    # Quadratic adjustments
    days_squared_penalty = -13.19 * (trip_duration_days ** 2)
    miles_squared_penalty = -0.0035 * (miles_traveled ** 2)
    
    # Interaction effects
    days_miles_bonus = 0.38 * trip_duration_days * miles_traveled
    days_receipts_bonus = 0.085 * trip_duration_days * total_receipts_amount
    
    # Threshold bonuses/penalties
    threshold_adjustments = 0
    
    # Trip duration thresholds
    if trip_duration_days >= 7:
        threshold_adjustments += 2.64  # Long trip bonus
    elif 3 <= trip_duration_days < 7:
        threshold_adjustments -= 21.34  # Medium trip penalty
    
    # Mileage thresholds
    if miles_traveled >= 500:
        threshold_adjustments += 8.11  # High mileage bonus
    elif miles_traveled <= 100:
        threshold_adjustments += 3.69  # Low mileage bonus
    
    # Receipt thresholds
    if total_receipts_amount >= 200:
        threshold_adjustments += 19.65  # High receipt bonus
    
    total = (base + daily_rate + mileage_rate + receipt_rate + 
             days_squared_penalty + miles_squared_penalty +
             days_miles_bonus + days_receipts_bonus + threshold_adjustments)
    
    return round(total, 2)

def calculate_reimbursement_v2(trip_duration_days, miles_traveled, total_receipts_amount):
    """
    Version 2: Business Rules Approach
    Based on interview insights and threshold analysis
    """
    # Daily allowance with trip length bonuses
    if trip_duration_days == 1:
        daily_allowance = 65.0  # Higher rate for single-day trips
    elif trip_duration_days <= 3:
        daily_allowance = 45.0 * trip_duration_days
    elif trip_duration_days <= 6:
        daily_allowance = 40.0 * trip_duration_days
    else:
        daily_allowance = 50.0 * trip_duration_days  # Long trip bonus
    
    # Tiered mileage rates
    if miles_traveled <= 100:
        mileage_allowance = miles_traveled * 0.75
    elif miles_traveled <= 300:
        mileage_allowance = 100 * 0.75 + (miles_traveled - 100) * 0.55
    elif miles_traveled <= 500:
        mileage_allowance = 100 * 0.75 + 200 * 0.55 + (miles_traveled - 300) * 0.65
    else:
        mileage_allowance = 100 * 0.75 + 200 * 0.55 + 200 * 0.65 + (miles_traveled - 500) * 0.70
    
    # Receipt reimbursement with progressive rates
    if total_receipts_amount <= 150:
        receipt_allowance = total_receipts_amount * 1.15
    else:
        receipt_allowance = 150 * 1.15 + (total_receipts_amount - 150) * 1.35
    
    total = daily_allowance + mileage_allowance + receipt_allowance
    return round(total, 2)

def calculate_reimbursement_v3(trip_duration_days, miles_traveled, total_receipts_amount):
    """
    Version 3: Hybrid Approach
    Combines linear model with business rule adjustments
    """
    # Start with the core linear relationship
    base_amount = 2.81 * trip_duration_days + 1.12 * miles_traveled + 1.095 * total_receipts_amount + 13.57
    
    # Apply threshold-based adjustments based on residual analysis
    adjustments = 0
    
    # Single-day trip bonus (from residual analysis)
    if trip_duration_days == 1:
        adjustments += 8.40
    elif trip_duration_days == 2:
        adjustments += 4.65
    elif trip_duration_days in [3, 4, 5]:
        adjustments -= 10.97  # Average penalty for 3-5 day trips
    elif trip_duration_days >= 7:
        adjustments += 20.0   # Long trip bonus
    
    # Mileage adjustments
    if miles_traveled <= 100:
        adjustments += 6.5    # Low mileage bonus
    elif 200 <= miles_traveled <= 300:
        adjustments -= 14.5   # Medium mileage penalty
    elif miles_traveled >= 500:
        adjustments += 8.0    # High mileage bonus
    
    # Receipt adjustments
    if total_receipts_amount >= 200:
        adjustments += 10.0   # High receipt bonus
    elif total_receipts_amount <= 50:
        adjustments += 5.0    # Low receipt bonus
    
    total = base_amount + adjustments
    return round(total, 2)

def test_all_algorithms():
    """Test all algorithm versions against the data"""
    # Load data
    with open('public_cases.json', 'r') as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    df = df[['trip_duration_days', 'miles_traveled', 'total_receipts_amount', 'reimbursement_amount']].dropna()
    
    algorithms = {
        "Enhanced Linear (v1)": calculate_reimbursement_v1,
        "Business Rules (v2)": calculate_reimbursement_v2,
        "Hybrid Model (v3)": calculate_reimbursement_v3
    }
    
    print("=== ALGORITHM COMPARISON ===\n")
    
    results = {}
    for name, func in algorithms.items():
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
                errors.append(1000)  # Large penalty for errors
        
        avg_error = np.mean(errors)
        max_error = np.max(errors)
        
        results[name] = {
            'avg_error': avg_error,
            'max_error': max_error,
            'exact_matches': exact_matches,
            'close_matches': close_matches,
            'accuracy': (exact_matches + close_matches) / len(df) * 100
        }
        
        print(f"{name}:")
        print(f"  Average Error: ${avg_error:.2f}")
        print(f"  Max Error: ${max_error:.2f}")
        print(f"  Exact Matches: {exact_matches}/{len(df)} ({exact_matches/len(df)*100:.1f}%)")
        print(f"  Close Matches: {close_matches}/{len(df)} ({close_matches/len(df)*100:.1f}%)")
        print(f"  Combined Accuracy: {results[name]['accuracy']:.1f}%")
        print()
    
    # Find best algorithm
    best_algo = min(results.keys(), key=lambda k: results[k]['avg_error'])
    print(f"🏆 Best Algorithm: {best_algo}")
    print(f"Average Error: ${results[best_algo]['avg_error']:.2f}")
    
    # Test on sample cases
    print(f"\n=== {best_algo.upper()} - SAMPLE CASES ===")
    best_func = algorithms[best_algo]
    
    test_cases = [
        (3, 150, 85.50, 289.15),
        (1, 50, 45.25, 141.19),
        (7, 450, 200.00, 777.50),
        (5, 300, 125.75, 501.40),
        (2, 100, 65.80, 223.47),
        (10, 650, 350.00, 1172.50),
        (1, 25, 30.00, 98.50)
    ]
    
    for days, miles, receipts, expected in test_cases:
        predicted = best_func(days, miles, receipts)
        error = abs(predicted - expected)
        accuracy = (1 - error/expected) * 100 if expected > 0 else 0
        print(f"  {days}d, {miles}mi, ${receipts:.2f} → ${predicted:.2f} (expected ${expected:.2f}) | Error: ${error:.2f} ({accuracy:.1f}% accurate)")
    
    return best_func, best_algo

if __name__ == "__main__":
    best_function, best_name = test_all_algorithms()
    
    print(f"\n=== FINAL ALGORITHM READY ===")
    print(f"Best performing algorithm: {best_name}")
    print("Ready for integration into the Streamlit interface.")