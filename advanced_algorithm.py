#!/usr/bin/env python3
"""
Advanced algorithm testing based on discovered patterns
"""

import json
import pandas as pd
import numpy as np

def load_data():
    with open('public_cases.json', 'r') as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    df_clean = df[['trip_duration_days', 'miles_traveled', 'total_receipts_amount', 'reimbursement_amount']].copy()
    return df_clean.dropna()

def test_algorithm(algorithm_func, df):
    """Test an algorithm against the data"""
    errors = []
    exact_matches = 0
    close_matches = 0
    
    for _, row in df.iterrows():
        days = int(row['trip_duration_days'])
        miles = int(row['miles_traveled'])
        receipts = float(row['total_receipts_amount'])
        expected = float(row['reimbursement_amount'])
        
        predicted = algorithm_func(days, miles, receipts)
        error = abs(predicted - expected)
        errors.append(error)
        
        if error <= 0.01:
            exact_matches += 1
        elif error <= 1.00:
            close_matches += 1
    
    return {
        'avg_error': np.mean(errors),
        'max_error': np.max(errors),
        'exact_matches': exact_matches,
        'close_matches': close_matches,
        'total_cases': len(df),
        'exact_rate': exact_matches / len(df) * 100,
        'close_rate': close_matches / len(df) * 100
    }

# Algorithm 1: Simple Linear (baseline)
def algorithm_linear(trip_duration_days, miles_traveled, total_receipts_amount):
    return round(2.81 * trip_duration_days + 1.12 * miles_traveled + 1.095 * total_receipts_amount + 13.57, 2)

# Algorithm 2: Tiered approach based on findings
def algorithm_tiered(trip_duration_days, miles_traveled, total_receipts_amount):
    # Base daily allowance with trip length bonus
    if trip_duration_days <= 3:
        daily_rate = 45.0
    else:
        daily_rate = 55.0  # Bonus for longer trips
    
    daily_allowance = daily_rate * trip_duration_days
    
    # Tiered mileage rates
    if miles_traveled <= 100:
        mileage_rate = 0.75
    elif miles_traveled <= 500:
        mileage_rate = 0.55
    else:
        mileage_rate = 0.58
    
    mileage_allowance = mileage_rate * miles_traveled
    
    # Receipt reimbursement - the big mystery
    # Data shows ~3.75x multiplier, but that seems too high
    # Let's try a more complex approach
    receipt_allowance = total_receipts_amount * 1.2
    
    total = daily_allowance + mileage_allowance + receipt_allowance
    return round(total, 2)

# Algorithm 3: Pattern-based (trying to match the 3.75x receipt pattern)
def algorithm_pattern_based(trip_duration_days, miles_traveled, total_receipts_amount):
    # Maybe the "receipt multiplier" includes other costs?
    # Base calculation
    base = 25.0 * trip_duration_days  # Lower base rate
    mileage = 0.45 * miles_traveled   # Standard mileage
    
    # Receipt handling - maybe it's reimbursement + per diem for meals?
    # If receipts represent only some expenses, the multiplier covers the rest
    receipt_component = total_receipts_amount * 2.8  # Higher multiplier
    
    total = base + mileage + receipt_component
    return round(total, 2)

# Algorithm 4: Complex business rules
def algorithm_complex(trip_duration_days, miles_traveled, total_receipts_amount):
    # Start with base components
    daily_base = 40.0
    
    # Trip length bonus
    if trip_duration_days >= 7:
        daily_base = 50.0
    elif trip_duration_days >= 5:
        daily_base = 45.0
    
    daily_allowance = daily_base * trip_duration_days
    
    # Mileage with distance-based rates
    if miles_traveled <= 50:
        mileage_allowance = miles_traveled * 1.2  # High rate for very short trips
    elif miles_traveled <= 200:
        mileage_allowance = 50 * 1.2 + (miles_traveled - 50) * 0.6
    elif miles_traveled <= 500:
        mileage_allowance = 50 * 1.2 + 150 * 0.6 + (miles_traveled - 200) * 0.5
    else:
        mileage_allowance = 50 * 1.2 + 150 * 0.6 + 300 * 0.5 + (miles_traveled - 500) * 0.55
    
    # Receipt handling - progressive rates
    if total_receipts_amount <= 50:
        receipt_allowance = total_receipts_amount * 1.8
    elif total_receipts_amount <= 150:
        receipt_allowance = 50 * 1.8 + (total_receipts_amount - 50) * 1.4
    else:
        receipt_allowance = 50 * 1.8 + 100 * 1.4 + (total_receipts_amount - 150) * 1.2
    
    total = daily_allowance + mileage_allowance + receipt_allowance
    
    # Minimum guarantee
    minimum = trip_duration_days * 65.0
    total = max(total, minimum)
    
    return round(total, 2)

# Algorithm 5: Data-driven reverse engineering
def algorithm_reverse_engineered(trip_duration_days, miles_traveled, total_receipts_amount):
    """
    Based on the insight that receipt_multiplier averages 3.75x,
    maybe the formula is simpler than we think
    """
    # What if it's just: base_per_trip + miles_rate + receipt_factor?
    
    # Fixed base per trip (regardless of length initially)
    base = 50.0 + (trip_duration_days - 1) * 35.0  # $50 first day, $35 each additional
    
    # Simple mileage
    mileage = miles_traveled * 0.52
    
    # The receipt "multiplier" might actually be: reimbursement + daily meal allowance
    # If receipts are ~$100 and reimbursement is ~$375, that's like $100 + $275 meal allowance
    receipt_reimb = total_receipts_amount  # 100% reimbursement
    meal_allowance = trip_duration_days * 42.0  # $42/day meal allowance
    
    total = base + mileage + receipt_reimb + meal_allowance
    return round(total, 2)

def main():
    df = load_data()
    print(f"Testing algorithms on {len(df)} cases\n")
    
    algorithms = {
        "Linear Regression": algorithm_linear,
        "Tiered Rates": algorithm_tiered,
        "Pattern-Based": algorithm_pattern_based,
        "Complex Rules": algorithm_complex,
        "Reverse Engineered": algorithm_reverse_engineered
    }
    
    results = {}
    for name, func in algorithms.items():
        print(f"Testing {name}...")
        result = test_algorithm(func, df)
        results[name] = result
        
        print(f"  Average Error: ${result['avg_error']:.2f}")
        print(f"  Max Error: ${result['max_error']:.2f}")
        print(f"  Exact Matches: {result['exact_matches']}/{result['total_cases']} ({result['exact_rate']:.1f}%)")
        print(f"  Close Matches: {result['close_matches']}/{result['total_cases']} ({result['close_rate']:.1f}%)")
        print()
    
    # Find best algorithm
    best_algo = min(results.keys(), key=lambda k: results[k]['avg_error'])
    print(f"Best Algorithm: {best_algo}")
    print(f"Average Error: ${results[best_algo]['avg_error']:.2f}")
    
    # Test a few specific cases with the best algorithm
    print("\nTesting best algorithm on sample cases:")
    test_cases = [
        (3, 150, 85.50, 289.15),
        (1, 50, 45.25, 141.19),
        (7, 450, 200.00, 777.50),
        (5, 300, 125.75, 501.40)
    ]
    
    best_func = algorithms[best_algo]
    for days, miles, receipts, expected in test_cases:
        predicted = best_func(days, miles, receipts)
        error = abs(predicted - expected)
        print(f"  {days}d, {miles}mi, ${receipts:.2f} → Predicted: ${predicted:.2f}, Expected: ${expected:.2f}, Error: ${error:.2f}")

if __name__ == "__main__":
    main()