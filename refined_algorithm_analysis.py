#!/usr/bin/env python3
"""
Refined analysis of the authentic challenge data to improve algorithm accuracy
"""

import json
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

def analyze_data_segments():
    """Detailed segmented analysis"""
    
    with open('public_cases.json', 'r') as f:
        data = json.load(f)
    
    # Convert to flat structure
    df_data = []
    for case in data:
        df_data.append({
            'days': case['input']['trip_duration_days'],
            'miles': case['input']['miles_traveled'],
            'receipts': case['input']['total_receipts_amount'],
            'output': case['expected_output']
        })
    
    df = pd.DataFrame(df_data)
    
    print("=== SEGMENTED ANALYSIS ===")
    
    # Analyze each trip duration separately
    for days in sorted(df['days'].unique()):
        subset = df[df['days'] == days]
        if len(subset) >= 5:
            print(f"\n{days}-day trips ({len(subset)} cases):")
            print(f"  Range: ${subset['output'].min():.2f} - ${subset['output'].max():.2f}")
            print(f"  Mean: ${subset['output'].mean():.2f}")
            
            # Try linear regression for this segment
            if len(subset) >= 10:
                X = subset[['miles', 'receipts']]
                y = subset['output']
                
                lr = LinearRegression()
                lr.fit(X, y)
                r2 = lr.score(X, y)
                
                print(f"  Linear R²: {r2:.4f}")
                print(f"  Miles coef: {lr.coef_[0]:.4f}")
                print(f"  Receipts coef: {lr.coef_[1]:.4f}")
                print(f"  Intercept: {lr.intercept_:.2f}")
    
    return df

def find_optimal_algorithm(df):
    """Find the best performing algorithm structure"""
    
    print("\n=== TESTING REFINED ALGORITHMS ===")
    
    def algorithm_v1(days, miles, receipts):
        """Refined single/multi-day segmentation"""
        if days == 1:
            # Single day - much lower base than before
            return round(90.0 + miles * 1.2 + receipts * 1.5, 2)
        else:
            # Multi-day
            return round(150.0 + days * 60.0 + miles * 0.5 + receipts * 0.4, 2)
    
    def algorithm_v2(days, miles, receipts):
        """Linear regression based per segment"""
        if days == 1:
            return round(89.32 + miles * 0.6959 + receipts * 1.0727, 2)
        elif days == 2:
            return round(160.45 + miles * 0.4892 + receipts * 0.4205, 2)
        elif days == 3:
            return round(234.89 + miles * 0.4456 + receipts * 0.3829, 2)
        else:
            return round(266.71 + days * 50.05 + miles * 0.4456 + receipts * 0.3829, 2)
    
    def algorithm_v3(days, miles, receipts):
        """Distance-based segmentation"""
        if days == 1:
            if miles <= 100:
                return round(85.0 + miles * 0.8 + receipts * 1.2, 2)
            else:
                return round(100.0 + miles * 0.65 + receipts * 1.0, 2)
        else:
            if days <= 3:
                return round(140.0 + days * 70.0 + miles * 0.48 + receipts * 0.42, 2)
            else:
                return round(200.0 + days * 55.0 + miles * 0.45 + receipts * 0.38, 2)
    
    def algorithm_v4(days, miles, receipts):
        """Random Forest approximation"""
        # Simplified version of complex RF decision boundaries
        if days == 1:
            base = 75.0
            if receipts > 50:
                base += 40.0
            if miles > 100:
                base += 25.0
            return round(base + miles * 0.75 + receipts * 1.1, 2)
        else:
            base = 120.0 + days * 65.0
            mile_rate = 0.5 if miles <= 200 else 0.45
            receipt_rate = 0.45 if receipts <= 500 else 0.38
            return round(base + miles * mile_rate + receipts * receipt_rate, 2)
    
    algorithms = {
        "Refined Segmented": algorithm_v1,
        "Linear Per Segment": algorithm_v2,
        "Distance Segmented": algorithm_v3,
        "RF Approximation": algorithm_v4
    }
    
    best_name = None
    best_error = float('inf')
    
    for name, algo in algorithms.items():
        errors = []
        
        for _, row in df.iterrows():
            try:
                predicted = algo(row['days'], row['miles'], row['receipts'])
                error = abs(predicted - row['output'])
                errors.append(error)
            except (ValueError, TypeError, ZeroDivisionError):
                errors.append(1000)
        
        avg_error = np.mean(errors)
        max_error = np.max(errors)
        exact = sum(1 for e in errors if e <= 0.01)
        close = sum(1 for e in errors if e <= 1.0)
        
        print(f"\n{name}:")
        print(f"  Avg Error: ${avg_error:.2f}")
        print(f"  Max Error: ${max_error:.2f}")
        print(f"  Exact: {exact}/{len(df)} ({exact/len(df)*100:.1f}%)")
        print(f"  Close: {close}/{len(df)} ({close/len(df)*100:.1f}%)")
        
        if avg_error < best_error:
            best_error = avg_error
            best_name = name
            best_algo = algo
    
    print(f"\nBest: {best_name} (${best_error:.2f} avg error)")
    return best_algo, best_name

def create_optimized_run_script(algorithm, name):
    """Create the final optimized run script"""
    
    if "Linear Per Segment" in name:
        script_content = '''#!/usr/bin/env python3
import sys

def calculate_reimbursement(trip_duration_days, miles_traveled, total_receipts_amount):
    """Optimized algorithm based on segmented linear regression analysis"""
    days = int(trip_duration_days)
    miles = float(miles_traveled)
    receipts = float(total_receipts_amount)
    
    if days == 1:
        return round(89.32 + miles * 0.6959 + receipts * 1.0727, 2)
    elif days == 2:
        return round(160.45 + miles * 0.4892 + receipts * 0.4205, 2)
    elif days == 3:
        return round(234.89 + miles * 0.4456 + receipts * 0.3829, 2)
    else:
        return round(266.71 + days * 50.05 + miles * 0.4456 + receipts * 0.3829, 2)

if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit(1)
    try:
        result = calculate_reimbursement(sys.argv[1], sys.argv[2], sys.argv[3])
        print(f"{result:.2f}")
    except (ValueError, TypeError, IndexError):
        sys.exit(1)
'''
    elif "Distance Segmented" in name:
        script_content = '''#!/usr/bin/env python3
import sys

def calculate_reimbursement(trip_duration_days, miles_traveled, total_receipts_amount):
    """Distance and duration segmented algorithm"""
    days = int(trip_duration_days)
    miles = float(miles_traveled)
    receipts = float(total_receipts_amount)
    
    if days == 1:
        if miles <= 100:
            return round(85.0 + miles * 0.8 + receipts * 1.2, 2)
        else:
            return round(100.0 + miles * 0.65 + receipts * 1.0, 2)
    else:
        if days <= 3:
            return round(140.0 + days * 70.0 + miles * 0.48 + receipts * 0.42, 2)
        else:
            return round(200.0 + days * 55.0 + miles * 0.45 + receipts * 0.38, 2)

if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit(1)
    try:
        result = calculate_reimbursement(sys.argv[1], sys.argv[2], sys.argv[3])
        print(f"{result:.2f}")
    except (ValueError, TypeError, IndexError):
        sys.exit(1)
'''
    else:
        script_content = '''#!/usr/bin/env python3
import sys

def calculate_reimbursement(trip_duration_days, miles_traveled, total_receipts_amount):
    """Refined segmented algorithm"""
    days = int(trip_duration_days)
    miles = float(miles_traveled)
    receipts = float(total_receipts_amount)
    
    if days == 1:
        return round(90.0 + miles * 1.2 + receipts * 1.5, 2)
    else:
        return round(150.0 + days * 60.0 + miles * 0.5 + receipts * 0.4, 2)

if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit(1)
    try:
        result = calculate_reimbursement(sys.argv[1], sys.argv[2], sys.argv[3])
        print(f"{result:.2f}")
    except (ValueError, TypeError, IndexError):
        sys.exit(1)
'''
    
    with open('run.py', 'w') as f:
        f.write(script_content)
    
    print(f"Updated run.py with {name} algorithm")

def main():
    df = analyze_data_segments()
    best_algo, best_name = find_optimal_algorithm(df)
    create_optimized_run_script(best_algo, best_name)
    
    # Test the updated algorithm
    print(f"\n=== TESTING UPDATED ALGORITHM ===")
    test_cases = [
        (3, 93, 1.42, 364.51),
        (1, 55, 3.6, 126.06),
        (1, 47, 17.97, 128.91),
        (2, 13, 4.67, 203.52),
        (3, 88, 5.78, 380.37)
    ]
    
    for days, miles, receipts, expected in test_cases:
        predicted = best_algo(days, miles, receipts)
        error = abs(predicted - expected)
        print(f"{days}d, {miles}mi, ${receipts:.2f} -> ${predicted:.2f} (exp ${expected:.2f}, err ${error:.2f})")

if __name__ == "__main__":
    main()