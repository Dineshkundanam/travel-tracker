#!/usr/bin/env python3
"""
Python-based evaluation script for the reimbursement challenge
Alternative to eval.sh that doesn't require jq and bc dependencies
"""

import json
import subprocess
import sys
import statistics
from typing import List, Tuple

def load_test_cases(filename: str = 'public_cases.json') -> List[Tuple[int, float, float, float]]:
    """Load test cases from JSON file"""
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
        
        test_cases = []
        for case in data:
            days = case['input']['trip_duration_days']
            miles = case['input']['miles_traveled']
            receipts = case['input']['total_receipts_amount']
            expected = case['expected_output']
            test_cases.append((days, miles, receipts, expected))
        
        return test_cases
    except FileNotFoundError:
        print(f"❌ Error: {filename} not found!")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"❌ Error: Invalid JSON in {filename}")
        sys.exit(1)

def run_algorithm(days: int, miles: float, receipts: float) -> float:
    """Run the algorithm via run.py script"""
    try:
        result = subprocess.run(
            ['python3', 'run.py', str(days), str(miles), str(receipts)],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode != 0:
            return None
        
        return float(result.stdout.strip())
    except (subprocess.TimeoutExpired, ValueError, FileNotFoundError):
        return None

def evaluate_algorithm():
    """Main evaluation function"""
    print("🧾 Python Challenge Evaluation")
    print("=" * 40)
    print()
    
    # Load test cases
    print("📊 Loading test cases...")
    test_cases = load_test_cases()
    print(f"Loaded {len(test_cases)} test cases")
    print()
    
    # Initialize tracking variables
    successful_runs = 0
    exact_matches = 0
    close_matches = 0
    errors = []
    max_error = 0
    max_error_case = None
    worst_cases = []
    best_cases = []
    
    print("🔍 Running evaluation...")
    
    # Process each test case
    for i, (days, miles, receipts, expected) in enumerate(test_cases):
        if i % 100 == 0:
            print(f"Progress: {i}/{len(test_cases)} cases processed...")
        
        predicted = run_algorithm(days, miles, receipts)
        
        if predicted is None:
            continue
        
        successful_runs += 1
        error = abs(predicted - expected)
        errors.append(error)
        
        # Track matches
        if error <= 0.01:
            exact_matches += 1
        elif error <= 1.00:
            close_matches += 1
        
        # Track worst case
        if error > max_error:
            max_error = error
            max_error_case = (days, miles, receipts, expected, predicted)
        
        # Collect worst and best cases for analysis
        case_info = (days, miles, receipts, expected, predicted, error)
        if len(worst_cases) < 10:
            worst_cases.append(case_info)
        else:
            worst_cases.sort(key=lambda x: x[5], reverse=True)
            if error > worst_cases[-1][5]:
                worst_cases[-1] = case_info
        
        if len(best_cases) < 10:
            best_cases.append(case_info)
        else:
            best_cases.sort(key=lambda x: x[5])
            if error < best_cases[-1][5]:
                best_cases[-1] = case_info
    
    print(f"Progress: {len(test_cases)}/{len(test_cases)} cases processed...")
    print()
    
    # Calculate statistics
    if errors:
        avg_error = statistics.mean(errors)
        median_error = statistics.median(errors)
        std_error = statistics.stdev(errors) if len(errors) > 1 else 0
    else:
        avg_error = median_error = std_error = 0
    
    # Display results
    print("📈 EVALUATION RESULTS")
    print("=" * 40)
    print(f"Total test cases: {len(test_cases)}")
    print(f"Successful runs: {successful_runs}")
    print(f"Failed runs: {len(test_cases) - successful_runs}")
    print()
    
    print("🎯 ACCURACY METRICS")
    print("-" * 20)
    print(f"Exact matches (≤$0.01): {exact_matches} ({exact_matches/len(test_cases)*100:.1f}%)")
    print(f"Close matches (≤$1.00): {close_matches} ({close_matches/len(test_cases)*100:.1f}%)")
    print()
    
    print("📊 ERROR STATISTICS")
    print("-" * 20)
    print(f"Average error: ${avg_error:.2f}")
    print(f"Median error: ${median_error:.2f}")
    print(f"Standard deviation: ${std_error:.2f}")
    print(f"Maximum error: ${max_error:.2f}")
    print()
    
    if max_error_case:
        days, miles, receipts, expected, predicted = max_error_case
        print("❌ WORST CASE")
        print("-" * 15)
        print(f"Input: {days}d, {miles}mi, ${receipts:.2f}")
        print(f"Expected: ${expected:.2f}")
        print(f"Predicted: ${predicted:.2f}")
        print(f"Error: ${max_error:.2f}")
        print()
    
    # Show performance summary
    if avg_error <= 50:
        grade = "🟢 EXCELLENT"
    elif avg_error <= 100:
        grade = "🟡 GOOD"
    elif avg_error <= 200:
        grade = "🟠 FAIR"
    else:
        grade = "🔴 NEEDS IMPROVEMENT"
    
    print(f"🏆 OVERALL PERFORMANCE: {grade}")
    print(f"Average Error: ${avg_error:.2f}")
    
    # Show top worst cases for debugging
    print("\n🔍 TOP 5 WORST CASES FOR DEBUGGING")
    print("-" * 40)
    worst_cases.sort(key=lambda x: x[5], reverse=True)
    for i, (days, miles, receipts, expected, predicted, error) in enumerate(worst_cases[:5]):
        print(f"{i+1}. {days}d, {miles}mi, ${receipts:.2f} → ${predicted:.2f} (exp ${expected:.2f}, err ${error:.2f})")
    
    return avg_error, exact_matches, successful_runs

if __name__ == "__main__":
    try:
        evaluate_algorithm()
    except KeyboardInterrupt:
        print("\n⚠️ Evaluation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        sys.exit(1)