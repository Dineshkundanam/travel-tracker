#!/usr/bin/env python3
"""
Run predictions on private_cases.json using the optimized algorithm
"""

import json
import subprocess
import sys
from typing import List, Dict

def load_private_cases(filename: str = 'private_cases.json') -> List[Dict]:
    """Load private test cases (input only)"""
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print(f"Error: {filename} not found!")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in {filename}")
        sys.exit(1)

def run_algorithm(days: int, miles: float, receipts: float) -> float:
    """Run the algorithm via run.py script"""
    try:
        result = subprocess.run(
            ['python', 'run.py', str(days), str(miles), str(receipts)],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode != 0:
            raise Exception(f"Algorithm failed: {result.stderr}")
        
        return float(result.stdout.strip())
    except (subprocess.TimeoutExpired, ValueError) as e:
        raise Exception(f"Algorithm execution error: {e}")

def generate_predictions():
    """Generate predictions for all private cases"""
    print("Private Cases Prediction Generator")
    print("=" * 40)
    
    # Load private cases
    print("Loading private cases...")
    private_cases = load_private_cases()
    print(f"Loaded {len(private_cases)} private cases")
    
    # Generate predictions
    predictions = []
    successful = 0
    failed = 0
    
    print("Generating predictions...")
    
    for i, case in enumerate(private_cases):
        if i % 1000 == 0 and i > 0:
            print(f"Progress: {i}/{len(private_cases)} cases processed...")
        
        try:
            days = case['trip_duration_days']
            miles = case['miles_traveled']
            receipts = case['total_receipts_amount']
            
            prediction = run_algorithm(days, miles, receipts)
            
            predictions.append({
                "case_id": i + 1,
                "input": {
                    "trip_duration_days": days,
                    "miles_traveled": miles,
                    "total_receipts_amount": receipts
                },
                "predicted_reimbursement": prediction
            })
            
            successful += 1
            
        except Exception as e:
            print(f"Failed case {i+1}: {e}")
            failed += 1
            continue
    
    print(f"Completed: {successful} successful, {failed} failed")
    
    # Save predictions
    output_file = 'private_cases_predictions.json'
    with open(output_file, 'w') as f:
        json.dump(predictions, f, indent=2)
    
    print(f"Predictions saved to: {output_file}")
    
    # Generate summary statistics
    if predictions:
        amounts = [p['predicted_reimbursement'] for p in predictions]
        print(f"\nPrediction Statistics:")
        print(f"  Total predictions: {len(amounts)}")
        print(f"  Range: ${min(amounts):.2f} - ${max(amounts):.2f}")
        print(f"  Average: ${sum(amounts)/len(amounts):.2f}")
        
        # Show sample predictions
        print(f"\nSample Predictions:")
        for i, pred in enumerate(predictions[:5]):
            inp = pred['input']
            amt = pred['predicted_reimbursement']
            print(f"  {i+1}. {inp['trip_duration_days']}d, {inp['miles_traveled']}mi, ${inp['total_receipts_amount']:.2f} → ${amt:.2f}")
    
    return predictions

if __name__ == "__main__":
    try:
        generate_predictions()
    except KeyboardInterrupt:
        print("\nPrediction generation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error during prediction generation: {e}")
        sys.exit(1)