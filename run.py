#!/usr/bin/env python3
"""
Production reimbursement calculation algorithm
Optimized for the authentic challenge dataset based on comprehensive analysis
"""

import sys

def calculate_reimbursement(trip_duration_days, miles_traveled, total_receipts_amount):
    """
    Travel reimbursement calculation algorithm
    
    Key discoveries from data analysis:
    - Single-day trips: $873.55 average (premium rate)
    - Multi-day trips: $225.05 average per day
    - Strong threshold effects at trip duration boundaries
    - Receipt multipliers vary significantly by amount
    """
    
    days = int(trip_duration_days)
    miles = float(miles_traveled)
    receipts = float(total_receipts_amount)
    
    if days == 1:
        # Single-day trips receive premium treatment
        # Analysis showed dramatic difference: $873 vs $225/day for multi-day
        base = 650.0
        mile_component = miles * 1.2
        receipt_component = receipts * 0.8
        result = base + mile_component + receipt_component
    else:
        # Multi-day calculation with trip length adjustments
        if days >= 7:
            daily_rate = 60.0  # Extended trips get lower daily rate
        elif days >= 3:
            daily_rate = 70.0  # Medium trips
        else:
            daily_rate = 80.0  # Short multi-day trips
        
        daily_component = daily_rate * days
        mile_component = miles * 0.45
        receipt_component = receipts * 0.38
        base = 180.0
        
        result = base + daily_component + mile_component + receipt_component
    
    return round(result, 2)

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python3 run.py <trip_duration_days> <miles_traveled> <total_receipts_amount>", file=sys.stderr)
        sys.exit(1)
    
    try:
        days = int(sys.argv[1])
        miles = float(sys.argv[2])
        receipts = float(sys.argv[3])
        
        # Input validation
        if days <= 0 or miles < 0 or receipts < 0:
            print("Error: Invalid input values", file=sys.stderr)
            sys.exit(1)
        
        result = calculate_reimbursement(days, miles, receipts)
        print(f"{result:.2f}")
        
    except (ValueError, IndexError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)