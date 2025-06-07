#!/usr/bin/env python3
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
    except:
        sys.exit(1)
