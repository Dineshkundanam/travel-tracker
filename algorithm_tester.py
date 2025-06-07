import pandas as pd
import numpy as np
from typing import Callable, Dict, List, Any

class AlgorithmTester:
    """Class for testing reimbursement calculation algorithms"""
    
    def __init__(self, df):
        self.df = df
        self.input_columns = ['trip_duration_days', 'miles_traveled', 'total_receipts_amount']
        self.output_column = 'reimbursement_amount'
    
    def test_algorithm(self, algorithm_func: Callable) -> Dict[str, Any]:
        """Test an algorithm function against the historical data"""
        results = {
            'predicted': [],
            'expected': [],
            'errors': [],
            'exact_matches': 0,
            'close_matches': 0,
            'total_cases': len(self.df)
        }
        
        for _, row in self.df.iterrows():
            # Get inputs
            trip_duration = int(row['trip_duration_days'])
            miles_traveled = int(row['miles_traveled'])
            receipts_amount = float(row['total_receipts_amount'])
            expected = float(row['reimbursement_amount'])
            
            try:
                # Call the algorithm
                predicted = algorithm_func(trip_duration, miles_traveled, receipts_amount)
                predicted = float(predicted)
                
                # Calculate error
                error = abs(predicted - expected)
                
                # Store results
                results['predicted'].append(predicted)
                results['expected'].append(expected)
                results['errors'].append(error)
                
                # Count matches
                if error <= 0.01:  # Exact match (within 1 cent)
                    results['exact_matches'] += 1
                elif error <= 1.00:  # Close match (within $1)
                    results['close_matches'] += 1
                    
            except Exception as e:
                # Handle algorithm errors
                results['predicted'].append(0.0)
                results['expected'].append(expected)
                results['errors'].append(expected)
        
        # Calculate summary statistics
        results['average_error'] = np.mean(results['errors'])
        results['median_error'] = np.median(results['errors'])
        results['max_error'] = np.max(results['errors'])
        results['min_error'] = np.min(results['errors'])
        results['std_error'] = np.std(results['errors'])
        
        # Calculate accuracy metrics
        results['exact_match_rate'] = results['exact_matches'] / results['total_cases']
        results['close_match_rate'] = results['close_matches'] / results['total_cases']
        results['accuracy_score'] = (results['exact_matches'] + results['close_matches']) / results['total_cases']
        
        # Calculate R-squared
        if len(results['predicted']) > 0:
            ss_res = np.sum((np.array(results['expected']) - np.array(results['predicted'])) ** 2)
            ss_tot = np.sum((np.array(results['expected']) - np.mean(results['expected'])) ** 2)
            results['r_squared'] = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        else:
            results['r_squared'] = 0
        
        return results
    
    def compare_algorithms(self, algorithms: Dict[str, Callable]) -> Dict[str, Dict[str, Any]]:
        """Compare multiple algorithms"""
        comparison_results = {}
        
        for name, algorithm in algorithms.items():
            print(f"Testing {name}...")
            comparison_results[name] = self.test_algorithm(algorithm)
        
        return comparison_results
    
    def get_worst_predictions(self, algorithm_func: Callable, n_cases: int = 10) -> pd.DataFrame:
        """Get the worst predictions from an algorithm"""
        results = self.test_algorithm(algorithm_func)
        
        # Create DataFrame with results
        results_df = pd.DataFrame({
            'trip_duration_days': self.df['trip_duration_days'],
            'miles_traveled': self.df['miles_traveled'],
            'total_receipts_amount': self.df['total_receipts_amount'],
            'expected': results['expected'],
            'predicted': results['predicted'],
            'error': results['errors']
        })
        
        # Sort by error and return worst cases
        worst_cases = results_df.nlargest(n_cases, 'error')
        return worst_cases
    
    def get_best_predictions(self, algorithm_func: Callable, n_cases: int = 10) -> pd.DataFrame:
        """Get the best predictions from an algorithm"""
        results = self.test_algorithm(algorithm_func)
        
        # Create DataFrame with results
        results_df = pd.DataFrame({
            'trip_duration_days': self.df['trip_duration_days'],
            'miles_traveled': self.df['miles_traveled'],
            'total_receipts_amount': self.df['total_receipts_amount'],
            'expected': results['expected'],
            'predicted': results['predicted'],
            'error': results['errors']
        })
        
        # Sort by error and return best cases
        best_cases = results_df.nsmallest(n_cases, 'error')
        return best_cases
    
    def analyze_error_patterns(self, algorithm_func: Callable) -> Dict[str, Any]:
        """Analyze patterns in prediction errors"""
        results = self.test_algorithm(algorithm_func)
        
        # Create DataFrame for analysis
        analysis_df = pd.DataFrame({
            'trip_duration_days': self.df['trip_duration_days'],
            'miles_traveled': self.df['miles_traveled'],
            'total_receipts_amount': self.df['total_receipts_amount'],
            'expected': results['expected'],
            'predicted': results['predicted'],
            'error': results['errors'],
            'relative_error': np.array(results['errors']) / np.array(results['expected'])
        })
        
        error_patterns = {}
        
        # Analyze errors by input ranges
        for column in self.input_columns:
            # Create quartiles
            quartiles = pd.qcut(analysis_df[column], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
            
            error_by_quartile = {}
            for quartile in ['Q1', 'Q2', 'Q3', 'Q4']:
                mask = quartiles == quartile
                if mask.sum() > 0:
                    error_by_quartile[quartile] = {
                        'mean_error': analysis_df[mask]['error'].mean(),
                        'median_error': analysis_df[mask]['error'].median(),
                        'mean_relative_error': analysis_df[mask]['relative_error'].mean(),
                        'count': mask.sum()
                    }
            
            error_patterns[f'{column}_quartiles'] = error_by_quartile
        
        # Overall error statistics
        error_patterns['overall'] = {
            'mean_error': np.mean(results['errors']),
            'median_error': np.median(results['errors']),
            'error_std': np.std(results['errors']),
            'mean_relative_error': np.mean(analysis_df['relative_error']),
            'error_skewness': analysis_df['error'].skew(),
            'error_kurtosis': analysis_df['error'].kurtosis()
        }
        
        return error_patterns
    
    def benchmark_algorithm(self, algorithm_func: Callable) -> Dict[str, Any]:
        """Comprehensive benchmark of an algorithm"""
        import time
        
        # Performance timing
        start_time = time.time()
        results = self.test_algorithm(algorithm_func)
        end_time = time.time()
        
        # Error analysis
        error_patterns = self.analyze_error_patterns(algorithm_func)
        
        # Worst and best cases
        worst_cases = self.get_worst_predictions(algorithm_func, 5)
        best_cases = self.get_best_predictions(algorithm_func, 5)
        
        benchmark = {
            'performance': {
                'total_time': end_time - start_time,
                'avg_time_per_case': (end_time - start_time) / len(self.df),
                'cases_per_second': len(self.df) / (end_time - start_time)
            },
            'accuracy': {
                'exact_matches': results['exact_matches'],
                'close_matches': results['close_matches'],
                'exact_match_rate': results['exact_match_rate'],
                'close_match_rate': results['close_match_rate'],
                'accuracy_score': results['accuracy_score']
            },
            'error_metrics': {
                'average_error': results['average_error'],
                'median_error': results['median_error'],
                'max_error': results['max_error'],
                'min_error': results['min_error'],
                'std_error': results['std_error'],
                'r_squared': results['r_squared']
            },
            'error_patterns': error_patterns,
            'worst_cases': worst_cases.to_dict('records'),
            'best_cases': best_cases.to_dict('records')
        }
        
        return benchmark
