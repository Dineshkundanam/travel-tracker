import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

class PatternDetector:
    """Class for detecting patterns in the reimbursement data"""
    
    def __init__(self, df):
        self.df = df
        self.input_columns = ['trip_duration_days', 'miles_traveled', 'total_receipts_amount']
        self.output_column = 'reimbursement_amount'
    
    def detect_patterns(self):
        """Main method to detect various patterns in the data"""
        patterns = {}
        
        patterns['linear_coefficients'] = self._detect_linear_patterns()
        patterns['threshold_rules'] = self._detect_threshold_patterns()
        patterns['clustering_patterns'] = self._detect_clustering_patterns()
        patterns['decision_tree_rules'] = self._extract_decision_tree_rules()
        patterns['multiplicative_patterns'] = self._detect_multiplicative_patterns()
        patterns['additive_components'] = self._detect_additive_components()
        
        return patterns
    
    def _detect_linear_patterns(self):
        """Detect linear relationships between inputs and output"""
        from sklearn.linear_model import LinearRegression
        
        X = self.df[self.input_columns]
        y = self.df[self.output_column]
        
        # Fit linear regression
        lr = LinearRegression()
        lr.fit(X, y)
        
        # Calculate R² score
        r2_score = lr.score(X, y)
        
        coefficients = dict(zip(self.input_columns, lr.coef_))
        
        return {
            'intercept': lr.intercept_,
            'coefficients': coefficients,
            'r2_score': r2_score,
            'formula': f"reimbursement = {lr.intercept_:.2f} + " + 
                      " + ".join([f"{coef:.2f} * {col}" for col, coef in coefficients.items()])
        }
    
    def _detect_threshold_patterns(self):
        """Detect threshold-based patterns"""
        threshold_patterns = {}
        
        for column in self.input_columns:
            # Find potential thresholds
            unique_values = sorted(self.df[column].unique())
            
            # Test common threshold values
            test_thresholds = []
            
            # Add percentile-based thresholds
            for percentile in [10, 25, 50, 75, 90, 95]:
                test_thresholds.append(np.percentile(unique_values, percentile))
            
            # Add round number thresholds
            if column == 'trip_duration_days':
                test_thresholds.extend([1, 3, 5, 7, 10, 14, 21, 30])
            elif column == 'miles_traveled':
                test_thresholds.extend([50, 100, 200, 300, 500, 1000, 1500, 2000])
            elif column == 'total_receipts_amount':
                test_thresholds.extend([25, 50, 100, 150, 200, 300, 500, 1000])
            
            best_threshold = None
            best_score = 0
            
            for threshold in test_thresholds:
                if threshold in unique_values:
                    below = self.df[self.df[column] <= threshold]
                    above = self.df[self.df[column] > threshold]
                    
                    if len(below) >= 10 and len(above) >= 10:
                        # Calculate how well this threshold separates the reimbursement amounts
                        below_mean = below[self.output_column].mean()
                        above_mean = above[self.output_column].mean()
                        
                        # Score based on difference in means relative to overall std
                        overall_std = self.df[self.output_column].std()
                        score = abs(above_mean - below_mean) / overall_std
                        
                        if score > best_score:
                            best_score = score
                            best_threshold = threshold
            
            if best_threshold is not None:
                below = self.df[self.df[column] <= best_threshold]
                above = self.df[self.df[column] > best_threshold]
                
                threshold_patterns[column] = {
                    'threshold': best_threshold,
                    'below_mean': below[self.output_column].mean(),
                    'above_mean': above[self.output_column].mean(),
                    'below_count': len(below),
                    'above_count': len(above),
                    'separation_score': best_score
                }
        
        return threshold_patterns
    
    def _detect_clustering_patterns(self):
        """Detect clustering patterns in the data"""
        # Standardize the input features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(self.df[self.input_columns])
        
        # Try different numbers of clusters
        cluster_results = {}
        
        for n_clusters in range(2, 8):
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(X_scaled)
            
            # Calculate cluster statistics
            cluster_stats = []
            for i in range(n_clusters):
                cluster_data = self.df[cluster_labels == i]
                if len(cluster_data) > 0:
                    stats = {
                        'cluster_id': i,
                        'size': len(cluster_data),
                        'avg_reimbursement': cluster_data[self.output_column].mean(),
                        'std_reimbursement': cluster_data[self.output_column].std(),
                        'avg_duration': cluster_data['trip_duration_days'].mean(),
                        'avg_miles': cluster_data['miles_traveled'].mean(),
                        'avg_receipts': cluster_data['total_receipts_amount'].mean()
                    }
                    cluster_stats.append(stats)
            
            # Calculate silhouette score
            from sklearn.metrics import silhouette_score
            silhouette = silhouette_score(X_scaled, cluster_labels)
            
            cluster_results[n_clusters] = {
                'silhouette_score': silhouette,
                'cluster_stats': cluster_stats
            }
        
        # Find best number of clusters
        best_n_clusters = max(cluster_results.keys(), 
                             key=lambda k: cluster_results[k]['silhouette_score'])
        
        return {
            'best_n_clusters': best_n_clusters,
            'all_results': cluster_results,
            'best_result': cluster_results[best_n_clusters]
        }
    
    def _extract_decision_tree_rules(self):
        """Extract rules from a decision tree"""
        X = self.df[self.input_columns]
        y = self.df[self.output_column]
        
        # Fit decision tree with limited depth to avoid overfitting
        dt = DecisionTreeRegressor(max_depth=5, min_samples_split=20, min_samples_leaf=10)
        dt.fit(X, y)
        
        # Extract rules
        tree_rules = self._get_tree_rules(dt, self.input_columns)
        
        # Calculate cross-validation score
        cv_scores = cross_val_score(dt, X, y, cv=5)
        
        return {
            'rules': tree_rules,
            'cv_score_mean': cv_scores.mean(),
            'cv_score_std': cv_scores.std(),
            'feature_importance': dict(zip(self.input_columns, dt.feature_importances_))
        }
    
    def _get_tree_rules(self, tree, feature_names):
        """Extract human-readable rules from decision tree"""
        tree_ = tree.tree_
        feature_name = [
            feature_names[i] if i != -2
            else "undefined!"
            for i in tree_.feature
        ]
        
        def recurse(node, depth):
            indent = "  " * depth
            if tree_.feature[node] != -2:
                name = feature_name[node]
                threshold = tree_.threshold[node]
                yield f"{indent}if {name} <= {threshold:.2f}:"
                yield from recurse(tree_.children_left[node], depth + 1)
                yield f"{indent}else:  # if {name} > {threshold:.2f}"
                yield from recurse(tree_.children_right[node], depth + 1)
            else:
                yield f"{indent}return {tree_.value[node][0][0]:.2f}"
        
        return list(recurse(0, 0))
    
    def _detect_multiplicative_patterns(self):
        """Detect multiplicative patterns (e.g., rate * quantity)"""
        multiplicative_patterns = {}
        
        # Test various multiplicative combinations
        combinations = [
            ('trip_duration_days', 'miles_traveled'),
            ('trip_duration_days', 'total_receipts_amount'),
            ('miles_traveled', 'total_receipts_amount')
        ]
        
        for col1, col2 in combinations:
            # Create multiplicative feature
            product = self.df[col1] * self.df[col2]
            
            # Calculate correlation with output
            correlation = product.corr(self.df[self.output_column])
            
            # Try to find a simple multiplier
            if correlation > 0.5:  # Only consider strong correlations
                # Simple regression: reimbursement = k * (col1 * col2)
                multiplier = (self.df[self.output_column] / product).mean()
                
                # Calculate how well this works
                predicted = multiplier * product
                r_squared = 1 - ((self.df[self.output_column] - predicted) ** 2).sum() / \
                           ((self.df[self.output_column] - self.df[self.output_column].mean()) ** 2).sum()
                
                multiplicative_patterns[f"{col1}_x_{col2}"] = {
                    'multiplier': multiplier,
                    'correlation': correlation,
                    'r_squared': r_squared
                }
        
        return multiplicative_patterns
    
    def _detect_additive_components(self):
        """Detect additive components in the reimbursement calculation"""
        # Try to decompose reimbursement into additive components
        
        # Test if reimbursement can be explained as sum of components
        components = {}
        
        # Component 1: Daily allowance (duration-based)
        daily_rates = self.df[self.output_column] / self.df['trip_duration_days']
        daily_rate_mean = daily_rates.mean()
        daily_rate_std = daily_rates.std()
        
        components['daily_allowance'] = {
            'rate_per_day': daily_rate_mean,
            'std_deviation': daily_rate_std,
            'coefficient_of_variation': daily_rate_std / daily_rate_mean
        }
        
        # Component 2: Mileage allowance
        mileage_rates = self.df[self.output_column] / self.df['miles_traveled']
        mileage_rate_mean = mileage_rates.mean()
        mileage_rate_std = mileage_rates.std()
        
        components['mileage_allowance'] = {
            'rate_per_mile': mileage_rate_mean,
            'std_deviation': mileage_rate_std,
            'coefficient_of_variation': mileage_rate_std / mileage_rate_mean
        }
        
        # Component 3: Receipt reimbursement
        receipt_ratios = self.df[self.output_column] / self.df['total_receipts_amount']
        receipt_ratio_mean = receipt_ratios.mean()
        receipt_ratio_std = receipt_ratios.std()
        
        components['receipt_reimbursement'] = {
            'ratio_to_receipts': receipt_ratio_mean,
            'std_deviation': receipt_ratio_std,
            'coefficient_of_variation': receipt_ratio_std / receipt_ratio_mean
        }
        
        # Try three-component additive model
        estimated_daily = daily_rate_mean * self.df['trip_duration_days']
        estimated_mileage = (mileage_rate_mean / 3) * self.df['miles_traveled']  # Scaled down
        estimated_receipts = 1.2 * self.df['total_receipts_amount']  # Common multiplier for receipts
        
        total_estimated = estimated_daily * 0.4 + estimated_mileage + estimated_receipts
        
        r_squared = 1 - ((self.df[self.output_column] - total_estimated) ** 2).sum() / \
                   ((self.df[self.output_column] - self.df[self.output_column].mean()) ** 2).sum()
        
        components['three_component_model'] = {
            'r_squared': r_squared,
            'daily_weight': 0.4,
            'mileage_rate': mileage_rate_mean / 3,
            'receipt_multiplier': 1.2
        }
        
        return components
    
    def segment_analysis(self, column, n_bins):
        """Perform segment analysis on a specific column"""
        # Create bins
        bins = pd.cut(self.df[column], bins=n_bins)
        
        segments = []
        for bin_interval in bins.cat.categories:
            mask = bins == bin_interval
            segment_data = self.df[mask]
            
            if len(segment_data) > 0:
                segment_info = {
                    'range': f"{bin_interval.left:.1f} - {bin_interval.right:.1f}",
                    'count': len(segment_data),
                    'avg_reimbursement': segment_data[self.output_column].mean(),
                    'median_reimbursement': segment_data[self.output_column].median(),
                    'std_reimbursement': segment_data[self.output_column].std(),
                    'min_reimbursement': segment_data[self.output_column].min(),
                    'max_reimbursement': segment_data[self.output_column].max()
                }
                segments.append(segment_info)
        
        return segments
