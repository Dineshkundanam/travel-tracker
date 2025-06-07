import pandas as pd
import numpy as np
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go

class DataAnalyzer:
    """Class for analyzing the historical reimbursement data"""
    
    def __init__(self, df):
        self.df = df
        self.input_columns = ['trip_duration_days', 'miles_traveled', 'total_receipts_amount']
        self.output_column = 'reimbursement_amount'
    
    def get_basic_statistics(self):
        """Get basic statistical summary of the data"""
        return self.df.describe()
    
    def get_correlation_matrix(self):
        """Calculate correlation matrix between all variables"""
        return self.df[self.input_columns + [self.output_column]].corr()
    
    def detect_outliers(self, method='iqr'):
        """Detect outliers in the dataset"""
        outliers = {}
        
        for column in self.input_columns + [self.output_column]:
            if method == 'iqr':
                Q1 = self.df[column].quantile(0.25)
                Q3 = self.df[column].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers[column] = self.df[
                    (self.df[column] < lower_bound) | 
                    (self.df[column] > upper_bound)
                ].index.tolist()
            elif method == 'zscore':
                z_scores = np.abs(stats.zscore(self.df[column]))
                outliers[column] = self.df[z_scores > 3].index.tolist()
        
        return outliers
    
    def analyze_distributions(self):
        """Analyze the distribution of each variable"""
        distributions = {}
        
        for column in self.input_columns + [self.output_column]:
            # Test for normality
            stat, p_value = stats.normaltest(self.df[column])
            is_normal = p_value > 0.05
            
            distributions[column] = {
                'mean': self.df[column].mean(),
                'median': self.df[column].median(),
                'std': self.df[column].std(),
                'skewness': stats.skew(self.df[column]),
                'kurtosis': stats.kurtosis(self.df[column]),
                'is_normal': is_normal,
                'normality_p_value': p_value
            }
        
        return distributions
    
    def find_linear_relationships(self):
        """Find linear relationships between inputs and output"""
        relationships = {}
        
        for input_col in self.input_columns:
            # Calculate Pearson correlation
            correlation, p_value = stats.pearsonr(
                self.df[input_col], 
                self.df[self.output_column]
            )
            
            # Perform linear regression
            slope, intercept, r_value, p_val, std_err = stats.linregress(
                self.df[input_col], 
                self.df[self.output_column]
            )
            
            relationships[input_col] = {
                'correlation': correlation,
                'correlation_p_value': p_value,
                'regression_slope': slope,
                'regression_intercept': intercept,
                'r_squared': r_value**2,
                'regression_p_value': p_val,
                'standard_error': std_err
            }
        
        return relationships
    
    def segment_data(self, column, n_segments=5):
        """Segment data based on a column and analyze each segment"""
        # Create segments
        segments = pd.cut(self.df[column], bins=n_segments, labels=False)
        
        segment_analysis = []
        for i in range(n_segments):
            segment_data = self.df[segments == i]
            
            if len(segment_data) > 0:
                analysis = {
                    'segment': i,
                    'count': len(segment_data),
                    'min_value': segment_data[column].min(),
                    'max_value': segment_data[column].max(),
                    'avg_reimbursement': segment_data[self.output_column].mean(),
                    'median_reimbursement': segment_data[self.output_column].median(),
                    'std_reimbursement': segment_data[self.output_column].std()
                }
                segment_analysis.append(analysis)
        
        return segment_analysis
    
    def find_threshold_effects(self):
        """Look for threshold effects in the data"""
        threshold_effects = {}
        
        for column in self.input_columns:
            # Try different potential thresholds
            values = sorted(self.df[column].unique())
            potential_thresholds = [
                np.percentile(values, p) for p in [25, 50, 75, 90, 95]
            ]
            
            best_threshold = None
            best_improvement = 0
            
            for threshold in potential_thresholds:
                # Split data at threshold
                below = self.df[self.df[column] <= threshold]
                above = self.df[self.df[column] > threshold]
                
                if len(below) > 10 and len(above) > 10:
                    # Calculate variance in each group
                    var_below = below[self.output_column].var()
                    var_above = above[self.output_column].var()
                    total_var = self.df[self.output_column].var()
                    
                    # Calculate improvement in variance explanation
                    weighted_var = (len(below) * var_below + len(above) * var_above) / len(self.df)
                    improvement = (total_var - weighted_var) / total_var
                    
                    if improvement > best_improvement:
                        best_improvement = improvement
                        best_threshold = threshold
            
            if best_threshold is not None:
                threshold_effects[column] = {
                    'threshold': best_threshold,
                    'variance_improvement': best_improvement,
                    'below_avg': self.df[self.df[column] <= best_threshold][self.output_column].mean(),
                    'above_avg': self.df[self.df[column] > best_threshold][self.output_column].mean()
                }
        
        return threshold_effects
    
    def calculate_feature_importance(self):
        """Calculate relative importance of each input feature"""
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.metrics import r2_score
        
        X = self.df[self.input_columns]
        y = self.df[self.output_column]
        
        # Fit random forest
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X, y)
        
        # Get feature importance
        importance = dict(zip(self.input_columns, rf.feature_importances_))
        
        # Calculate individual R² scores
        individual_r2 = {}
        for col in self.input_columns:
            individual_pred = rf.predict(X[[col]])
            individual_r2[col] = r2_score(y, individual_pred)
        
        return {
            'feature_importance': importance,
            'individual_r2': individual_r2,
            'model_r2': r2_score(y, rf.predict(X))
        }
    
    def get_extreme_cases(self, n_cases=10):
        """Get extreme cases for analysis"""
        extreme_cases = {
            'highest_reimbursement': self.df.nlargest(n_cases, self.output_column),
            'lowest_reimbursement': self.df.nsmallest(n_cases, self.output_column),
            'highest_ratio_to_receipts': self.df.nlargest(
                n_cases, 
                self.df[self.output_column] / (self.df['total_receipts_amount'] + 0.01)
            ),
            'lowest_ratio_to_receipts': self.df.nsmallest(
                n_cases, 
                self.df[self.output_column] / (self.df['total_receipts_amount'] + 0.01)
            )
        }
        
        return extreme_cases
