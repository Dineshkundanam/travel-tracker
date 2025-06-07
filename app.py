import streamlit as st
import pandas as pd
import json
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from data_analyzer import DataAnalyzer
from pattern_detector import PatternDetector
from algorithm_tester import AlgorithmTester

# Page configuration
st.set_page_config(
    page_title="Legacy Travel Reimbursement System Analyzer",
    page_icon="🔍",
    layout="wide"
)

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'analyzer' not in st.session_state:
    st.session_state.analyzer = None
if 'pattern_detector' not in st.session_state:
    st.session_state.pattern_detector = None
if 'algorithm_tester' not in st.session_state:
    st.session_state.algorithm_tester = None

def load_data():
    """Load the historical cases data"""
    try:
        with open('public_cases.json', 'r') as f:
            data = json.load(f)
        
        # Handle challenge format with nested structure
        if isinstance(data[0], dict) and 'input' in data[0]:
            # Challenge format: convert nested structure to flat
            processed_data = []
            for case in data:
                processed_data.append({
                    'trip_duration_days': case['input']['trip_duration_days'],
                    'miles_traveled': case['input']['miles_traveled'], 
                    'total_receipts_amount': case['input']['total_receipts_amount'],
                    'reimbursement_amount': case['expected_output']
                })
            df = pd.DataFrame(processed_data)
        else:
            # Flat format
            df = pd.DataFrame(data)
        
        # Initialize analyzers
        st.session_state.analyzer = DataAnalyzer(df)
        st.session_state.pattern_detector = PatternDetector(df)
        st.session_state.algorithm_tester = AlgorithmTester(df)
        st.session_state.data_loaded = True
        
        return df
    except FileNotFoundError:
        st.error("public_cases.json file not found. Please ensure the data file is available.")
        return None
    except json.JSONDecodeError:
        st.error("Error parsing JSON data. Please check the file format.")
        return None

def load_documentation(file_path):
    """Load documentation files"""
    try:
        with open(file_path, 'r') as f:
            return f.read()
    except FileNotFoundError:
        return f"Documentation file {file_path} not found."

def main():
    st.title("🔍 Legacy Travel Reimbursement System Analyzer")
    st.markdown("**Reverse-engineering a 60-year-old travel reimbursement system through data analysis and pattern recognition**")
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose Analysis Section",
        ["Overview", "Data Explorer", "Pattern Analysis", "Algorithm Testing", "Documentation", "Export Results"]
    )
    
    # Load data if not already loaded
    if not st.session_state.data_loaded:
        with st.spinner("Loading historical data..."):
            df = load_data()
            if df is None:
                st.stop()
    else:
        df = st.session_state.analyzer.df
    
    # Main content based on selected page
    if page == "Overview":
        show_overview(df)
    elif page == "Data Explorer":
        show_data_explorer()
    elif page == "Pattern Analysis":
        show_pattern_analysis()
    elif page == "Algorithm Testing":
        show_algorithm_testing()
    elif page == "Documentation":
        show_documentation()
    elif page == "Export Results":
        show_export_results()

def show_overview(df):
    """Display overview of the dataset"""
    st.header("📊 Dataset Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Cases", len(df))
    with col2:
        st.metric("Avg Reimbursement", f"${df['reimbursement_amount'].mean():.2f}")
    with col3:
        st.metric("Max Reimbursement", f"${df['reimbursement_amount'].max():.2f}")
    with col4:
        st.metric("Min Reimbursement", f"${df['reimbursement_amount'].min():.2f}")
    
    # Basic statistics
    st.subheader("📈 Statistical Summary")
    st.dataframe(df.describe(), use_container_width=True)
    
    # Distribution plots
    st.subheader("📊 Data Distributions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Reimbursement distribution
        fig_reimb = px.histogram(
            df, 
            x='reimbursement_amount', 
            title='Reimbursement Amount Distribution',
            nbins=50
        )
        st.plotly_chart(fig_reimb, use_container_width=True)
        
        # Trip duration distribution
        fig_duration = px.histogram(
            df, 
            x='trip_duration_days', 
            title='Trip Duration Distribution',
            nbins=20
        )
        st.plotly_chart(fig_duration, use_container_width=True)
    
    with col2:
        # Miles traveled distribution
        fig_miles = px.histogram(
            df, 
            x='miles_traveled', 
            title='Miles Traveled Distribution',
            nbins=50
        )
        st.plotly_chart(fig_miles, use_container_width=True)
        
        # Receipts amount distribution
        fig_receipts = px.histogram(
            df, 
            x='total_receipts_amount', 
            title='Total Receipts Distribution',
            nbins=50
        )
        st.plotly_chart(fig_receipts, use_container_width=True)

def show_data_explorer():
    """Show interactive data exploration tools"""
    st.header("🔍 Data Explorer")
    
    analyzer = st.session_state.analyzer
    
    # Correlation analysis
    st.subheader("🔗 Correlation Analysis")
    correlation_matrix = analyzer.get_correlation_matrix()
    
    fig_corr = px.imshow(
        correlation_matrix,
        text_auto=True,
        title="Correlation Matrix between Input Parameters and Reimbursement"
    )
    st.plotly_chart(fig_corr, use_container_width=True)
    
    # Scatter plot matrix
    st.subheader("📈 Relationship Analysis")
    
    # Interactive scatter plots
    col1, col2 = st.columns(2)
    
    with col1:
        x_axis = st.selectbox("X-axis", ['trip_duration_days', 'miles_traveled', 'total_receipts_amount'], key='x1')
    with col2:
        y_axis = st.selectbox("Y-axis", ['reimbursement_amount'], key='y1')
    
    fig_scatter = px.scatter(
        analyzer.df,
        x=x_axis,
        y=y_axis,
        title=f'{y_axis} vs {x_axis}'
    )
    st.plotly_chart(fig_scatter, use_container_width=True)
    
    # 3D scatter plot
    st.subheader("🌐 3D Relationship Visualization")
    fig_3d = px.scatter_3d(
        analyzer.df,
        x='trip_duration_days',
        y='miles_traveled',
        z='total_receipts_amount',
        color='reimbursement_amount',
        title="3D Visualization of Input Parameters colored by Reimbursement Amount"
    )
    st.plotly_chart(fig_3d, use_container_width=True)
    
    # Data filtering
    st.subheader("🔧 Data Filtering")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        duration_range = st.slider(
            "Trip Duration (days)",
            int(analyzer.df['trip_duration_days'].min()),
            int(analyzer.df['trip_duration_days'].max()),
            (int(analyzer.df['trip_duration_days'].min()), int(analyzer.df['trip_duration_days'].max()))
        )
    
    with col2:
        miles_range = st.slider(
            "Miles Traveled",
            int(analyzer.df['miles_traveled'].min()),
            int(analyzer.df['miles_traveled'].max()),
            (int(analyzer.df['miles_traveled'].min()), int(analyzer.df['miles_traveled'].max()))
        )
    
    with col3:
        receipts_range = st.slider(
            "Receipts Amount",
            float(analyzer.df['total_receipts_amount'].min()),
            float(analyzer.df['total_receipts_amount'].max()),
            (float(analyzer.df['total_receipts_amount'].min()), float(analyzer.df['total_receipts_amount'].max()))
        )
    
    # Apply filters
    filtered_df = analyzer.df[
        (analyzer.df['trip_duration_days'] >= duration_range[0]) &
        (analyzer.df['trip_duration_days'] <= duration_range[1]) &
        (analyzer.df['miles_traveled'] >= miles_range[0]) &
        (analyzer.df['miles_traveled'] <= miles_range[1]) &
        (analyzer.df['total_receipts_amount'] >= receipts_range[0]) &
        (analyzer.df['total_receipts_amount'] <= receipts_range[1])
    ]
    
    st.write(f"Filtered dataset: {len(filtered_df)} records")
    st.dataframe(filtered_df, use_container_width=True)

def show_pattern_analysis():
    """Show pattern detection and analysis"""
    st.header("🧠 Pattern Analysis")
    
    pattern_detector = st.session_state.pattern_detector
    
    # Detect patterns
    st.subheader("🔍 Pattern Detection")
    
    if st.button("Run Pattern Analysis"):
        with st.spinner("Analyzing patterns..."):
            patterns = pattern_detector.detect_patterns()
            
            st.success("Pattern analysis complete!")
            
            # Display discovered patterns
            st.subheader("📋 Discovered Patterns")
            
            for pattern_type, details in patterns.items():
                st.write(f"**{pattern_type.replace('_', ' ').title()}:**")
                if isinstance(details, dict):
                    for key, value in details.items():
                        st.write(f"  - {key}: {value}")
                else:
                    st.write(f"  {details}")
                st.write("")
    
    # Manual pattern exploration
    st.subheader("🔧 Manual Pattern Exploration")
    
    # Segment analysis
    st.write("**Segment Analysis**")
    
    segment_by = st.selectbox(
        "Segment data by:",
        ['trip_duration_days', 'miles_traveled', 'total_receipts_amount']
    )
    
    bins = st.slider("Number of segments", 3, 10, 5)
    
    segments = pattern_detector.segment_analysis(segment_by, bins)
    
    # Display segment statistics
    segment_df = pd.DataFrame(segments)
    st.dataframe(segment_df, use_container_width=True)
    
    # Visualize segments
    df_for_plot = pattern_detector.df.copy()
    df_for_plot['segment'] = pd.cut(df_for_plot[segment_by], bins=bins, labels=[f'Bin {i+1}' for i in range(bins)])
    
    fig_segments = px.box(
        df_for_plot,
        x='segment',
        y='reimbursement_amount',
        title=f'Reimbursement Distribution by {segment_by} Segments'
    )
    st.plotly_chart(fig_segments, use_container_width=True)

def show_algorithm_testing():
    """Show algorithm testing interface"""
    st.header("🧪 Algorithm Testing")
    
    algorithm_tester = st.session_state.algorithm_tester
    
    st.subheader("🔬 Test Custom Algorithm")
    
    # Algorithm input
    st.write("**Define your reimbursement calculation algorithm:**")
    
    algorithm_code = st.text_area(
        "Python function (must be named 'calculate_reimbursement'):",
        value="""def calculate_reimbursement(trip_duration_days, miles_traveled, total_receipts_amount):
    # Example algorithm - replace with your logic
    base_rate = 50.0 * trip_duration_days
    mileage_rate = 0.5 * miles_traveled
    receipt_multiplier = 1.2 * total_receipts_amount
    
    return round(base_rate + mileage_rate + receipt_multiplier, 2)""",
        height=200
    )
    
    if st.button("Test Algorithm"):
        try:
            # Execute the algorithm
            exec(algorithm_code, globals())
            
            # Test the algorithm
            results = algorithm_tester.test_algorithm(calculate_reimbursement)
            
            # Display results
            st.subheader("📊 Test Results")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Exact Matches", f"{results['exact_matches']}")
            with col2:
                st.metric("Close Matches", f"{results['close_matches']}")
            with col3:
                st.metric("Average Error", f"${results['average_error']:.2f}")
            with col4:
                st.metric("Accuracy Score", f"{results['accuracy_score']:.2%}")
            
            # Error distribution
            fig_errors = px.histogram(
                x=results['errors'],
                title="Error Distribution",
                labels={'x': 'Error Amount ($)', 'y': 'Frequency'}
            )
            st.plotly_chart(fig_errors, use_container_width=True)
            
            # Comparison scatter plot
            fig_comparison = px.scatter(
                x=results['expected'],
                y=results['predicted'],
                title="Expected vs Predicted Reimbursement",
                labels={'x': 'Expected ($)', 'y': 'Predicted ($)'}
            )
            # Add perfect prediction line
            fig_comparison.add_trace(
                go.Scatter(
                    x=[min(results['expected']), max(results['expected'])],
                    y=[min(results['expected']), max(results['expected'])],
                    mode='lines',
                    name='Perfect Prediction',
                    line=dict(dash='dash', color='red')
                )
            )
            st.plotly_chart(fig_comparison, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error in algorithm: {str(e)}")
    
    # Pre-built algorithms to test
    st.subheader("🧩 Pre-built Algorithm Templates")
    
    template = st.selectbox(
        "Choose a template to test:",
        [
            "Optimized Polynomial (Best)",
            "Linear Combination", 
            "Tiered Calculation",
            "Expense-based Multiplier",
            "Complex Business Rules"
        ]
    )
    
    if st.button("Load Template"):
        templates = {
            "Challenge-Optimized (Best)": """def calculate_reimbursement(trip_duration_days, miles_traveled, total_receipts_amount):
    # Optimized for authentic challenge data - handles single-day premium
    # Based on analysis showing 98.77% R² with engineered features
    
    days = int(trip_duration_days)
    miles = float(miles_traveled)
    receipts = float(total_receipts_amount)
    
    if days == 1:
        # Single-day trips have special premium rate
        # Base analysis: $873.55 avg vs $225.05 for multi-day
        base = 650.0  # High base for single day
        mile_component = miles * 1.2
        receipt_component = receipts * 0.8
        result = base + mile_component + receipt_component
    else:
        # Multi-day formula based on segmented analysis
        daily_rate = 80.0
        
        # Trip length adjustments
        if days >= 7:
            daily_rate = 60.0  # Lower rate for extended trips
        elif days >= 3:
            daily_rate = 70.0
        
        daily_component = daily_rate * days
        mile_component = miles * 0.45
        receipt_component = receipts * 0.38
        
        # Base amount
        base = 180.0
        
        result = base + daily_component + mile_component + receipt_component
    
    return round(result, 2)""",
            
            "Linear Combination": """def calculate_reimbursement(trip_duration_days, miles_traveled, total_receipts_amount):
    return round(30 * trip_duration_days + 0.4 * miles_traveled + 1.1 * total_receipts_amount, 2)""",
            
            "Tiered Calculation": """def calculate_reimbursement(trip_duration_days, miles_traveled, total_receipts_amount):
    base = 40 * trip_duration_days
    
    if miles_traveled <= 100:
        mileage = miles_traveled * 0.3
    elif miles_traveled <= 500:
        mileage = 100 * 0.3 + (miles_traveled - 100) * 0.4
    else:
        mileage = 100 * 0.3 + 400 * 0.4 + (miles_traveled - 500) * 0.5
    
    receipt_bonus = total_receipts_amount * 1.15
    
    return round(base + mileage + receipt_bonus, 2)""",
            
            "Expense-based Multiplier": """def calculate_reimbursement(trip_duration_days, miles_traveled, total_receipts_amount):
    daily_allowance = 45 * trip_duration_days
    travel_cost = miles_traveled * 0.45
    
    if total_receipts_amount > 200:
        multiplier = 1.25
    elif total_receipts_amount > 100:
        multiplier = 1.15
    else:
        multiplier = 1.0
    
    return round((daily_allowance + travel_cost + total_receipts_amount) * multiplier, 2)""",
            
            "Complex Business Rules": """def calculate_reimbursement(trip_duration_days, miles_traveled, total_receipts_amount):
    # Base daily allowance
    base = 35 * trip_duration_days
    
    # Mileage with weekend bonus
    if trip_duration_days > 5:  # Assuming weekend travel
        mileage_rate = 0.55
    else:
        mileage_rate = 0.45
    
    mileage = miles_traveled * mileage_rate
    
    # Receipt reimbursement with caps
    receipt_reimb = min(total_receipts_amount * 1.2, trip_duration_days * 150)
    
    # Minimum guarantee
    total = max(base + mileage + receipt_reimb, trip_duration_days * 75)
    
    return round(total, 2)"""
        }
        
        st.code(templates[template], language='python')

def show_documentation():
    """Show project documentation"""
    st.header("📚 Documentation")
    
    tab1, tab2 = st.tabs(["PRD", "Employee Interviews"])
    
    with tab1:
        st.subheader("Product Requirements Document")
        prd_content = load_documentation("PRD.md")
        st.markdown(prd_content)
    
    with tab2:
        st.subheader("Employee Interview Notes")
        interviews_content = load_documentation("INTERVIEWS.md")
        st.markdown(interviews_content)

def show_export_results():
    """Show export functionality"""
    st.header("📤 Export Results")
    
    st.subheader("💾 Export Analysis Results")
    
    if st.button("Generate Analysis Report"):
        # Generate comprehensive report
        analyzer = st.session_state.analyzer
        pattern_detector = st.session_state.pattern_detector
        
        report = {
            "summary_statistics": analyzer.df.describe().to_dict(),
            "correlation_matrix": analyzer.get_correlation_matrix().to_dict(),
            "patterns_detected": pattern_detector.detect_patterns() if pattern_detector else {},
            "data_insights": {
                "total_records": len(analyzer.df),
                "average_reimbursement": float(analyzer.df['reimbursement_amount'].mean()),
                "reimbursement_range": [
                    float(analyzer.df['reimbursement_amount'].min()),
                    float(analyzer.df['reimbursement_amount'].max())
                ]
            }
        }
        
        # Convert to JSON for download
        report_json = json.dumps(report, indent=2)
        
        st.download_button(
            label="Download Analysis Report (JSON)",
            data=report_json,
            file_name="reimbursement_analysis_report.json",
            mime="application/json"
        )
    
    st.subheader("🔧 Export Algorithm Code")
    
    final_algorithm = st.text_area(
        "Final Algorithm Code:",
        value="""def calculate_reimbursement(trip_duration_days, miles_traveled, total_receipts_amount):
    # Your reverse-engineered algorithm here
    # Based on pattern analysis and testing
    
    # Example implementation
    base_rate = 42.5 * trip_duration_days
    mileage_compensation = 0.48 * miles_traveled
    receipt_reimbursement = 1.18 * total_receipts_amount
    
    total = base_rate + mileage_compensation + receipt_reimbursement
    
    return round(total, 2)""",
        height=300
    )
    
    if st.button("Export Algorithm"):
        st.download_button(
            label="Download run.sh Script",
            data=f"""#!/bin/bash

# ACME Corp Legacy Reimbursement System Replica
# Generated by Legacy System Analyzer

python3 << 'EOF'
{final_algorithm}

import sys
if len(sys.argv) != 4:
    print("Usage: ./run.sh trip_duration_days miles_traveled total_receipts_amount")
    sys.exit(1)

trip_duration_days = int(sys.argv[1])
miles_traveled = int(sys.argv[2])
total_receipts_amount = float(sys.argv[3])

result = calculate_reimbursement(trip_duration_days, miles_traveled, total_receipts_amount)
print(result)
EOF""",
            file_name="run.sh",
            mime="text/plain"
        )

if __name__ == "__main__":
    main()
