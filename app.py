import dash
from dash import dcc, html, Input, Output, State, dash_table
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import time
import random
import argparse
import numpy_financial as npf
import os

# Import the cruise model
from simple_cruise_model import (
    run_simulation, 
    run_simulation_batch, 
    create_default_state_configs,
    calculate_summary_metrics, 
    print_simulation_summary,
    calculate_monthly_irr
)
from simulation_config import (
    StateConfig, 
    SimulationConfig, 
    BASELINE_CONFIG, 
    OPTIMISTIC_CONFIG, 
    PESSIMISTIC_CONFIG,
    DEFAULT_CONFIG
)

# Modify the default simulation configs to match our new values
DEFAULT_CONFIG.basic_training_cost = 2500
DEFAULT_CONFIG.advanced_training_dropout_rate = 0.0
DEFAULT_CONFIG.disney_cruise_salary_variation = 1.0
DEFAULT_CONFIG.costa_cruise_salary_variation = 1.0
DEFAULT_CONFIG.disney_cruise_dropout_rate = 0.0  # Set Disney cruise dropout rate to 0
DEFAULT_CONFIG.costa_cruise_dropout_rate = 0.0   # Set Costa cruise dropout rate to 0

# Apply these changes to our preset scenarios as well
BASELINE_CONFIG.basic_training_cost = 2500
BASELINE_CONFIG.advanced_training_dropout_rate = 0.0
BASELINE_CONFIG.disney_cruise_salary_variation = 1.0
BASELINE_CONFIG.costa_cruise_salary_variation = 1.0
BASELINE_CONFIG.disney_cruise_dropout_rate = 0.0  # Set Disney cruise dropout rate to 0
BASELINE_CONFIG.costa_cruise_dropout_rate = 0.0   # Set Costa cruise dropout rate to 0

OPTIMISTIC_CONFIG.basic_training_cost = 2500
OPTIMISTIC_CONFIG.advanced_training_dropout_rate = 0.0
OPTIMISTIC_CONFIG.disney_cruise_salary_variation = 1.0
OPTIMISTIC_CONFIG.costa_cruise_salary_variation = 1.0
OPTIMISTIC_CONFIG.disney_cruise_dropout_rate = 0.0  # Set Disney cruise dropout rate to 0
OPTIMISTIC_CONFIG.costa_cruise_dropout_rate = 0.0   # Set Costa cruise dropout rate to 0

PESSIMISTIC_CONFIG.basic_training_cost = 2500
PESSIMISTIC_CONFIG.advanced_training_dropout_rate = 0.0
PESSIMISTIC_CONFIG.disney_cruise_salary_variation = 1.0
PESSIMISTIC_CONFIG.costa_cruise_salary_variation = 1.0
PESSIMISTIC_CONFIG.disney_cruise_dropout_rate = 0.0  # Set Disney cruise dropout rate to 0
PESSIMISTIC_CONFIG.costa_cruise_dropout_rate = 0.0   # Set Costa cruise dropout rate to 0

# Initialize the Dash app with production config
app = dash.Dash(
    __name__,
    suppress_callback_exceptions=True,
    update_title=None,
    routes_pathname_prefix='/',
    requests_pathname_prefix='/',
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1"}
    ]
)

# Set server timeout (optional, adjust as needed)
server = app.server  # Expose the server variable for production
server.config.update({
    'SEND_FILE_MAX_AGE_DEFAULT': 0,
    'TEMPLATES_AUTO_RELOAD': True
})

# Enable the app to be embedded in an iframe
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>Cruise Career Analysis Tool</title>
        {%favicon%}
        {%css%}
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# Define the preset scenarios
preset_scenarios = {
    'baseline': {
        'name': 'Baseline',
        'description': 'Moderate assumptions about cruise career progression.',
        'config': BASELINE_CONFIG
    },
    'optimistic': {
        'name': 'Optimistic',
        'description': 'Favorable conditions with higher salaries and lower dropout rates.',
        'config': OPTIMISTIC_CONFIG
    },
    'pessimistic': {
        'name': 'Pessimistic',
        'description': 'Challenging conditions with lower salaries and higher dropout rates.',
        'config': PESSIMISTIC_CONFIG
    }
}

# Dropdown options for number of students
student_options = [
    {'label': '10 simulations', 'value': 10},
    {'label': '50 simulations', 'value': 50},
    {'label': '100 simulations', 'value': 100},
    {'label': '200 simulations', 'value': 200},
    {'label': '500 simulations', 'value': 500}
]

# Dropdown options for number of simulations
sim_options = [
    {'label': '10 Monte Carlo runs (faster)', 'value': 10},
    {'label': '50 Monte Carlo runs', 'value': 50},
    {'label': '100 Monte Carlo runs (recommended)', 'value': 100}
]

# Define the layout of the app
app.layout = html.Div([
    html.H1("Cruise Career Analysis Tool", style={'textAlign': 'center', 'marginBottom': '30px'}),
    
    # Tabs for different sections
    dcc.Tabs([
        dcc.Tab(label='About', children=[
            html.Div([
                html.H1("Cruise Career Analysis Tool", style={'textAlign': 'center', 'marginBottom': '30px', 'color': '#2c3e50'}),
                
                # Background Section
                html.Div([
                    html.H2("Background", style={'color': '#2c3e50', 'borderBottom': '1px solid #eee', 'paddingBottom': '10px'}),
                    html.P([
                        "This tool simulates career paths for crew members through a sequence of training and cruise assignments, ",
                        "analyzing financial outcomes including training costs, payments, and return on investment."
                    ], style={'fontSize': '16px', 'lineHeight': '1.6'}),
                ], style={'marginBottom': '30px'}),
                
                # Objectives Section
                html.Div([
                    html.H2("Objectives", style={'color': '#2c3e50', 'borderBottom': '1px solid #eee', 'paddingBottom': '10px'}),
                    html.P([
                        "This simulation tool helps stakeholders understand the financial outcomes of cruise careers across various scenarios. The tool specifically aims to:"
                    ], style={'fontSize': '16px', 'lineHeight': '1.6'}),
                    html.Ul([
                        html.Li("Model expected returns on training investments across different career paths", style={'fontSize': '16px', 'lineHeight': '1.6'}),
                        html.Li("Simulate how dropout rates and salary variations affect career trajectories", style={'fontSize': '16px', 'lineHeight': '1.6'}),
                        html.Li("Provide program designers with data-driven insights for career structure", style={'fontSize': '16px', 'lineHeight': '1.6'}),
                        html.Li("Analyze the impact of different payment structures on return on investment", style={'fontSize': '16px', 'lineHeight': '1.6'})
                    ], style={'paddingLeft': '30px'})
                ], style={'marginBottom': '30px'}),
                
                # Implementation Section
                html.Div([
                    html.H2("Implementation", style={'color': '#2c3e50', 'borderBottom': '1px solid #eee', 'paddingBottom': '10px'}),
                    html.P([
                        "This interactive dashboard allows users to:"
                    ], style={'fontSize': '16px', 'lineHeight': '1.6'}),
                    html.Ul([
                        html.Li("Select preset scenarios or customize career parameters", style={'fontSize': '16px', 'lineHeight': '1.6'}),
                        html.Li("Adjust training costs, dropout rates, and salary progression", style={'fontSize': '16px', 'lineHeight': '1.6'}),
                        html.Li("Customize payment percentages and cruise durations", style={'fontSize': '16px', 'lineHeight': '1.6'}),
                        html.Li("Run Monte Carlo simulations to test robustness", style={'fontSize': '16px', 'lineHeight': '1.6'}),
                        html.Li("Compare multiple scenarios to identify optimal career structures", style={'fontSize': '16px', 'lineHeight': '1.6'})
                    ], style={'paddingLeft': '30px'})
                ], style={'marginBottom': '30px'})
            ], style={'padding': '20px', 'maxWidth': '1200px', 'margin': '0 auto'})
        ]),
        
        dcc.Tab(label='Simulation', children=[
            html.Div([
                html.H2("Cruise Career Simulation Dashboard", style={'textAlign': 'center', 'marginBottom': '30px'}),
                
                html.Div([
                    # Left panel for inputs
                    html.Div([
                        html.H3("Simulation Parameters", style={'marginBottom': '20px'}),
                        
                        # Training parameters
                        html.Div([
                            html.H4("Training Parameters", style={'marginBottom': '15px'}),
                            
                            html.Div([
                                html.Label("Include Transportation and Placement:"),
                                dcc.RadioItems(
                                    id="include-advanced-training",
                                    options=[
                                        {'label': 'Yes', 'value': True},
                                        {'label': 'No', 'value': False}
                                    ],
                                    value=True,
                                    inline=True
                                )
                            ], style={'marginBottom': '15px'}),
                            
                            html.Div([
                                html.Label("Training Cost ($):"),
                                dcc.Input(
                                    id="basic-training-cost",
                                    type="number",
                                    value=2500,
                                    min=0,
                                    step=100,
                                    style={"width": "100%"}
                                )
                            ], style={'marginBottom': '15px'}),
                            
                            html.Div([
                                html.Label("Training Failure Rate (%):"),
                                dcc.Slider(
                                    id="basic-training-dropout-rate",
                                    min=0,
                                    max=50,
                                    step=1,
                                    value=10,
                                    marks={i: f'{i}%' for i in range(0, 51, 10)},
                                )
                            ], style={'marginBottom': '15px'}),
                            
                            html.Div([
                                html.Label("Training Duration (months):"),
                                dcc.Input(
                                    id="basic-training-duration",
                                    type="number",
                                    value=6,
                                    min=1,
                                    max=24,
                                    step=1,
                                    style={"width": "100%"}
                                )
                            ], style={'marginBottom': '15px'}),
                            
                            # New Offer Stage Parameters
                            html.Div([
                                html.H5("Offer Stage Parameters", style={'marginBottom': '10px', 'color': '#ff9800'}),
                                
                                html.Div([
                                    html.Label("Include Offer Stage:"),
                                    dcc.RadioItems(
                                        id="include-offer-stage",
                                        options=[
                                            {'label': 'Yes', 'value': True},
                                            {'label': 'No', 'value': False}
                                        ],
                                        value=True,
                                        inline=True
                                    )
                                ], style={'marginBottom': '15px'}),
                                
                                html.Div([
                                    html.Label("No Offer Rate (%):"),
                                    dcc.Slider(
                                        id="no-offer-rate",
                                        min=0,
                                        max=50,
                                        step=1,
                                        value=30,
                                        marks={i: f'{i}%' for i in range(0, 51, 10)},
                                    )
                                ], style={'marginBottom': '15px'}),
                                
                                html.Div([
                                    html.Label("Offer Stage Duration (months):"),
                                    dcc.Input(
                                        id="offer-stage-duration",
                                        type="number",
                                        value=1,
                                        min=1,
                                        max=12,
                                        step=1,
                                        style={"width": "100%"}
                                    )
                                ], style={'marginBottom': '15px'})
                            ], style={'marginBottom': '15px', 'backgroundColor': '#fff3e0', 'padding': '15px', 'borderRadius': '5px'}),
                            
                            # Add the transportation and placement parameters directly in the layout instead of using conditional rendering
                            html.Div([
                                html.Div([
                                    html.Label("Transportation and Placement Cost ($):"),
                                    dcc.Input(
                                        id="advanced-training-cost",
                                        type="number",
                                        value=500,
                                        min=0,
                                        step=100,
                                        style={"width": "100%"}
                                    )
                                ], style={'marginBottom': '15px'}),
                                
                                html.Div([
                                    html.Label("Transportation and Placement Dropout Rate (%):"),
                                    dcc.Slider(
                                        id="advanced-training-dropout-rate",
                                        min=0,
                                        max=50,
                                        step=1,
                                        value=0,
                                        marks={i: f'{i}%' for i in range(0, 51, 10)},
                                    )
                                ], style={'marginBottom': '15px'}),
                                
                                html.Div([
                                    html.Label("Transportation and Placement Duration (months):"),
                                    dcc.Input(
                                        id="advanced-training-duration",
                                        type="number",
                                        value=5,
                                        min=1,
                                        max=24,
                                        step=1,
                                        style={"width": "100%"}
                                    )
                                ], style={'marginBottom': '15px'})
                            ], id="advanced-training-container", style={'marginBottom': '15px', 'backgroundColor': '#e6f7ff', 'padding': '15px', 'borderRadius': '5px'}),
                            
                            # New Early Termination Stage Parameters
                            html.Div([
                                html.H5("Early Termination Parameters", style={'marginBottom': '10px', 'color': '#f44336'}),
                                
                                html.Div([
                                    html.Label("Include Early Termination Stage:"),
                                    dcc.RadioItems(
                                        id="include-early-termination",
                                        options=[
                                            {'label': 'Yes', 'value': True},
                                            {'label': 'No', 'value': False}
                                        ],
                                        value=True,
                                        inline=True
                                    )
                                ], style={'marginBottom': '15px'}),
                                
                                html.Div([
                                    html.Label("Early Termination Rate (%):"),
                                    dcc.Slider(
                                        id="early-termination-rate",
                                        min=0,
                                        max=50,
                                        step=1,
                                        value=10,
                                        marks={i: f'{i}%' for i in range(0, 51, 10)},
                                    )
                                ], style={'marginBottom': '15px'}),
                                
                                html.Div([
                                    html.Label("Early Termination Stage Duration (months):"),
                                    dcc.Input(
                                        id="early-termination-duration",
                                        type="number",
                                        value=1,
                                        min=1,
                                        max=12,
                                        step=1,
                                        style={"width": "100%"}
                                    )
                                ], style={'marginBottom': '15px'})
                            ], style={'marginBottom': '15px', 'backgroundColor': '#ffebee', 'padding': '15px', 'borderRadius': '5px'})
                        ], style={'marginBottom': '20px', 'backgroundColor': '#f1f1f1', 'padding': '15px', 'borderRadius': '5px'}),
                        
                        # Cruise parameters
                        html.Div([
                            html.H4("Provider Settings", style={'marginBottom': '15px'}),
                            
                            html.Div([
                                html.Label("Disney Allocation (%):"),
                                dcc.Slider(
                                    id="disney-allocation-pct",
                                    min=0,
                                    max=100,
                                    step=5,
                                    value=30,
                                    marks={i: f'{i}%' for i in range(0, 101, 20)},
                                )
                            ], style={'marginBottom': '15px'}),
                            
                            html.Div([
                                html.Label("Costa Allocation (%):"),
                                dcc.Slider(
                                    id="costa-allocation-pct",
                                    min=0,
                                    max=100,
                                    step=5,
                                    value=70,
                                    marks={i: f'{i}%' for i in range(0, 101, 20)},
                                )
                            ], style={'marginBottom': '15px'}),
                            
                            html.Div([
                                html.Label("Number of Cruises:"),
                                dcc.Input(
                                    id="num-cruises",
                                    type="number",
                                    value=3,
                                    min=1,
                                    max=5,
                                    step=1,
                                    style={"width": "100%"}
                                )
                            ], style={'marginBottom': '15px'}),
                            
                            html.Hr(),
                            
                            # Disney Cruise Parameters
                            html.H5("Disney Cruise Parameters", style={'marginBottom': '15px', 'color': '#2196F3'}),
                            
                            html.Div([
                                html.Label("Disney First Cruise Salary ($):"),
                                dcc.Input(
                                    id="disney-first-cruise-salary",
                                    type="number",
                                    value=5100,
                                    min=0,
                                    step=100,
                                    style={"width": "100%"}
                                )
                            ], style={'marginBottom': '15px'}),
                            
                            html.Div([
                                html.Label("Disney Second Cruise Salary ($):"),
                                dcc.Input(
                                    id="disney-second-cruise-salary",
                                    type="number",
                                    value=5400,
                                    min=0,
                                    step=100,
                                    style={"width": "100%"}
                                )
                            ], style={'marginBottom': '15px'}),
                            
                            html.Div([
                                html.Label("Disney Third Cruise Salary ($):"),
                                dcc.Input(
                                    id="disney-third-cruise-salary",
                                    type="number",
                                    value=18000,
                                    min=0,
                                    step=100,
                                    style={"width": "100%"}
                                )
                            ], style={'marginBottom': '15px'}),
                            
                            html.Div([
                                html.Label("Disney Cruise Duration (months):"),
                                dcc.Input(
                                    id="disney-cruise-duration",
                                    type="number",
                                    value=6,
                                    min=1,
                                    max=24,
                                    step=1,
                                    style={"width": "100%"}
                                )
                            ], style={'marginBottom': '15px'}),
                            
                            html.Div([
                                html.Label("Disney Cruise Dropout Rate (%):"),
                                dcc.Slider(
                                    id="disney-cruise-dropout-rate",
                                    min=0,
                                    max=20,
                                    step=0.5,
                                    value=3,
                                    marks={i: f'{i}%' for i in range(0, 21, 5)},
                                )
                            ], style={'marginBottom': '15px'}),
                            
                            html.Div([
                                html.Label("Disney Cruise Salary Variation (%):"),
                                dcc.Slider(
                                    id="disney-cruise-salary-variation",
                                    min=0,
                                    max=20,
                                    step=0.5,
                                    value=1,
                                    marks={i: f'{i}%' for i in range(0, 21, 5)},
                                )
                            ], style={'marginBottom': '15px'}),
                            
                            html.Div([
                                html.Label("Disney Cruise Payment Fraction (%):"),
                                dcc.Slider(
                                    id="disney-cruise-payment-fraction",
                                    min=0,
                                    max=30,
                                    step=0.5,
                                    value=14,
                                    marks={i: f'{i}%' for i in range(0, 31, 5)},
                                )
                            ], style={'marginBottom': '15px'}),
                            
                            html.Hr(),
                            
                            # Costa Cruise Parameters
                            html.H5("Costa Cruise Parameters", style={'marginBottom': '15px', 'color': '#4CAF50'}),
                            
                            html.Div([
                                html.Label("Costa First Cruise Salary ($):"),
                                dcc.Input(
                                    id="costa-first-cruise-salary",
                                    type="number",
                                    value=5100,
                                    min=0,
                                    step=100,
                                    style={"width": "100%"}
                                )
                            ], style={'marginBottom': '15px'}),
                            
                            html.Div([
                                html.Label("Costa Second Cruise Salary ($):"),
                                dcc.Input(
                                    id="costa-second-cruise-salary",
                                    type="number",
                                    value=5850,
                                    min=0,
                                    step=100,
                                    style={"width": "100%"}
                                )
                            ], style={'marginBottom': '15px'}),
                            
                            html.Div([
                                html.Label("Costa Third Cruise Salary ($):"),
                                dcc.Input(
                                    id="costa-third-cruise-salary",
                                    type="number",
                                    value=9000,
                                    min=0,
                                    step=100,
                                    style={"width": "100%"}
                                )
                            ], style={'marginBottom': '15px'}),
                            
                            html.Div([
                                html.Label("Costa Cruise Duration (months):"),
                                dcc.Input(
                                    id="costa-cruise-duration",
                                    type="number",
                                    value=7,
                                    min=1,
                                    max=24,
                                    step=1,
                                    style={"width": "100%"}
                                )
                            ], style={'marginBottom': '15px'}),
                            
                            html.Div([
                                html.Label("Costa Cruise Dropout Rate (%):"),
                                dcc.Slider(
                                    id="costa-cruise-dropout-rate",
                                    min=0,
                                    max=20,
                                    step=0.5,
                                    value=3,
                                    marks={i: f'{i}%' for i in range(0, 21, 5)},
                                )
                            ], style={'marginBottom': '15px'}),
                            
                            html.Div([
                                html.Label("Costa Cruise Salary Variation (%):"),
                                dcc.Slider(
                                    id="costa-cruise-salary-variation",
                                    min=0,
                                    max=20,
                                    step=0.5,
                                    value=1,
                                    marks={i: f'{i}%' for i in range(0, 21, 5)},
                                )
                            ], style={'marginBottom': '15px'}),
                            
                            html.Div([
                                html.Label("Costa Cruise Payment Fraction (%):"),
                                dcc.Slider(
                                    id="costa-cruise-payment-fraction",
                                    min=0,
                                    max=30,
                                    step=0.5,
                                    value=14,
                                    marks={i: f'{i}%' for i in range(0, 31, 5)},
                                )
                            ], style={'marginBottom': '15px'}),
                        ], style={'marginBottom': '20px', 'backgroundColor': '#f1f1f1', 'padding': '15px', 'borderRadius': '5px'}),
                        
                        # Break parameters
                        html.Div([
                            html.H4("Break Parameters", style={'marginBottom': '15px'}),
                            
                            html.Div([
                                html.Label("Include Breaks Between Cruises:"),
                                dcc.RadioItems(
                                    id="include-breaks",
                                    options=[
                                        {'label': 'Yes', 'value': True},
                                        {'label': 'No', 'value': False}
                                    ],
                                    value=True,
                                    inline=True
                                )
                            ], style={'marginBottom': '15px'}),
                            
                            html.Div([
                                html.Label("Break Duration (months):"),
                                dcc.Input(
                                    id="break-duration",
                                    type="number",
                                    value=2,
                                    min=1,
                                    max=12,
                                    step=1,
                                    style={"width": "100%"}
                                )
                            ], style={'marginBottom': '15px'}),
                            
                            html.Div([
                                html.Label("Break Dropout Rate (%):"),
                                dcc.Slider(
                                    id="break-dropout-rate",
                                    min=0,
                                    max=20,
                                    step=0.5,
                                    value=0,
                                    marks={i: f'{i}%' for i in range(0, 21, 5)},
                                )
                            ], style={'marginBottom': '15px'}),
                            
                            html.P("Breaks represent time between cruises with no salary or payments.",
                                  style={'fontSize': '0.85em', 'fontStyle': 'italic', 'color': '#666'})
                        ], style={'marginBottom': '20px', 'backgroundColor': '#f1f1f1', 'padding': '15px', 'borderRadius': '5px'}),
                        
                        # Preset scenarios selector
                        html.Div([
                            html.Label("Preset Scenarios:", style={'fontWeight': 'bold', 'fontSize': '16px'}),
                            html.P("Select a pre-configured scenario to automatically set career parameters", 
                                 style={'fontSize': '0.85em', 'margin': '2px 0 10px 0'}),
                            dcc.Dropdown(
                                id="preset-scenario",
                                options=[{'label': preset_scenarios[k]['name'], 'value': k} for k in preset_scenarios.keys()],
                                value="baseline",
                                placeholder="Select a preset scenario",
                                style={'fontWeight': 'bold'}
                            ),
                            html.Div(id="preset-description", style={'color': '#666', 'fontSize': '0.9em', 'marginTop': '5px', 'fontStyle': 'italic'})
                        ], style={'marginBottom': '20px', 'backgroundColor': '#e6f7ff', 'padding': '15px', 'borderRadius': '5px'}),
                        
                        # Basic settings
                        html.Div([
                            html.H4("Basic Settings", style={'marginBottom': '15px'}),
                            
                            html.Div([
                                html.Label("Number of Monte Carlo Runs:"),
                                dcc.Dropdown(
                                    id="num-sims",
                                    options=sim_options,
                                    value=50
                                ),
                                html.P("Each run uses the same parameters but with different random values for salary variation, dropout chance, etc.",
                                      style={'fontSize': '0.8em', 'color': '#666', 'marginTop': '5px'})
                            ], style={'marginBottom': '15px'}),
                            
                            html.Div([
                                html.Label("Number of Career Simulations:"),
                                dcc.Dropdown(
                                    id="num-students",
                                    options=student_options,
                                    value=100
                                ),
                                html.P("Each simulation follows one student through their career path. Results are aggregated across all simulations.",
                                      style={'fontSize': '0.8em', 'color': '#666', 'marginTop': '5px'})
                            ], style={'marginBottom': '15px'}),
                            
                            # Run Simulation Button
                            html.Div([
                                html.Button(
                                    "Run Simulation", 
                                    id="run-simulation", 
                                    n_clicks=0,
                                    style={
                                        'backgroundColor': '#4CAF50',
                                        'color': 'white',
                                        'padding': '10px 20px',
                                        'fontSize': '16px',
                                        'fontWeight': 'bold',
                                        'border': 'none',
                                        'borderRadius': '4px',
                                        'cursor': 'pointer',
                                        'width': '100%'
                                    }
                                ),
                                html.Div(id="loading-message", style={'marginTop': '10px', 'textAlign': 'center'})
                            ], style={'marginBottom': '20px'})
                        ], style={'marginBottom': '20px', 'backgroundColor': '#f1f1f1', 'padding': '15px', 'borderRadius': '5px'})
                    ], style={'width': '35%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '20px', 'boxShadow': '0 4px 8px 0 rgba(0,0,0,0.2)', 'backgroundColor': '#f9f9f9', 'borderRadius': '8px'}),
                    
                    # Right panel for results
                    html.Div([
                        html.H3("Results"),
                        
                        html.Div([
                            html.Div(id="summary-stats", style={'marginBottom': '20px'}),
                            
                            dcc.Tabs([
                                dcc.Tab(label='Overview', children=[
                                    html.Div(id="overview-content")
                                ]),
                                dcc.Tab(label='Scenario Comparison', children=[
                                    html.Div([
                                        html.H4("Compare Saved Scenarios", style={'marginBottom': '15px'}),
                                        html.P("Save multiple scenarios and compare their results side by side."),
                                        
                                        html.Div([
                                            html.Div([
                                                html.Label("Scenario Name:"),
                                                dcc.Input(
                                                    id="scenario-name-input",
                                                    type="text",
                                                    placeholder="Enter a name for this scenario",
                                                    style={'width': '100%'}
                                                )
                                            ], style={'width': '60%', 'display': 'inline-block'}),
                                            
                                            html.Div([
                                                html.Button(
                                                    "Save Current Scenario", 
                                                    id="save-scenario-button", 
                                                    n_clicks=0,
                                                    style={
                                                        'backgroundColor': '#4CAF50',
                                                        'color': 'white',
                                                        'padding': '10px',
                                                        'borderRadius': '5px',
                                                        'border': 'none',
                                                        'width': '100%',
                                                        'cursor': 'pointer'
                                                    }
                                                )
                                            ], style={'width': '35%', 'display': 'inline-block', 'float': 'right'})
                                        ], style={'marginBottom': '20px'}),
                                        
                                        html.Div([
                                            html.H5("Saved Scenarios", style={'marginBottom': '10px'}),
                                            html.Div(id="saved-scenarios-list"),
                                            html.Div([
                                                html.Button(
                                                    "Compare Selected Scenarios", 
                                                    id="compare-scenarios-button", 
                                                    n_clicks=0,
                                                    style={
                                                        'backgroundColor': '#2196F3',
                                                        'color': 'white',
                                                        'padding': '10px',
                                                        'borderRadius': '5px',
                                                        'border': 'none',
                                                        'marginRight': '10px',
                                                        'cursor': 'pointer'
                                                    }
                                                ),
                                                html.Button(
                                                    "Clear All Scenarios", 
                                                    id="clear-scenarios-button", 
                                                    n_clicks=0,
                                                    style={
                                                        'backgroundColor': '#f44336',
                                                        'color': 'white',
                                                        'padding': '10px',
                                                        'borderRadius': '5px',
                                                        'border': 'none',
                                                        'cursor': 'pointer'
                                                    }
                                                )
                                            ], style={'marginTop': '15px', 'marginBottom': '20px'})
                                        ], style={'marginBottom': '20px'}),
                                        
                                        html.Div(id="scenario-comparison-results")
                                    ])
                                ]),
                            ], style={'marginTop': '20px'})
                        ])
                    ], style={'width': '65%', 'display': 'inline-block', 'padding': '20px', 'boxShadow': '0 4px 8px 0 rgba(0,0,0,0.2)', 'backgroundColor': '#f9f9f9', 'borderRadius': '8px'})
                ], style={'display': 'flex', 'justifyContent': 'space-between', 'gap': '20px', 'marginBottom': '30px'}),
            ])
        ])
    ]),
    
    # Data stores
    dcc.Store(id='simulation-results-store'),
    dcc.Store(id='simulation-config-store'),
    dcc.Store(id='saved-scenarios-store', data={})
])

# Callback to update preset description
@app.callback(
    Output("preset-description", "children"),
    [Input("preset-scenario", "value")]
)
def update_preset_description(preset_name):
    if preset_name and preset_name in preset_scenarios:
        return preset_scenarios[preset_name]['description']
    return ""

# Callback to show/hide advanced training container
@app.callback(
    Output("advanced-training-container", "style"),
    [Input("include-advanced-training", "value")]
)
def toggle_advanced_training_visibility(include_advanced):
    if include_advanced:
        return {'display': 'block', 'marginBottom': '15px', 'backgroundColor': '#e6f7ff', 'padding': '15px', 'borderRadius': '5px'}
    else:
        return {'display': 'none'}

# Callback to show/hide offer stage container
@app.callback(
    [Output("no-offer-rate", "disabled"),
     Output("offer-stage-duration", "disabled")],
    [Input("include-offer-stage", "value")]
)
def toggle_offer_stage_controls(include_offer):
    if include_offer:
        return False, False
    else:
        return True, True

# Callback to show/hide early termination container
@app.callback(
    [Output("early-termination-rate", "disabled"),
     Output("early-termination-duration", "disabled")],
    [Input("include-early-termination", "value")]
)
def toggle_early_termination_controls(include_early_termination):
    if include_early_termination:
        return False, False
    else:
        return True, True

# Callback to update all parameters when a preset is selected
@app.callback(
    [
        # Training parameters
        Output("include-advanced-training", "value"),
        Output("basic-training-cost", "value"),
        Output("basic-training-dropout-rate", "value"),
        Output("basic-training-duration", "value"),
        
        # Offer stage parameters
        Output("include-offer-stage", "value"),
        Output("no-offer-rate", "value"),
        Output("offer-stage-duration", "value"),
        
        # Transportation and placement parameters
        Output("advanced-training-cost", "value"),
        Output("advanced-training-dropout-rate", "value"),
        Output("advanced-training-duration", "value"),
        
        # Early termination parameters
        Output("include-early-termination", "value"),
        Output("early-termination-rate", "value"),
        Output("early-termination-duration", "value"),
        
        # Provider parameters
        Output("disney-allocation-pct", "value"),
        Output("costa-allocation-pct", "value"),
        Output("num-cruises", "value"),
        
        # Disney parameters
        Output("disney-first-cruise-salary", "value"),
        Output("disney-second-cruise-salary", "value"),
        Output("disney-third-cruise-salary", "value"),
        Output("disney-cruise-duration", "value"),
        Output("disney-cruise-dropout-rate", "value"),
        Output("disney-cruise-salary-variation", "value"),
        Output("disney-cruise-payment-fraction", "value"),
        
        # Costa parameters
        Output("costa-first-cruise-salary", "value"),
        Output("costa-second-cruise-salary", "value"),
        Output("costa-third-cruise-salary", "value"),
        Output("costa-cruise-duration", "value"),
        Output("costa-cruise-dropout-rate", "value"),
        Output("costa-cruise-salary-variation", "value"),
        Output("costa-cruise-payment-fraction", "value"),
        
        # Break parameters
        Output("include-breaks", "value"),
        Output("break-duration", "value"),
        Output("break-dropout-rate", "value"),
    ],
    [Input("preset-scenario", "value")]
)
def update_from_preset(preset_name):
    if preset_name is None or preset_name not in preset_scenarios:
        preset_name = 'baseline'
    
    # Get the config from the preset
    config = preset_scenarios[preset_name]['config']
    
    # Return all values from the config
    return (
        config.include_advanced_training,
        2500,  # Changed basic_training_cost to 2500
        config.basic_training_dropout_rate * 100,
        config.basic_training_duration,
        
        # Offer stage parameters
        config.include_offer_stage,
        config.no_offer_rate * 100,
        config.offer_stage_duration,
        
        # Transportation and placement parameters
        config.advanced_training_cost,
        0,  # Changed advanced_training_dropout_rate to 0%
        config.advanced_training_duration,
        
        # Early termination parameters
        config.include_early_termination,
        config.early_termination_rate * 100,
        config.early_termination_duration,
        
        # Provider settings
        config.disney_allocation_pct,
        config.costa_allocation_pct,
        config.num_cruises,
        
        # Disney cruise settings
        config.disney_first_cruise_salary,
        config.disney_second_cruise_salary,
        config.disney_third_cruise_salary,
        config.disney_cruise_duration,
        config.disney_cruise_dropout_rate * 100,
        1,  # Changed disney_cruise_salary_variation to 1%
        config.disney_cruise_payment_fraction * 100,
        
        # Costa cruise settings
        config.costa_first_cruise_salary,
        config.costa_second_cruise_salary,
        config.costa_third_cruise_salary,
        config.costa_cruise_duration,
        config.costa_cruise_dropout_rate * 100,
        1,  # Changed costa_cruise_salary_variation to 1%
        config.costa_cruise_payment_fraction * 100,
        
        # Break settings
        config.include_breaks,
        config.break_duration,
        config.break_dropout_rate * 100
    )

# Callback to create a simulation configuration and store it
@app.callback(
    Output("simulation-config-store", "data"),
    [
        # Training parameters
        Input("include-advanced-training", "value"),
        Input("basic-training-cost", "value"),
        Input("basic-training-dropout-rate", "value"),
        Input("basic-training-duration", "value"),
        
        # Offer stage parameters
        Input("include-offer-stage", "value"),
        Input("no-offer-rate", "value"),
        Input("offer-stage-duration", "value"),
        
        # Transportation and placement parameters
        Input("advanced-training-cost", "value"),
        Input("advanced-training-dropout-rate", "value"),
        Input("advanced-training-duration", "value"),
        
        # Early termination parameters
        Input("include-early-termination", "value"),
        Input("early-termination-rate", "value"),
        Input("early-termination-duration", "value"),
        
        # Provider settings
        Input("disney-allocation-pct", "value"),
        Input("costa-allocation-pct", "value"),
        Input("num-cruises", "value"),
        
        # Disney cruise settings
        Input("disney-first-cruise-salary", "value"),
        Input("disney-second-cruise-salary", "value"),
        Input("disney-third-cruise-salary", "value"),
        Input("disney-cruise-duration", "value"),
        Input("disney-cruise-dropout-rate", "value"),
        Input("disney-cruise-salary-variation", "value"),
        Input("disney-cruise-payment-fraction", "value"),
        
        # Costa cruise settings
        Input("costa-first-cruise-salary", "value"),
        Input("costa-second-cruise-salary", "value"),
        Input("costa-third-cruise-salary", "value"),
        Input("costa-cruise-duration", "value"),
        Input("costa-cruise-dropout-rate", "value"),
        Input("costa-cruise-salary-variation", "value"),
        Input("costa-cruise-payment-fraction", "value"),
        
        # Break settings
        Input("include-breaks", "value"),
        Input("break-duration", "value"),
        Input("break-dropout-rate", "value"),
        
        # Simulation settings
        Input("num-students", "value"),
        Input("num-sims", "value")
    ]
)
def update_simulation_config(
    # Training parameters
    include_advanced_training, 
    basic_training_cost, 
    basic_training_dropout_rate,
    basic_training_duration,
    
    # Offer stage parameters
    include_offer_stage,
    no_offer_rate,
    offer_stage_duration,
    
    # Transportation and placement parameters
    advanced_training_cost,
    advanced_training_dropout_rate,
    advanced_training_duration,
    
    # Early termination parameters
    include_early_termination,
    early_termination_rate,
    early_termination_duration,
    
    # Provider settings
    disney_allocation_pct,
    costa_allocation_pct,
    num_cruises,
    
    # Disney cruise settings
    disney_first_cruise_salary,
    disney_second_cruise_salary,
    disney_third_cruise_salary,
    disney_cruise_duration,
    disney_cruise_dropout_rate,
    disney_cruise_salary_variation,
    disney_cruise_payment_fraction,
    
    # Costa cruise settings
    costa_first_cruise_salary,
    costa_second_cruise_salary,
    costa_third_cruise_salary,
    costa_cruise_duration,
    costa_cruise_dropout_rate,
    costa_cruise_salary_variation,
    costa_cruise_payment_fraction,
    
    # Break settings
    include_breaks,
    break_duration,
    break_dropout_rate,
    
    # Simulation settings
    num_students,
    num_sims
):
    # Convert percentage inputs to decimal values
    basic_training_dropout_rate = basic_training_dropout_rate / 100 if basic_training_dropout_rate is not None else 0.15
    no_offer_rate = no_offer_rate / 100 if no_offer_rate is not None else 0.30
    early_termination_rate = early_termination_rate / 100 if early_termination_rate is not None else 0.10
    advanced_training_dropout_rate = advanced_training_dropout_rate / 100 if advanced_training_dropout_rate is not None else 0.12
    
    disney_cruise_dropout_rate = disney_cruise_dropout_rate / 100 if disney_cruise_dropout_rate is not None else 0.03
    disney_cruise_payment_fraction = disney_cruise_payment_fraction / 100 if disney_cruise_payment_fraction is not None else 0.14
    
    costa_cruise_dropout_rate = costa_cruise_dropout_rate / 100 if costa_cruise_dropout_rate is not None else 0.03
    costa_cruise_payment_fraction = costa_cruise_payment_fraction / 100 if costa_cruise_payment_fraction is not None else 0.14
    
    break_dropout_rate = break_dropout_rate / 100 if break_dropout_rate is not None else 0.0
    
    # Create a config dict that can be serialized to JSON
    config = {
        # Training parameters
        'include_advanced_training': include_advanced_training,
        'basic_training_cost': basic_training_cost,
        'basic_training_dropout_rate': basic_training_dropout_rate,
        'basic_training_duration': basic_training_duration,
        
        # Offer stage parameters
        'include_offer_stage': include_offer_stage,
        'no_offer_rate': no_offer_rate,
        'offer_stage_duration': offer_stage_duration,
        
        # Transportation and placement parameters
        'advanced_training_cost': advanced_training_cost,
        'advanced_training_dropout_rate': advanced_training_dropout_rate,
        'advanced_training_duration': advanced_training_duration,
        
        # Early termination parameters
        'include_early_termination': include_early_termination,
        'early_termination_rate': early_termination_rate,
        'early_termination_duration': early_termination_duration,
        
        # Provider settings
        'disney_allocation_pct': disney_allocation_pct,
        'costa_allocation_pct': costa_allocation_pct,
        'num_cruises': num_cruises,
        
        # Disney cruise settings
        'disney_first_cruise_salary': disney_first_cruise_salary,
        'disney_second_cruise_salary': disney_second_cruise_salary,
        'disney_third_cruise_salary': disney_third_cruise_salary,
        'disney_cruise_duration': disney_cruise_duration,
        'disney_cruise_dropout_rate': disney_cruise_dropout_rate,
        'disney_cruise_salary_variation': disney_cruise_salary_variation,
        'disney_cruise_payment_fraction': disney_cruise_payment_fraction,
        
        # Costa cruise settings
        'costa_first_cruise_salary': costa_first_cruise_salary,
        'costa_second_cruise_salary': costa_second_cruise_salary,
        'costa_third_cruise_salary': costa_third_cruise_salary,
        'costa_cruise_duration': costa_cruise_duration,
        'costa_cruise_dropout_rate': costa_cruise_dropout_rate,
        'costa_cruise_salary_variation': costa_cruise_salary_variation,
        'costa_cruise_payment_fraction': costa_cruise_payment_fraction,
        
        # Break settings
        'include_breaks': include_breaks,
        'break_duration': break_duration,
        'break_dropout_rate': break_dropout_rate,
        
        # Simulation settings
        'num_students': num_students,
        'num_sims': num_sims,
    }
    
    return config

# *** START: Helper function for progression calculation ***
def calculate_progression_data(state_metrics, config):
    """
    Calculates the theoretical progression of students through states based on config dropout rates.

    Args:
        state_metrics (dict): The pre-processed state metrics dictionary.
        config (dict): The simulation configuration dictionary.

    Returns:
        list: A list of dictionaries, each containing progression data for a state.
              Keys: 'state', 'entered', 'completed', 'dropouts', 'dropout_rate',
                    'avg_salary', 'avg_payment', 'active_months', 'cash_flow_per_student'
    """
    progression_data = []
    total_entered_initial = config.get('num_students', 100)
    entered_count = total_entered_initial # Initialize for the first state

    # Sort state_metrics by state index to ensure correct order
    sorted_metrics = sorted(state_metrics.items(), key=lambda x: int(x[0]))

    for state_idx_str, metrics in sorted_metrics:
        state_idx = int(state_idx_str)
        if isinstance(metrics, dict):
            state_name = metrics.get('name', f"State {state_idx_str}")

            # Determine 'entered' count
            if state_idx == 0:
                entered = entered_count
            else:
                # Entered count is the 'completed' count from the previous state
                if progression_data: # Check if there's a previous state
                    entered = progression_data[-1].get('completed', 0)
                else:
                    entered = 0 # Should not happen if state_idx > 0

            # Calculate dropouts and completions based on configured dropout rates
            dropout_rate = 0.0 # Default as decimal
            if state_name == "Training":
                dropout_rate = config.get('basic_training_dropout_rate', 0)
            elif state_name == "Offer Stage":
                dropout_rate = config.get('no_offer_rate', 0)
            elif state_name == "Transportation and placement":
                dropout_rate = config.get('advanced_training_dropout_rate', 0)
            elif state_name == "Early Termination Stage":
                dropout_rate = config.get('early_termination_rate', 0)
            elif state_name == "First Cruise":
                dropout_rate = config.get('disney_cruise_dropout_rate', 0)
            elif "Break" in state_name:
                dropout_rate = config.get('break_dropout_rate', 0)
            elif "Cruise" in state_name:  # For subsequent cruises
                dropout_rate = config.get('costa_cruise_dropout_rate', 0)

            dropouts = round(entered * dropout_rate)
            completed = entered - dropouts

            # Get financial metrics for the state
            state_salary = metrics.get('avg_salary', 0)
            state_payment = metrics.get('avg_payment', 0)

            # Get state duration for converting to monthly values
            if state_name == "Training":
                state_duration = config.get('basic_training_duration', 0)
            elif state_name == "Offer Stage":
                state_duration = config.get('offer_stage_duration', 0)
            elif state_name == "Transportation and placement":
                state_duration = config.get('advanced_training_duration', 0)
            elif state_name == "Early Termination Stage":
                state_duration = config.get('early_termination_duration', 0)
            elif state_name == "First Cruise":
                state_duration = config.get('disney_cruise_duration', 0)
            elif "Break" in state_name:
                state_duration = config.get('break_duration', 0)
            else:  # Subsequent cruises
                state_duration = config.get('costa_cruise_duration', 0)

            # Convert state salary and payment to monthly values if duration > 0
            avg_monthly_salary = state_salary / state_duration if state_duration > 0 else 0
            avg_monthly_payment = state_payment / state_duration if state_duration > 0 else 0

            # Calculate cash flow per student for this state
            if state_name == "Training":
                cash_flow = -config.get('basic_training_cost', 0)
            elif state_name == "Offer Stage":
                cash_flow = 0  # No cost for offer stage
            elif state_name == "Transportation and placement":
                if config.get('include_advanced_training', False):
                    cash_flow = -config.get('advanced_training_cost', 0)
                else:
                    cash_flow = 0
            elif state_name == "Early Termination Stage":
                cash_flow = 0  # No cost for early termination stage
            else:  # Cruises and breaks
                cash_flow = state_payment  # Use total state payment

            progression_data.append({
                'state': state_name,
                'entered': entered,
                'completed': completed,
                'dropouts': dropouts,
                'dropout_rate': dropout_rate * 100,  # Convert to percentage for display
                'avg_salary': avg_monthly_salary,  # Now properly monthly
                'avg_payment': avg_monthly_payment,  # Now properly monthly
                'active_months': state_duration,
                'cash_flow_per_student': cash_flow
            })
        else:
             # Handle unexpected metric format
             progression_data.append({
                'state': f"State {state_idx_str} (error)",
                'entered': entered if state_idx > 0 else entered_count,
                'completed': 0,
                'dropouts': 0,
                'dropout_rate': 0.0,
                'avg_salary': 0,
                'avg_payment': 0,
                'active_months': 0,
                'cash_flow_per_student': 0
            })

    return progression_data
# *** END: Helper function for progression calculation ***

# Callback to run the simulation
@app.callback(
    [Output("loading-message", "children"),
     Output("simulation-results-store", "data")],
    [Input("run-simulation", "n_clicks")],
    [State("simulation-config-store", "data")]
)
def run_simulation_callback(n_clicks, config_data):
    if n_clicks is None or n_clicks == 0 or not config_data:
        return "", None
    
    # Create a SimulationConfig object from the stored config
    config = SimulationConfig(
        # General simulation parameters
        num_students=config_data.get('num_students', 100),
        
        # Training parameters
        include_advanced_training=config_data.get('include_advanced_training', True),
        basic_training_cost=config_data.get('basic_training_cost', 2000),
        basic_training_dropout_rate=config_data.get('basic_training_dropout_rate', 0.15),
        basic_training_duration=config_data.get('basic_training_duration', 6),
        
        # Offer stage parameters
        include_offer_stage=config_data.get('include_offer_stage', True),
        no_offer_rate=config_data.get('no_offer_rate', 0.30),
        offer_stage_duration=config_data.get('offer_stage_duration', 1),
        
        # Transportation and placement parameters
        advanced_training_cost=config_data.get('advanced_training_cost', 2000),
        advanced_training_dropout_rate=config_data.get('advanced_training_dropout_rate', 0.12),
        advanced_training_duration=config_data.get('advanced_training_duration', 5),
        
        # Early termination parameters
        include_early_termination=config_data.get('include_early_termination', True),
        early_termination_rate=config_data.get('early_termination_rate', 0.10),
        early_termination_duration=config_data.get('early_termination_duration', 1),
        
        # Provider allocation
        disney_allocation_pct=config_data.get('disney_allocation_pct', 30.0),
        costa_allocation_pct=config_data.get('costa_allocation_pct', 70.0),
        
        # Disney cruise parameters
        disney_first_cruise_salary=config_data.get('disney_first_cruise_salary', 5100),
        disney_second_cruise_salary=config_data.get('disney_second_cruise_salary', 5400),
        disney_third_cruise_salary=config_data.get('disney_third_cruise_salary', 18000),
        disney_cruise_duration=config_data.get('disney_cruise_duration', 6),
        disney_cruise_dropout_rate=config_data.get('disney_cruise_dropout_rate', 0.03),
        disney_cruise_salary_variation=config_data.get('disney_cruise_salary_variation', 5.0),
        disney_cruise_payment_fraction=config_data.get('disney_cruise_payment_fraction', 0.14),
        
        # Costa cruise parameters
        costa_first_cruise_salary=config_data.get('costa_first_cruise_salary', 5100),
        costa_second_cruise_salary=config_data.get('costa_second_cruise_salary', 5850),
        costa_third_cruise_salary=config_data.get('costa_third_cruise_salary', 9000),
        costa_cruise_duration=config_data.get('costa_cruise_duration', 7),
        costa_cruise_dropout_rate=config_data.get('costa_cruise_dropout_rate', 0.03),
        costa_cruise_salary_variation=config_data.get('costa_cruise_salary_variation', 5.0),
        costa_cruise_payment_fraction=config_data.get('costa_cruise_payment_fraction', 0.14),
        
        # Break parameters
        include_breaks=config_data.get('include_breaks', True),
        break_duration=config_data.get('break_duration', 2),
        break_dropout_rate=config_data.get('break_dropout_rate', 0.0),
        
        # Number of cruises
        num_cruises=config_data.get('num_cruises', 3)
    )
    
    random_seed = random.randint(1, 10000)
    config.random_seed = random_seed
    
    try:
        num_careers = config_data.get('num_students', 100)
        state_configs = config.create_state_configs()
        
        print(f"\nStarting simulation with {num_careers} careers...")
        
        # Run multiple simulations until we find one that completes all states
        # (or at least goes through more of them) for better cash flow visualization
        print("Running individual simulations to find a good cash flow example...")
        best_sim = None
        max_states_completed = -1
        
        # Try up to 10 times to find a simulation that completes all states
        for attempt in range(10):
            test_sim = run_simulation(state_configs=state_configs, random_seed=random_seed + attempt, simulation_config=config)
            states_completed = len(test_sim.get('completed_states', []))
            is_dropout = test_sim.get('dropout', True)
            
            # Keep track of the simulation that completes the most states
            if states_completed > max_states_completed:
                max_states_completed = states_completed
                best_sim = test_sim
                print(f"Found better simulation with {states_completed} completed states, dropout: {is_dropout}")
                
                # If we found a simulation that completed all states, we can stop
                if not is_dropout:
                    print("Found simulation that completes all states!")
                    break
        
        # Use our best simulation for state results
        single_sim = best_sim if best_sim else run_simulation(state_configs=state_configs, random_seed=random_seed, simulation_config=config)
        print(f"Using simulation with {len(single_sim.get('state_results', []))} states for cash flow")
        
        # Run the batch simulation for aggregate statistics
        batch_results = run_simulation_batch(config)
        
        # The rest of the function remains the same
        if not batch_results:
            print("Error: Failed to generate batch simulation results")
            return "Error: Failed to generate simulation results", None
        
        # Add the state results from the single simulation to the batch results
        batch_results['state_results'] = single_sim['state_results']
        
        # Verify we have state results
        print(f"State results count: {len(batch_results.get('state_results', []))}")
        
        # *** START: Pre-process state_metrics (Ensure this block is present and correct) ***
        if 'state_metrics' in batch_results and isinstance(batch_results['state_metrics'], dict):
            cleaned_metrics = {}
            for state_idx, metric_info in batch_results['state_metrics'].items():
                cleaned_state_data = {}
                actual_name = f"State {state_idx}" # Default
                
                if isinstance(metric_info, dict):
                    # Copy existing serializable data
                    for key, val in metric_info.items():
                         if isinstance(val, (int, float, str, bool, type(None))):
                             cleaned_state_data[key] = val
                         
                    # Attempt to extract the simple name
                    name_raw = metric_info.get('name')
                    if isinstance(name_raw, dict) and 'name' in name_raw:
                        actual_name = name_raw['name']
                    elif isinstance(name_raw, str):
                        actual_name = name_raw
                        
                    # Ensure the 'name' key holds the simple string name
                    cleaned_state_data['name'] = actual_name
                    
                    # Explicitly preserve avg_state_salary as avg_salary for the overview tab
                    if 'avg_state_salary' in metric_info:
                        cleaned_state_data['avg_salary'] = metric_info['avg_state_salary']
                    
                elif isinstance(metric_info, str): # Handle case where metric is just a name
                     cleaned_state_data['name'] = metric_info
                     cleaned_state_data['avg_salary'] = 0
                     cleaned_state_data['avg_payment'] = 0
                     cleaned_state_data['active_months'] = 0

                cleaned_metrics[str(state_idx)] = cleaned_state_data
                
            batch_results['state_metrics'] = cleaned_metrics # Replace original with cleaned version
        # *** END: Pre-process state_metrics ***

        # Add provider metrics to the cleaned output
        if 'provider_metrics' in batch_results and isinstance(batch_results['provider_metrics'], dict):
            provider_metrics = {}
            for provider, metrics in batch_results['provider_metrics'].items():
                if isinstance(metrics, dict):
                    provider_metrics[provider] = {
                        k: v for k, v in metrics.items() 
                        if isinstance(v, (int, float, str, bool, type(None)))
                    }
            batch_results['provider_metrics'] = provider_metrics

        # Add provider distribution to the cleaned output
        if 'provider_distribution' in batch_results and isinstance(batch_results['provider_distribution'], dict):
            provider_distribution = {
                k: v for k, v in batch_results['provider_distribution'].items()
                if isinstance(v, (int, float)) and k is not None
            }
            batch_results['provider_distribution'] = provider_distribution

        # Serialization logic (more explicit for state_metrics)
        serializable_results = {}
        for key, value in batch_results.items():
            if key == 'state_metrics':
                # Explicitly build the serializable dict for state_metrics
                temp_metrics_dict = {}
                if isinstance(value, dict): # Value should be the cleaned_metrics dict
                    for state_idx_str, state_data_dict in value.items():
                        # state_data_dict should contain simple types after pre-processing
                        temp_metrics_dict[state_idx_str] = state_data_dict
                serializable_results[key] = temp_metrics_dict # Assign the explicitly built dict
            elif key == 'state_results':
                # Preserve the state results array as-is
                serializable_results[key] = value
            # *** START Handle New Aggregated State Data ***
            elif key in ['state_total_costs', 'state_total_payments', 'state_entry_counts']:
                 # These should already be dicts with string keys (state index as string) and numeric values
                 serializable_results[key] = {str(k): v for k, v in value.items()} if isinstance(value, dict) else {}
            # *** END Handle New Aggregated State Data ***
            elif key in ['provider_metrics', 'provider_distribution']:
                # These are already cleaned above
                serializable_results[key] = value
            elif isinstance(value, (int, float, str, bool, type(None))):
                serializable_results[key] = value
            elif isinstance(value, dict):
                # General dictionary handling (excluding state_metrics)
                serializable_dict = {}
                for k, v in value.items():
                    if isinstance(v, (int, float, str, bool, type(None))):
                        serializable_dict[str(k)] = v
                    elif hasattr(v, 'to_dict'):
                        serializable_dict[str(k)] = v.to_dict()
                    else:
                        pass 
                serializable_results[key] = serializable_dict
            elif hasattr(value, 'to_dict'): # Pandas objects
                serializable_results[key] = value.to_dict()
            else:
                 pass # Skip other non-serializable types
        
        serializable_results['config'] = config_data
        
        print(f"Simulation complete with {len(serializable_results.get('state_results', []))} states in results")
        return f"Completed {num_careers} career simulations!", serializable_results
    
    except Exception as e:
        import traceback
        print(f"Error during simulation or serialization: {e}")
        traceback.print_exc()
        return f"Error in simulation: {str(e)}", None

# Callback to update summary stats
@app.callback(
    Output("summary-stats", "children"),
    [Input("simulation-results-store", "data")]
)
def update_summary_stats(results):
    if not results:
        return "Run a simulation to see results"
    
    # Extract key metrics
    completion_rate = results.get('completion_rate', 0)
    dropout_rate = results.get('dropout_rate', 0)
    avg_training_cost = results.get('avg_training_cost', 0)
    avg_total_payments = results.get('avg_total_payments', 0)
    avg_net_cash_flow = results.get('avg_net_cash_flow', 0)
    
    # Create a summary statistics card
    return html.Div([
        html.H4("Summary Statistics", style={'textAlign': 'center', 'marginBottom': '20px'}),
        html.Div([
            html.Div([
                html.H5("Career Outcomes", style={'textAlign': 'center', 'marginBottom': '10px'}),
                html.Div([
                    html.Div([
                        html.P("Completion Rate:", style={'fontWeight': 'bold'}),
                        html.P(f"{completion_rate:.1f}%")
                    ], style={'marginBottom': '10px'}),
                    html.Div([
                        html.P("Dropout Rate:", style={'fontWeight': 'bold'}),
                        html.P(f"{dropout_rate:.1f}%")
                    ])
                ])
            ], style={'width': '50%', 'display': 'inline-block', 'verticalAlign': 'top'}),
            
            html.Div([
                html.H5("Financial Outcomes", style={'textAlign': 'center', 'marginBottom': '10px'}),
                html.Div([
                    html.Div([
                        html.P("Avg Training Cost:", style={'fontWeight': 'bold'}),
                        html.P(f"${avg_training_cost:.2f}")
                    ], style={'marginBottom': '10px'}),
                    html.Div([
                        html.P("Avg Total Payments:", style={'fontWeight': 'bold'}),
                        html.P(f"${avg_total_payments:.2f}")
                    ], style={'marginBottom': '10px'}),
                    html.Div([
                        html.P("Avg Net Cash Flow:", style={'fontWeight': 'bold'}),
                        html.P(f"${avg_net_cash_flow:.2f}")
                    ])
                ])
            ], style={'width': '50%', 'display': 'inline-block', 'verticalAlign': 'top'})
        ], style={'backgroundColor': '#f8f9fa', 'padding': '15px', 'borderRadius': '10px'})
    ])

# Callback to update overview content
@app.callback(
    Output("overview-content", "children"),
    [Input("simulation-results-store", "data")]
)
def update_overview_content(results):
    if not results:
        return "Run a simulation to see results"

    # Extract data
    state_metrics = results.get('state_metrics', {})
    config = results.get('config', {})
    state_total_costs = results.get('state_total_costs', {})
    state_total_payments = results.get('state_total_payments', {})
    state_entry_counts = results.get('state_entry_counts', {})
    provider_metrics = results.get('provider_metrics', {})
    provider_distribution = results.get('provider_distribution', {})
    num_simulations = config.get('num_students', 0)

    # Create provider distribution boxes
    provider_boxes = None
    if provider_distribution:
        total_students = sum(provider_distribution.values())
        provider_boxes = html.Div([
            html.H5("Provider Distribution", style={'textAlign': 'center', 'marginBottom': '15px'}),
            html.Div([
                html.Div([
                    html.H5("Disney", style={'textAlign': 'center', 'marginBottom': '10px', 'color': '#2196F3'}),
                    html.Div(f"{provider_distribution.get('Disney', 0)} students", style={'textAlign': 'center'}),
                    html.Div(f"({provider_distribution.get('Disney', 0)/total_students*100:.1f}%)", 
                             style={'textAlign': 'center', 'fontSize': '18px'})
                ], style={'width': '45%', 'display': 'inline-block', 'backgroundColor': '#e3f2fd', 'padding': '15px', 'borderRadius': '5px', 'marginRight': '10%'}),
                
                html.Div([
                    html.H5("Costa", style={'textAlign': 'center', 'marginBottom': '10px', 'color': '#4CAF50'}),
                    html.Div(f"{provider_distribution.get('Costa', 0)} students", style={'textAlign': 'center'}),
                    html.Div(f"({provider_distribution.get('Costa', 0)/total_students*100:.1f}%)", 
                             style={'textAlign': 'center', 'fontSize': '18px'})
                ], style={'width': '45%', 'display': 'inline-block', 'backgroundColor': '#e8f5e9', 'padding': '15px', 'borderRadius': '5px'})
            ], style={'marginBottom': '20px'})
        ], style={'marginBottom': '30px'})

    # Process state data for visualizations and metrics
    state_data = []
    for state_idx_str, metrics in sorted(state_metrics.items(), key=lambda x: int(x[0])):
        if isinstance(metrics, dict):
            state_name = metrics.get('name', f"State {state_idx_str}")
            provider = metrics.get('provider', "")
            
            total_cost = state_total_costs.get(state_idx_str, 0.0)
            total_payment = state_total_payments.get(state_idx_str, 0.0)
            entry_count = state_entry_counts.get(state_idx_str, 0)

            if entry_count > 0:
                net_cash_flow = total_payment - total_cost
                state_data.append({
                    "State": state_name,
                    "Provider": provider,
                    "Total Costs": total_cost,
                    "Total Payments": total_payment,
                    "Net Cash Flow": net_cash_flow,
                    "Simulations Entered": entry_count,
                    "Entry Rate (%)": (entry_count / num_simulations * 100) if num_simulations > 0 else 0,
                    "Avg Payment Per Student": total_payment / entry_count if entry_count > 0 else 0,
                    "Payment Fraction (%)": (
                        config.get('disney_cruise_payment_fraction', 0.14) * 100 if "Disney" in provider and "Cruise" in state_name
                        else config.get('costa_cruise_payment_fraction', 0.14) * 100 if "Costa" in provider and "Cruise" in state_name
                        else 0
                    ),
                    "Implied Avg Salary": (
                        (total_payment / entry_count) / (
                            config.get('disney_cruise_payment_fraction', 0.14) if "Disney" in provider and "Cruise" in state_name
                            else config.get('costa_cruise_payment_fraction', 0.14) if "Costa" in provider and "Cruise" in state_name
                            else 1
                        ) if entry_count > 0 and (
                            config.get('disney_cruise_payment_fraction', 0.14) if "Disney" in provider and "Cruise" in state_name
                            else config.get('costa_cruise_payment_fraction', 0.14) if "Costa" in provider and "Cruise" in state_name
                            else 1
                        ) > 0 else 0
                    )
                })

    # Calculate overall financials
    overall_total_costs = sum(item["Total Costs"] for item in state_data)
    overall_total_payments = sum(item["Total Payments"] for item in state_data)
    overall_net_cash_flow = overall_total_payments - overall_total_costs

    # Create financial summary boxes
    financial_summary = html.Div([
        html.H5("Overall Financial Summary", style={'textAlign': 'center', 'marginBottom': '15px'}),
        html.Div([
            html.Div([
                html.P("Total Training Costs:", style={'fontWeight': 'bold'}),
                html.P(f"${overall_total_costs:,.2f}", style={'color': 'red'})
            ], style={'display': 'inline-block', 'width': '33%', 'textAlign': 'center'}),

            html.Div([
                html.P("Total Payments Received:", style={'fontWeight': 'bold'}),
                html.P(f"${overall_total_payments:,.2f}", style={'color': 'green'})
            ], style={'display': 'inline-block', 'width': '33%', 'textAlign': 'center'}),

            html.Div([
                html.P("Net Cash Flow:", style={'fontWeight': 'bold'}),
                html.P(f"${overall_net_cash_flow:,.2f}",
                      style={'color': 'green' if overall_net_cash_flow >= 0 else 'red'})
            ], style={'display': 'inline-block', 'width': '33%', 'textAlign': 'center'})
        ], style={'border': '1px solid #ddd', 'padding': '15px', 'borderRadius': '5px', 'backgroundColor': '#f9f9f9'})
    ])

    # Create provider-specific metrics if available
    provider_metrics_boxes = None
    if provider_metrics:
        provider_metrics_boxes = html.Div([
            html.H5("Provider-Specific Results", style={'textAlign': 'center', 'marginBottom': '15px'}),
            html.Div([
                # Disney metrics
                html.Div([
                    html.H5("Disney", style={'textAlign': 'center', 'marginBottom': '10px', 'color': '#2196F3'}),
                    html.Div([
                        html.P("Avg Training Cost:", style={'fontWeight': 'bold'}),
                        html.P(f"${provider_metrics.get('Disney', {}).get('avg_training_cost', 0):,.2f}")
                    ], style={'marginBottom': '5px'}),
                    html.Div([
                        html.P("Avg Total Payments:", style={'fontWeight': 'bold'}),
                        html.P(f"${provider_metrics.get('Disney', {}).get('avg_total_payments', 0):,.2f}")
                    ], style={'marginBottom': '5px'}),
                    html.Div([
                        html.P("Avg Net Cash Flow:", style={'fontWeight': 'bold'}),
                        html.P(f"${provider_metrics.get('Disney', {}).get('avg_net_cash_flow', 0):,.2f}")
                    ], style={'marginBottom': '5px'}),
                    html.Div([
                        html.P("ROI:", style={'fontWeight': 'bold'}),
                        html.P(f"{provider_metrics.get('Disney', {}).get('avg_roi', 0):.1f}%")
                    ])
                ], style={'width': '45%', 'display': 'inline-block', 'backgroundColor': '#e3f2fd', 'padding': '15px', 'borderRadius': '5px', 'marginRight': '10%'}),
                
                # Costa metrics
                html.Div([
                    html.H5("Costa", style={'textAlign': 'center', 'marginBottom': '10px', 'color': '#4CAF50'}),
                    html.Div([
                        html.P("Avg Training Cost:", style={'fontWeight': 'bold'}),
                        html.P(f"${provider_metrics.get('Costa', {}).get('avg_training_cost', 0):,.2f}")
                    ], style={'marginBottom': '5px'}),
                    html.Div([
                        html.P("Avg Total Payments:", style={'fontWeight': 'bold'}),
                        html.P(f"${provider_metrics.get('Costa', {}).get('avg_total_payments', 0):,.2f}")
                    ], style={'marginBottom': '5px'}),
                    html.Div([
                        html.P("Avg Net Cash Flow:", style={'fontWeight': 'bold'}),
                        html.P(f"${provider_metrics.get('Costa', {}).get('avg_net_cash_flow', 0):,.2f}")
                    ], style={'marginBottom': '5px'}),
                    html.Div([
                        html.P("ROI:", style={'fontWeight': 'bold'}),
                        html.P(f"{provider_metrics.get('Costa', {}).get('avg_roi', 0):.1f}%")
                    ])
                ], style={'width': '45%', 'display': 'inline-block', 'backgroundColor': '#e8f5e9', 'padding': '15px', 'borderRadius': '5px'})
            ], style={'marginBottom': '20px'})
        ], style={'marginBottom': '30px'})

    # Create detailed state metrics table
    state_metrics_table = dash_table.DataTable(
        id='state-metrics-table',
        data=state_data,
        columns=[
            {"name": "State", "id": "State"},
            {"name": "Entries", "id": "Simulations Entered", 'type': 'numeric'},
            {"name": "Entry %", "id": "Entry Rate (%)", 'type': 'numeric', 'format': {'specifier': '.1f'}},
            {"name": "Total Costs", "id": "Total Costs", 'type': 'numeric', 'format': {'specifier': '$,.2f'}},
            {"name": "Total Payments", "id": "Total Payments", 'type': 'numeric', 'format': {'specifier': '$,.2f'}},
            {"name": "Net Cash Flow", "id": "Net Cash Flow", 'type': 'numeric', 'format': {'specifier': '$,.2f'}},
            {"name": "Avg Payment/Student", "id": "Avg Payment Per Student", 'type': 'numeric', 'format': {'specifier': '$,.2f'}},
            {"name": "Payment %", "id": "Payment Fraction (%)", 'type': 'numeric', 'format': {'specifier': '.1f'}},
            {"name": "Implied Salary", "id": "Implied Avg Salary", 'type': 'numeric', 'format': {'specifier': '$,.2f'}}
        ],
        style_table={
            'overflowX': 'auto',
            'minWidth': '100%'
        },
        style_cell={
            'textAlign': 'center',
            'padding': '5px',
            'font-family': 'Arial, sans-serif',
            'font-size': '12px',
            'whiteSpace': 'normal',
            'height': 'auto',
            'minWidth': '60px',
            'maxWidth': '180px'
        },
        style_cell_conditional=[
            {'if': {'column_id': 'State'}, 'textAlign': 'left', 'minWidth': '80px', 'maxWidth': '120px'},
            {'if': {'column_id': 'Simulations Entered'}, 'minWidth': '60px', 'maxWidth': '80px'},
            {'if': {'column_id': 'Entry Rate (%)'}, 'minWidth': '60px', 'maxWidth': '80px'},
            {'if': {'column_id': 'Total Costs'}, 'minWidth': '90px', 'maxWidth': '120px'},
            {'if': {'column_id': 'Total Payments'}, 'minWidth': '90px', 'maxWidth': '120px'},
            {'if': {'column_id': 'Net Cash Flow'}, 'minWidth': '90px', 'maxWidth': '120px'},
            {'if': {'column_id': 'Avg Payment Per Student'}, 'minWidth': '90px', 'maxWidth': '120px'},
            {'if': {'column_id': 'Payment Fraction (%)'}, 'minWidth': '60px', 'maxWidth': '80px'},
            {'if': {'column_id': 'Implied Avg Salary'}, 'minWidth': '90px', 'maxWidth': '120px'}
        ],
        style_header={
            'backgroundColor': 'rgb(230, 230, 230)',
            'fontWeight': 'bold',
            'textAlign': 'center',
            'padding': '5px',
            'whiteSpace': 'normal',
            'height': 'auto'
        },
        style_data={
            'whiteSpace': 'normal',
            'height': 'auto',
            'lineHeight': '15px'
        },
        style_data_conditional=[
            {
                'if': {'row_index': 'odd'},
                'backgroundColor': 'rgb(248, 248, 248)'
            },
            {
                'if': {'filter_query': '{Net Cash Flow} < 0'},
                'color': 'red'
            },
            {
                'if': {'filter_query': '{Net Cash Flow} > 0'},
                'color': 'green'
            }
        ],
        tooltip_data=[
            {
                'Implied Avg Salary': {'value': 'Calculated as: Avg Payment Per Student / Payment Fraction'},
                'Avg Payment Per Student': {'value': 'Total Payments / Simulations Entered'},
                'Payment Fraction (%)': {'value': 'Percentage of salary paid as payment'}
            } for row in state_data
        ],
        tooltip_duration=None,
        page_size=20,
        style_as_list_view=True
    )

    # Combine all elements
    return html.Div([
        html.H4("Simulation Overview", style={'textAlign': 'center', 'marginBottom': '20px'}),
        
        # Provider distribution section (if available)
        provider_boxes if provider_boxes else None,
        
        # Provider-specific metrics (if available)
        provider_metrics_boxes if provider_metrics_boxes else None,
        
        # Financial section
        html.Div([
            financial_summary,
            html.Div([
                html.H5("Detailed State Metrics", style={'textAlign': 'center', 'marginBottom': '15px'}),
                html.P("This table shows detailed metrics for each state in the simulation:",
                      style={'marginBottom': '15px'}),
                html.Ul([
                    html.Li("Simulations Entered: Number of simulations that reached this state"),
                    html.Li("Entry Rate: Percentage of total simulations that entered the state"),
                    html.Li("Total Costs: Sum of all costs incurred in this state"),
                    html.Li("Total Payments: Sum of all payments received in this state"),
                    html.Li("Net Cash Flow: Difference between payments and costs"),
                    html.Li("Avg Payment Per Student: Total payments divided by number of students who entered the state"),
                    html.Li("Payment Fraction: Percentage of salary that is paid as payment"),
                    html.Li("Implied Salary: Calculated as Avg Payment Per Student / Payment Fraction")
                ], style={'marginBottom': '15px'}),
                state_metrics_table
            ])
        ])
    ])

# Run the app
if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=10000) 
