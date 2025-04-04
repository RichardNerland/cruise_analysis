import dash
from dash import dcc, html, Input, Output, State, dash_table
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import time
import random
import argparse

# Import the cruise model
from simple_cruise_model import (
    run_simulation, 
    run_simulation_batch, 
    create_default_state_configs,
    calculate_summary_metrics, 
    print_simulation_summary
)
from simulation_config import (
    StateConfig, 
    SimulationConfig, 
    BASELINE_CONFIG, 
    OPTIMISTIC_CONFIG, 
    PESSIMISTIC_CONFIG
)

# Initialize the Dash app
app = dash.Dash(__name__, suppress_callback_exceptions=True)
server = app.server  # Expose the server variable for production

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
                            ], style={'marginBottom': '15px'})
                            
                        ], style={'marginBottom': '20px', 'backgroundColor': '#f1f1f1', 'padding': '15px', 'borderRadius': '5px'}),
                        
                        # Training parameters
                        html.Div([
                            html.H4("Training Parameters", style={'marginBottom': '15px'}),
                            
                            html.Div([
                                html.Label("Include Advanced Training:"),
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
                                html.Label("Basic Training Cost ($):"),
                                dcc.Input(
                                    id="basic-training-cost",
                                    type="number",
                                    value=2000,
                                    min=0,
                                    step=100,
                                    style={"width": "100%"}
                                )
                            ], style={'marginBottom': '15px'}),
                            
                            html.Div([
                                html.Label("Basic Training Dropout Rate (%):"),
                                dcc.Slider(
                                    id="basic-training-dropout-rate",
                                    min=0,
                                    max=50,
                                    step=1,
                                    value=15,
                                    marks={i: f'{i}%' for i in range(0, 51, 10)},
                                )
                            ], style={'marginBottom': '15px'}),
                            
                            html.Div([
                                html.Label("Basic Training Duration (months):"),
                                dcc.Input(
                                    id="basic-training-duration",
                                    type="number",
                                    value=3,
                                    min=1,
                                    max=24,
                                    step=1,
                                    style={"width": "100%"}
                                )
                            ], style={'marginBottom': '15px'}),
                            
                            # Add the advanced training parameters directly in the layout instead of using conditional rendering
                            html.Div([
                                html.Div([
                                    html.Label("Advanced Training Cost ($):"),
                                    dcc.Input(
                                        id="advanced-training-cost",
                                        type="number",
                                        value=2000,
                                        min=0,
                                        step=100,
                                        style={"width": "100%"}
                                    )
                                ], style={'marginBottom': '15px'}),
                                
                                html.Div([
                                    html.Label("Advanced Training Dropout Rate (%):"),
                                    dcc.Slider(
                                        id="advanced-training-dropout-rate",
                                        min=0,
                                        max=50,
                                        step=1,
                                        value=12,
                                        marks={i: f'{i}%' for i in range(0, 51, 10)},
                                    )
                                ], style={'marginBottom': '15px'}),
                                
                                html.Div([
                                    html.Label("Advanced Training Duration (months):"),
                                    dcc.Input(
                                        id="advanced-training-duration",
                                        type="number",
                                        value=3,
                                        min=1,
                                        max=24,
                                        step=1,
                                        style={"width": "100%"}
                                    )
                                ], style={'marginBottom': '15px'})
                            ], id="advanced-training-container", style={'marginBottom': '15px', 'backgroundColor': '#e6f7ff', 'padding': '15px', 'borderRadius': '5px'})
                        ], style={'marginBottom': '20px', 'backgroundColor': '#f1f1f1', 'padding': '15px', 'borderRadius': '5px'}),
                        
                        # Cruise parameters
                        html.Div([
                            html.H4("Cruise Parameters", style={'marginBottom': '15px'}),
                            
                            html.Div([
                                html.Label("Number of Additional Cruises:"),
                                dcc.Input(
                                    id="num-additional-cruises",
                                    type="number",
                                    value=3,
                                    min=0,
                                    max=10,
                                    step=1,
                                    style={"width": "100%"}
                                )
                            ], style={'marginBottom': '15px'}),
                            
                            html.Div([
                                html.Label("First Cruise Base Salary ($):"),
                                dcc.Input(
                                    id="first-cruise-base-salary",
                                    type="number",
                                    value=5000,
                                    min=0,
                                    step=100,
                                    style={"width": "100%"}
                                )
                            ], style={'marginBottom': '15px'}),
                            
                            html.Div([
                                html.Label("First Cruise Dropout Rate (%):"),
                                dcc.Slider(
                                    id="first-cruise-dropout-rate",
                                    min=0,
                                    max=50,
                                    step=1,
                                    value=15,
                                    marks={i: f'{i}%' for i in range(0, 51, 10)},
                                )
                            ], style={'marginBottom': '15px'}),
                            
                            html.Div([
                                html.Label("First Cruise Salary Variation (%):"),
                                dcc.Slider(
                                    id="first-cruise-salary-variation",
                                    min=0,
                                    max=20,
                                    step=0.5,
                                    value=6,
                                    marks={i: f'{i}%' for i in range(0, 21, 5)},
                                )
                            ], style={'marginBottom': '15px'}),
                            
                            html.Div([
                                html.Label("First Cruise Duration (months):"),
                                dcc.Input(
                                    id="first-cruise-duration",
                                    type="number",
                                    value=8,
                                    min=1,
                                    max=24,
                                    step=1,
                                    style={"width": "100%"}
                                )
                            ], style={'marginBottom': '15px'}),
                            
                            html.Div([
                                html.Label("First Cruise Payment Fraction (%):"),
                                dcc.Slider(
                                    id="first-cruise-payment-fraction",
                                    min=0,
                                    max=30,
                                    step=0.5,
                                    value=14,
                                    marks={i: f'{i}%' for i in range(0, 31, 5)},
                                )
                            ], style={'marginBottom': '15px'}),
                            
                            html.Hr(),
                            
                            html.Div([
                                html.Label("Subsequent Cruise Dropout Rate (%):"),
                                dcc.Slider(
                                    id="subsequent-cruise-dropout-rate",
                                    min=0,
                                    max=20,
                                    step=0.5,
                                    value=2,
                                    marks={i: f'{i}%' for i in range(0, 21, 5)},
                                )
                            ], style={'marginBottom': '15px'}),
                            
                            html.Div([
                                html.Label("Subsequent Cruise Salary Increase (%):"),
                                dcc.Slider(
                                    id="subsequent-cruise-salary-increase",
                                    min=0,
                                    max=25,
                                    step=0.5,
                                    value=10,
                                    marks={i: f'{i}%' for i in range(0, 26, 5)},
                                )
                            ], style={'marginBottom': '15px'}),
                            
                            html.Div([
                                html.Label("Subsequent Cruise Salary Variation (%):"),
                                dcc.Slider(
                                    id="subsequent-cruise-salary-variation",
                                    min=0,
                                    max=20,
                                    step=0.5,
                                    value=5,
                                    marks={i: f'{i}%' for i in range(0, 21, 5)},
                                )
                            ], style={'marginBottom': '15px'}),
                            
                            html.Div([
                                html.Label("Subsequent Cruise Duration (months):"),
                                dcc.Input(
                                    id="subsequent-cruise-duration",
                                    type="number",
                                    value=8,
                                    min=1,
                                    max=24,
                                    step=1,
                                    style={"width": "100%"}
                                )
                            ], style={'marginBottom': '15px'}),
                            
                            html.Div([
                                html.Label("Subsequent Cruise Payment Fraction (%):"),
                                dcc.Slider(
                                    id="subsequent-cruise-payment-fraction",
                                    min=0,
                                    max=30,
                                    step=0.5,
                                    value=14,
                                    marks={i: f'{i}%' for i in range(0, 31, 5)},
                                )
                            ], style={'marginBottom': '15px'}),
                        ], style={'marginBottom': '20px', 'backgroundColor': '#f1f1f1', 'padding': '15px', 'borderRadius': '5px'}),
                        
                        # Run simulation button
                        html.Div([
                            html.Button(
                                "Run Career Simulations", 
                                id="run-simulation", 
                                n_clicks=0,
                                style={
                                    'backgroundColor': '#4CAF50',
                                    'color': 'white',
                                    'padding': '12px 20px',
                                    'borderRadius': '5px',
                                    'border': 'none',
                                    'fontSize': '16px',
                                    'cursor': 'pointer',
                                    'width': '100%',
                                    'fontWeight': 'bold'
                                }
                            ),
                            html.Div(id="loading-message", style={'marginTop': '10px', 'color': '#888'})
                        ])
                    ], style={'width': '30%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '20px', 'boxShadow': '0 4px 8px 0 rgba(0,0,0,0.2)', 'backgroundColor': '#f9f9f9', 'borderRadius': '8px'}),
                    
                    # Right panel for results
                    html.Div([
                        html.H3("Results"),
                        
                        html.Div([
                            html.Div(id="summary-stats", style={'marginBottom': '20px'}),
                            
                            dcc.Tabs([
                                dcc.Tab(label='Overview', children=[
                                    html.Div(id="overview-content")
                                ]),
                                dcc.Tab(label='State Progression', children=[
                                    html.Div(id="state-progression-content")
                                ]),
                                dcc.Tab(label='Financial Analysis', children=[
                                    html.Div(id="financial-analysis-content")
                                ]),
                                dcc.Tab(label='Monthly Metrics', children=[
                                    html.Div(id="monthly-metrics-content")
                                ]),
                                dcc.Tab(label='Detailed Data', children=[
                                    html.Div(id="detailed-data")
                                ]),
                                dcc.Tab(label='State Tracking', children=[
                                    html.Div(id="state-tracking-content")
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
                                dcc.Tab(label='Active Students', children=[
                                    html.Div(id="active-students-content")
                                ])
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

# Callback to update all parameters when a preset is selected
@app.callback(
    [
        # Training parameters
        Output("include-advanced-training", "value"),
        Output("basic-training-cost", "value"),
        Output("basic-training-dropout-rate", "value"),
        Output("basic-training-duration", "value"),
        Output("advanced-training-cost", "value"),
        Output("advanced-training-dropout-rate", "value"),
        Output("advanced-training-duration", "value"),
        
        # Cruise parameters
        Output("num-additional-cruises", "value"),
        Output("first-cruise-base-salary", "value"),
        Output("first-cruise-dropout-rate", "value"),
        Output("first-cruise-salary-variation", "value"),
        Output("first-cruise-duration", "value"),
        Output("first-cruise-payment-fraction", "value"),
        Output("subsequent-cruise-dropout-rate", "value"),
        Output("subsequent-cruise-salary-increase", "value"),
        Output("subsequent-cruise-salary-variation", "value"),
        Output("subsequent-cruise-duration", "value"),
        Output("subsequent-cruise-payment-fraction", "value"),
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
        config.basic_training_cost,
        config.basic_training_dropout_rate * 100,
        config.basic_training_duration,
        config.advanced_training_cost,
        config.advanced_training_dropout_rate * 100,
        config.advanced_training_duration,
        config.num_additional_cruises,
        config.first_cruise_base_salary,
        config.first_cruise_dropout_rate * 100,
        config.first_cruise_salary_variation,
        config.first_cruise_duration,
        config.first_cruise_payment_fraction * 100,
        config.subsequent_cruise_dropout_rate * 100,
        config.subsequent_cruise_salary_increase,
        config.subsequent_cruise_salary_variation,
        config.subsequent_cruise_duration,
        config.subsequent_cruise_payment_fraction * 100
    )

# Callback to create a simulation configuration and store it
@app.callback(
    Output("simulation-config-store", "data"),
    [
        Input("include-advanced-training", "value"),
        Input("basic-training-cost", "value"),
        Input("basic-training-dropout-rate", "value"),
        Input("basic-training-duration", "value"),
        Input("advanced-training-cost", "value"),
        Input("advanced-training-dropout-rate", "value"),
        Input("advanced-training-duration", "value"),
        Input("num-additional-cruises", "value"),
        Input("first-cruise-base-salary", "value"),
        Input("first-cruise-dropout-rate", "value"),
        Input("first-cruise-salary-variation", "value"),
        Input("first-cruise-duration", "value"),
        Input("first-cruise-payment-fraction", "value"),
        Input("subsequent-cruise-dropout-rate", "value"),
        Input("subsequent-cruise-salary-increase", "value"),
        Input("subsequent-cruise-salary-variation", "value"),
        Input("subsequent-cruise-duration", "value"),
        Input("subsequent-cruise-payment-fraction", "value"),
        Input("num-students", "value"),
        Input("num-sims", "value")
    ]
)
def update_simulation_config(
    include_advanced_training, 
    basic_training_cost, 
    basic_training_dropout_rate,
    basic_training_duration,
    advanced_training_cost,
    advanced_training_dropout_rate,
    advanced_training_duration,
    num_additional_cruises,
    first_cruise_base_salary,
    first_cruise_dropout_rate,
    first_cruise_salary_variation,
    first_cruise_duration,
    first_cruise_payment_fraction,
    subsequent_cruise_dropout_rate,
    subsequent_cruise_salary_increase,
    subsequent_cruise_salary_variation,
    subsequent_cruise_duration,
    subsequent_cruise_payment_fraction,
    num_students,
    num_sims
):
    # Convert percentage inputs to decimal values
    basic_training_dropout_rate = basic_training_dropout_rate / 100 if basic_training_dropout_rate else 0.15
    advanced_training_dropout_rate = advanced_training_dropout_rate / 100 if advanced_training_dropout_rate else 0.12
    first_cruise_dropout_rate = first_cruise_dropout_rate / 100 if first_cruise_dropout_rate else 0.15
    first_cruise_payment_fraction = first_cruise_payment_fraction / 100 if first_cruise_payment_fraction else 0.14
    subsequent_cruise_dropout_rate = subsequent_cruise_dropout_rate / 100 if subsequent_cruise_dropout_rate else 0.02
    subsequent_cruise_payment_fraction = subsequent_cruise_payment_fraction / 100 if subsequent_cruise_payment_fraction else 0.14
    
    # Create a config dict that can be serialized to JSON
    config = {
        'include_advanced_training': include_advanced_training,
        'basic_training_cost': basic_training_cost,
        'basic_training_dropout_rate': basic_training_dropout_rate,
        'basic_training_duration': basic_training_duration,
        'advanced_training_cost': advanced_training_cost,
        'advanced_training_dropout_rate': advanced_training_dropout_rate,
        'advanced_training_duration': advanced_training_duration,
        'num_additional_cruises': num_additional_cruises,
        'first_cruise_base_salary': first_cruise_base_salary,
        'first_cruise_dropout_rate': first_cruise_dropout_rate,
        'first_cruise_salary_variation': first_cruise_salary_variation,
        'first_cruise_duration': first_cruise_duration,
        'first_cruise_payment_fraction': first_cruise_payment_fraction,
        'subsequent_cruise_dropout_rate': subsequent_cruise_dropout_rate,
        'subsequent_cruise_salary_increase': subsequent_cruise_salary_increase,
        'subsequent_cruise_salary_variation': subsequent_cruise_salary_variation,
        'subsequent_cruise_duration': subsequent_cruise_duration,
        'subsequent_cruise_payment_fraction': subsequent_cruise_payment_fraction,
        'num_students': num_students,
        'num_sims': num_sims,
    }
    
    return config

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
        num_students=config_data.get('num_students', 100),
        include_advanced_training=config_data.get('include_advanced_training', True),
        basic_training_cost=config_data.get('basic_training_cost', 2000),
        basic_training_dropout_rate=config_data.get('basic_training_dropout_rate', 0.15),
        basic_training_duration=config_data.get('basic_training_duration', 3),
        advanced_training_cost=config_data.get('advanced_training_cost', 2000),
        advanced_training_dropout_rate=config_data.get('advanced_training_dropout_rate', 0.12),
        advanced_training_duration=config_data.get('advanced_training_duration', 3),
        num_additional_cruises=config_data.get('num_additional_cruises', 3),
        first_cruise_base_salary=config_data.get('first_cruise_base_salary', 5000),
        first_cruise_dropout_rate=config_data.get('first_cruise_dropout_rate', 0.15),
        first_cruise_salary_variation=config_data.get('first_cruise_salary_variation', 6.0),
        first_cruise_duration=config_data.get('first_cruise_duration', 8),
        first_cruise_payment_fraction=config_data.get('first_cruise_payment_fraction', 0.14),
        subsequent_cruise_dropout_rate=config_data.get('subsequent_cruise_dropout_rate', 0.02),
        subsequent_cruise_salary_increase=config_data.get('subsequent_cruise_salary_increase', 10.0),
        subsequent_cruise_salary_variation=config_data.get('subsequent_cruise_salary_variation', 5.0),
        subsequent_cruise_duration=config_data.get('subsequent_cruise_duration', 8),
        subsequent_cruise_payment_fraction=config_data.get('subsequent_cruise_payment_fraction', 0.14)
    )
    
    random_seed = random.randint(1, 10000)
    config.random_seed = random_seed
    
    try:
        num_careers = config_data.get('num_students', 100)
        results = run_simulation_batch(config)
        
        # *** START: Pre-process state_metrics (Ensure this block is present and correct) ***
        if 'state_metrics' in results and isinstance(results['state_metrics'], dict):
            cleaned_metrics = {}
            for state_idx, metric_info in results['state_metrics'].items():
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
                    
                elif isinstance(metric_info, str): # Handle case where metric is just a name
                     cleaned_state_data['name'] = metric_info
                     cleaned_state_data['avg_salary'] = 0
                     cleaned_state_data['avg_payment'] = 0
                     cleaned_state_data['active_months'] = 0

                cleaned_metrics[str(state_idx)] = cleaned_state_data
                
            results['state_metrics'] = cleaned_metrics # Replace original with cleaned version
        # *** END: Pre-process state_metrics ***

        # Serialization logic (more explicit for state_metrics)
        serializable_results = {}
        for key, value in results.items():
            if key == 'state_metrics':
                # Explicitly build the serializable dict for state_metrics
                temp_metrics_dict = {}
                if isinstance(value, dict): # Value should be the cleaned_metrics dict
                    for state_idx_str, state_data_dict in value.items():
                        # state_data_dict should contain simple types after pre-processing
                        temp_metrics_dict[state_idx_str] = state_data_dict 
                serializable_results[key] = temp_metrics_dict # Assign the explicitly built dict
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
    avg_duration = results.get('avg_duration', 0) / 12  # Convert months to years
    avg_training_cost = results.get('avg_training_cost', 0)
    avg_total_payments = results.get('avg_total_payments', 0)
    avg_net_cash_flow = results.get('avg_net_cash_flow', 0)
    avg_roi = results.get('avg_roi', 0)
    
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
                    ], style={'marginBottom': '10px'}),
                    html.Div([
                        html.P("Average Duration:", style={'fontWeight': 'bold'}),
                        html.P(f"{avg_duration:.1f} years")
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
                    ], style={'marginBottom': '10px'}),
                    html.Div([
                        html.P("Avg ROI:", style={'fontWeight': 'bold'}),
                        html.P(f"{avg_roi:.1f}%")
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
    
    # Extract data (state_metrics should be pre-cleaned)
    state_distribution = results.get('state_distribution', {})
    state_metrics = results.get('state_metrics', {})
    
    # Prepare data for plotting state distribution
    states_dist = []
    counts = []
    percentages = []
    
    # *** REMOVE the get_state_name helper function definition ***
    # def get_state_name(state_idx, metrics_dict): ... (removed)

    total = sum(v for v in state_distribution.values() if isinstance(v, (int, float))) 

    for state_idx, count in state_distribution.items():
        # Directly access the cleaned name from the pre-processed state_metrics
        metric_info = state_metrics.get(str(state_idx), {}) # Get metrics for this index
        state_name = metric_info.get('name', f"State {state_idx}") # Access cleaned name
            
        states_dist.append(state_name)
        counts.append(count)
        percentages.append(f"{(count/total)*100:.1f}%" if total > 0 else "0.0%")
    
    # Create state distribution table
    state_table = html.Div([
        html.H5("Final State Distribution", style={'textAlign': 'center', 'marginBottom': '15px'}),
        dash_table.DataTable(
            data=[
                {"State": state, "Count": count, "Percentage": pct}
                for state, count, pct in zip(states_dist, counts, percentages) # Use states_dist
            ],
            columns=[
                {"name": "State", "id": "State"},
                {"name": "Count", "id": "Count"},
                {"name": "Percentage", "id": "Percentage"}
            ],
            style_cell={'textAlign': 'center', 'padding': '10px'},
            style_header={
                'backgroundColor': 'rgb(230, 230, 230)',
                'fontWeight': 'bold'
            },
            style_data_conditional=[
                {
                    'if': {'row_index': 'odd'},
                    'backgroundColor': 'rgb(248, 248, 248)'
                }
            ]
        )
    ], style={'marginBottom': '30px'})
    
    # Create state metrics table
    metrics_rows = []
    state_order_map = {name: i for i, name in enumerate(states_dist)} # Map name to original sort order
    
    for state_idx, metrics in state_metrics.items():
        # Directly access the cleaned name
        state_name = metrics.get('name', f"State {state_idx}")
            
        # Extract numeric values safely (metrics should be dict)
        if isinstance(metrics, dict):
            avg_salary = metrics.get('avg_salary', 0)
            avg_salary = avg_salary if isinstance(avg_salary, (int, float)) else 0
            
            avg_payment = metrics.get('avg_payment', 0)
            avg_payment = avg_payment if isinstance(avg_payment, (int, float)) else 0
            
            active_months = metrics.get('active_months', 0)
            active_months = active_months if isinstance(active_months, (int, float)) else 0
            
            metrics_rows.append({
                "State": state_name,
                "Avg Salary": f"${avg_salary:.2f}",
                "Avg Payment": f"${avg_payment:.2f}",
                "Active Months": int(active_months) 
            })
        else: # Should not happen if pre-processing worked
             metrics_rows.append({
                "State": f"State {state_idx} (error)",
                "Avg Salary": "$0.00",
                "Avg Payment": "$0.00",
                "Active Months": 0
            })
            
    # Sort metrics_rows based on the original state order from distribution
    metrics_rows = sorted(metrics_rows, key=lambda row: state_order_map.get(row['State'], float('inf')))

    metrics_table = html.Div([
        html.H5("State Metrics", style={'textAlign': 'center', 'marginBottom': '15px'}),
        dash_table.DataTable(
            data=metrics_rows,
            columns=[
                {"name": "State", "id": "State"},
                {"name": "Avg Salary", "id": "Avg Salary"},
                {"name": "Avg Payment", "id": "Avg Payment"},
                {"name": "Active Months", "id": "Active Months"}
            ],
            style_cell={'textAlign': 'center', 'padding': '10px'},
            style_header={
                'backgroundColor': 'rgb(230, 230, 230)',
                'fontWeight': 'bold'
            },
            style_data_conditional=[
                {
                    'if': {'row_index': 'odd'},
                    'backgroundColor': 'rgb(248, 248, 248)'
                }
            ]
        )
    ])
    
    # Combine elements
    return html.Div([
        html.H4("Simulation Overview", style={'textAlign': 'center', 'marginBottom': '20px'}),
        state_table,
        metrics_table
    ])

# Callback to update detailed data
@app.callback(
    Output("detailed-data", "children"),
    [Input("simulation-results-store", "data")]
)
def update_detailed_data(results):
    if not results:
        return "Run a simulation to see results"
    
    # Extract config data
    config = results.get('config', {})
    
    # Create config table
    config_data = [
        {"Parameter": "Basic Training Cost", "Value": f"${config.get('basic_training_cost', 0):.2f}"},
        {"Parameter": "Basic Training Dropout Rate", "Value": f"{config.get('basic_training_dropout_rate', 0)*100:.1f}%"},
        {"Parameter": "Basic Training Duration", "Value": f"{config.get('basic_training_duration', 0)} months"},
        {"Parameter": "Include Advanced Training", "Value": "Yes" if config.get('include_advanced_training', True) else "No"}
    ]
    
    if config.get('include_advanced_training', True):
        config_data.extend([
            {"Parameter": "Advanced Training Cost", "Value": f"${config.get('advanced_training_cost', 0):.2f}"},
            {"Parameter": "Advanced Training Dropout Rate", "Value": f"{config.get('advanced_training_dropout_rate', 0)*100:.1f}%"},
            {"Parameter": "Advanced Training Duration", "Value": f"{config.get('advanced_training_duration', 0)} months"}
        ])
    
    config_data.extend([
        {"Parameter": "Number of Additional Cruises", "Value": f"{config.get('num_additional_cruises', 0)}"},
        {"Parameter": "First Cruise Base Salary", "Value": f"${config.get('first_cruise_base_salary', 0):.2f}"},
        {"Parameter": "First Cruise Dropout Rate", "Value": f"{config.get('first_cruise_dropout_rate', 0)*100:.1f}%"},
        {"Parameter": "First Cruise Salary Variation", "Value": f"{config.get('first_cruise_salary_variation', 0):.1f}%"},
        {"Parameter": "First Cruise Duration", "Value": f"{config.get('first_cruise_duration', 0)} months"},
        {"Parameter": "First Cruise Payment Fraction", "Value": f"{config.get('first_cruise_payment_fraction', 0)*100:.1f}%"},
        {"Parameter": "Subsequent Cruise Dropout Rate", "Value": f"{config.get('subsequent_cruise_dropout_rate', 0)*100:.1f}%"},
        {"Parameter": "Subsequent Cruise Salary Increase", "Value": f"{config.get('subsequent_cruise_salary_increase', 0):.1f}%"},
        {"Parameter": "Subsequent Cruise Salary Variation", "Value": f"{config.get('subsequent_cruise_salary_variation', 0):.1f}%"},
        {"Parameter": "Subsequent Cruise Duration", "Value": f"{config.get('subsequent_cruise_duration', 0)} months"},
        {"Parameter": "Subsequent Cruise Payment Fraction", "Value": f"{config.get('subsequent_cruise_payment_fraction', 0)*100:.1f}%"}
    ])
    
    config_table = html.Div([
        html.H5("Simulation Configuration", style={'textAlign': 'center', 'marginBottom': '15px'}),
        dash_table.DataTable(
            data=config_data,
            columns=[
                {"name": "Parameter", "id": "Parameter"},
                {"name": "Value", "id": "Value"}
            ],
            style_cell={'textAlign': 'left', 'padding': '10px'},
            style_header={
                'backgroundColor': 'rgb(230, 230, 230)',
                'fontWeight': 'bold'
            },
            style_data_conditional=[
                {
                    'if': {'row_index': 'odd'},
                    'backgroundColor': 'rgb(248, 248, 248)'
                }
            ]
        )
    ], style={'marginBottom': '30px'})
    
    # Create ROI metrics table
    roi_data = [
        {"Metric": "Average Training Cost", "Value": f"${results.get('avg_training_cost', 0):.2f}"},
        {"Metric": "Average Total Payments", "Value": f"${results.get('avg_total_payments', 0):.2f}"},
        {"Metric": "Average Net Cash Flow", "Value": f"${results.get('avg_net_cash_flow', 0):.2f}"},
        {"Metric": "Average ROI", "Value": f"{results.get('avg_roi', 0):.1f}%"},
        {"Metric": "ROI Standard Deviation", "Value": f"{results.get('roi_std', 0):.1f}%"},
        {"Metric": "ROI 10th Percentile", "Value": f"{results.get('roi_10th', 0):.1f}%"},
        {"Metric": "ROI 90th Percentile", "Value": f"{results.get('roi_90th', 0):.1f}%"}
    ]
    
    roi_table = html.Div([
        html.H5("Financial Metrics", style={'textAlign': 'center', 'marginBottom': '15px'}),
        dash_table.DataTable(
            data=roi_data,
            columns=[
                {"name": "Metric", "id": "Metric"},
                {"name": "Value", "id": "Value"}
            ],
            style_cell={'textAlign': 'left', 'padding': '10px'},
            style_header={
                'backgroundColor': 'rgb(230, 230, 230)',
                'fontWeight': 'bold'
            },
            style_data_conditional=[
                {
                    'if': {'row_index': 'odd'},
                    'backgroundColor': 'rgb(248, 248, 248)'
                }
            ]
        )
    ])
    
    return html.Div([
        html.H4("Detailed Simulation Data", style={'textAlign': 'center', 'marginBottom': '20px'}),
        config_table,
        roi_table
    ])

# Callback to update state tracking content
@app.callback(
    Output("state-tracking-content", "children"),
    [Input("simulation-results-store", "data")]
)
def update_state_tracking(results):
    if not results:
        return "Run a simulation to see results"
    
    # Extract data (state_metrics should be pre-cleaned)
    state_metrics = results.get('state_metrics', {})
    state_distribution = results.get('state_distribution', {})
    
    # *** REMOVE the get_state_name helper function definition ***
    # def get_state_name(state_idx, metrics_dict): ... (removed)

    # Create a list to store the data for each state
    state_data = []
    
    total_simulations = sum(v for v in state_distribution.values() if isinstance(v, (int, float)))
    
    # Create a row for each state
    for state_idx, metrics in sorted(state_metrics.items(), key=lambda x: int(x[0])):
        # Directly access the cleaned name
        state_name = metrics.get('name', f"State {state_idx}") # Access cleaned name
        active_careers = state_distribution.get(str(state_idx), 0) # Use string index for distribution

        # Ensure metrics is a dictionary before accessing sub-keys
        if isinstance(metrics, dict):
            # Calculate dropouts 
            # ... (dropout calculation logic remains the same) ...
            dropouts = 0
            if int(state_idx) > 0:
                prev_state_idx_int = int(state_idx) - 1
                previous_state_careers = 0
                if str(prev_state_idx_int) in state_distribution:
                     previous_state_careers = state_distribution.get(str(prev_state_idx_int), 0)
                completed_previous = state_distribution.get(str(prev_state_idx_int), active_careers)
                dropouts = max(0, completed_previous - active_careers)
                
            # Get average salary of active careers
            avg_salary = metrics.get('avg_salary', 0)
            if not isinstance(avg_salary, (int, float)): avg_salary = 0
            
            # Calculate cash flow for this state
            cash_flow = 0
            if "Training" in state_name:
                training_cost = 0
                if "Basic" in state_name:
                    training_cost = results.get('config', {}).get('basic_training_cost', 0)
                elif "Advanced" in state_name:
                    training_cost = results.get('config', {}).get('advanced_training_cost', 0)
                cash_flow = -training_cost
            else:
                payment = metrics.get('avg_payment', 0)
                if not isinstance(payment, (int, float)): payment = 0
                active_months = metrics.get('active_months', 0)
                if not isinstance(active_months, (int, float)): active_months = 0
                cash_flow = payment * active_months

            state_data.append({
                "State": state_name, # Use cleaned name
                "Active Careers": active_careers,
                "Dropouts": dropouts,
                "Average Salary": f"${avg_salary:.2f}",
                "State Cash Flow Per Student": f"${cash_flow:.2f}",
                "Cash Flow Value": cash_flow
            })
        else:
             # Should not happen if pre-processing worked
             state_data.append({
                "State": f"State {state_idx} (error)",
                "Active Careers": active_careers,
                "Dropouts": 0,
                "Average Salary": "$0.00",
                "State Cash Flow Per Student": "$0.00",
                "Cash Flow Value": 0
            })

    
    # Create the table (No changes needed here)
    # ...
    tracking_table = html.Div([
        html.H5("State-by-State Tracking", style={'textAlign': 'center', 'marginBottom': '15px'}),
        dash_table.DataTable(
            data=[{k: v for k, v in item.items() if k != "Cash Flow Value"} for item in state_data], # Filter out sort key
            columns=[
                {"name": "State", "id": "State"},
                {"name": "Active Careers", "id": "Active Careers"},
                {"name": "Dropouts (Est.)", "id": "Dropouts"}, # Updated name
                {"name": "Average Salary", "id": "Average Salary"},
                {"name": "Cash Flow/Student", "id": "State Cash Flow Per Student"} # Updated name
            ],
            style_cell={'textAlign': 'center', 'padding': '10px'},
            style_header={
                'backgroundColor': 'rgb(230, 230, 230)',
                'fontWeight': 'bold'
            },
            style_data_conditional=[
                {
                    'if': {'row_index': 'odd'},
                    'backgroundColor': 'rgb(248, 248, 248)'
                },
                {
                    'if': {'filter_query': '{State Cash Flow Per Student} contains "-"'}, # Updated column ID
                    'color': 'red'
                },
                {
                    'if': {'filter_query': '{State Cash Flow Per Student} = "$0.00"'}, # Exact match
                    'color': 'grey'
                },
                {
                    'if': {'filter_query': '{State Cash Flow Per Student} contains "$" && !({State Cash Flow Per Student} contains "-") && !({State Cash Flow Per Student} = "$0.00")'}, # Updated column ID
                    'color': 'green'
                }
            ]
        )
    ])
    
    # Calculate summary metrics (No changes needed)
    # ...
    total_training_costs = sum(item["Cash Flow Value"] * item["Active Careers"] for item in state_data if item["Cash Flow Value"] < 0)
    total_payments = sum(item["Cash Flow Value"] * item["Active Careers"] for item in state_data if item["Cash Flow Value"] > 0)
    net_cash_flow = total_payments + total_training_costs
    
    # Create cash flow summary (No changes needed)
    # ...
    cash_flow_summary = html.Div([
        html.H5("Cash Flow Summary", style={'textAlign': 'center', 'marginTop': '30px', 'marginBottom': '15px'}),
        html.Div([
            html.Div([
                html.P("Total Training Costs:", style={'fontWeight': 'bold'}),
                html.P(f"${abs(total_training_costs):.2f}", style={'color': 'red'})
            ], style={'display': 'inline-block', 'width': '33%', 'textAlign': 'center'}),
            
            html.Div([
                html.P("Total Payments Received:", style={'fontWeight': 'bold'}),
                html.P(f"${total_payments:.2f}", style={'color': 'green'})
            ], style={'display': 'inline-block', 'width': '33%', 'textAlign': 'center'}),
            
            html.Div([
                html.P("Net Cash Flow:", style={'fontWeight': 'bold'}),
                html.P(f"${net_cash_flow:.2f}", 
                      style={'color': 'green' if net_cash_flow >= 0 else 'red'})
            ], style={'display': 'inline-block', 'width': '33%', 'textAlign': 'center'})
        ], style={'border': '1px solid #ddd', 'padding': '15px', 'borderRadius': '5px', 'backgroundColor': '#f9f9f9'})
    ])

    # Create the cash flow visualization (No changes needed)
    # ...
    cash_flow_fig = go.Figure()
    state_total_cash_flow = [item['Cash Flow Value'] * item['Active Careers'] for item in state_data]
    
    cash_flow_fig.add_trace(go.Bar(
        x=[item['State'] for item in state_data],
        y=state_total_cash_flow, # Use total cash flow for the state
        name="Total Cash Flow",
        marker_color=['red' if cf < 0 else 'green' for cf in state_total_cash_flow],
        text=[f"${abs(cf):.2f}" for cf in state_total_cash_flow],
        textposition='auto'
    ))
    
    cash_flow_fig.update_layout(
        title='Total Cash Flow by State', # Updated title
        xaxis_title='State',
        yaxis_title='Total Cash Flow ($)', # Updated axis title
        template='plotly_white'
    )
    
    cash_flow_graph = html.Div([
        html.H5("Cash Flow Visualization", style={'textAlign': 'center', 'marginTop': '30px', 'marginBottom': '15px'}),
        dcc.Graph(figure=cash_flow_fig)
    ])
    
    return html.Div([
        html.H4("State Tracking Analysis", style={'textAlign': 'center', 'marginBottom': '20px'}),
        html.P("This table shows detailed tracking information for each state in the simulation:", 
               style={'marginBottom': '15px'}),
        html.Ul([
            html.Li("Active Careers: Number of simulated careers currently in this state"),
            html.Li("Dropouts: Estimated number of simulations with dropouts during this state"),
            html.Li("Average Salary: Average monthly salary in this state"),
            html.Li("Total Cash Flow: Negative for training costs, positive for payments received (Aggregated across active careers)") # Updated description
        ], style={'marginBottom': '20px'}),
        tracking_table,
        cash_flow_summary,
        cash_flow_graph
    ])

# Callbacks for scenario comparison functionality
@app.callback(
    [Output("saved-scenarios-store", "data"),
     Output("saved-scenarios-list", "children")],
    [Input("save-scenario-button", "n_clicks"),
     Input("clear-scenarios-button", "n_clicks")],
    [State("scenario-name-input", "value"),
     State("simulation-results-store", "data"),
     State("saved-scenarios-store", "data")]
)
def manage_saved_scenarios(save_clicks, clear_clicks, scenario_name, current_results, saved_scenarios):
    ctx = dash.callback_context
    if not ctx.triggered:
        # Initial load - return empty list
        return {}, []
    
    # Initialize saved_scenarios if it doesn't exist or is not a dict
    if saved_scenarios is None or not isinstance(saved_scenarios, dict):
        saved_scenarios = {}
    
    # Handle clear button click
    if ctx.triggered[0]['prop_id'] == 'clear-scenarios-button.n_clicks' and clear_clicks:
        return {}, []
    
    # Handle save button click
    if ctx.triggered[0]['prop_id'] == 'save-scenario-button.n_clicks' and save_clicks:
        if not scenario_name or not current_results:
            return saved_scenarios, [html.Div("Please enter a scenario name and run a simulation first.")]
        
        # Add current results to saved scenarios
        saved_scenarios[scenario_name] = current_results
        
        # Create list of saved scenarios
        scenario_items = []
        for name in saved_scenarios:
            roi_value = saved_scenarios[name].get('avg_roi', 0)
            scenario_items.append(html.Div([
                html.Button(
                    f"{name} - ROI: {roi_value:.1f}%",
                    id={'type': 'scenario-button', 'index': name},
                    n_clicks=0,
                    style={
                        'backgroundColor': '#e1f5fe',
                        'border': '1px solid #81d4fa',
                        'borderRadius': '4px',
                        'padding': '8px 12px',
                        'marginRight': '10px',
                        'cursor': 'pointer',
                        'width': '100%',
                        'textAlign': 'left'
                    }
                )
            ], style={'marginBottom': '10px'}))
        
        return saved_scenarios, scenario_items
    
    # Return current state if no action taken
    return saved_scenarios, dash.no_update

@app.callback(
    Output("scenario-comparison-results", "children"),
    [Input("compare-scenarios-button", "n_clicks")],
    [State("saved-scenarios-store", "data")]
)
def compare_scenarios(n_clicks, saved_scenarios):
    if not n_clicks:
        return html.Div("Save scenarios to compare them.")
    
    # Initialize saved_scenarios if it doesn't exist or is not a dict
    if saved_scenarios is None or not isinstance(saved_scenarios, dict):
        saved_scenarios = {}
    
    # Get all saved scenarios
    selected_scenarios = []
    for name, data in saved_scenarios.items():
        selected_scenarios.append({
            "name": name,
            "data": data
        })
    
    if not selected_scenarios:
        return html.Div("No scenarios available for comparison.")
    
    # Create comparison elements
    comparison_elements = []
    
    # 1. ROI Comparison Chart
    roi_data = []
    for scenario in selected_scenarios:
        roi_data.append({
            'Scenario': scenario['name'],
            'ROI': scenario['data'].get('avg_roi', 0),
            'Net Cash Flow': scenario['data'].get('avg_net_cash_flow', 0)
        })
    
    roi_df = pd.DataFrame(roi_data)
    
    roi_fig = go.Figure()
    roi_fig.add_trace(go.Bar(
        x=roi_df['Scenario'],
        y=roi_df['ROI'],
        name='ROI (%)',
        marker_color='rgb(26, 118, 255)',
        text=roi_df['ROI'].apply(lambda x: f"{x:.1f}%"),
        textposition='auto'
    ))
    
    roi_fig.update_layout(
        title='ROI Comparison by Scenario',
        xaxis_title='Scenario',
        yaxis_title='ROI (%)',
        template='plotly_white'
    )
    
    comparison_elements.append(dcc.Graph(figure=roi_fig))
    
    # 2. Net Cash Flow Comparison
    flow_fig = go.Figure()
    flow_fig.add_trace(go.Bar(
        x=roi_df['Scenario'],
        y=roi_df['Net Cash Flow'],
        name='Net Cash Flow ($)',
        marker_color='rgb(55, 83, 109)',
        text=roi_df['Net Cash Flow'].apply(lambda x: f"${x:.2f}"),
        textposition='auto'
    ))
    
    flow_fig.update_layout(
        title='Net Cash Flow Comparison by Scenario',
        xaxis_title='Scenario',
        yaxis_title='Net Cash Flow ($)',
        template='plotly_white'
    )
    
    comparison_elements.append(dcc.Graph(figure=flow_fig))
    
    # 3. Key Metrics Comparison Table
    metrics_data = []
    for scenario in selected_scenarios:
        metrics_data.append({
            'Scenario': scenario['name'],
            'ROI': f"{scenario['data'].get('avg_roi', 0):.1f}%",
            'Training Cost': f"${scenario['data'].get('avg_training_cost', 0):.2f}",
            'Total Payments': f"${scenario['data'].get('avg_total_payments', 0):.2f}",
            'Net Cash Flow': f"${scenario['data'].get('avg_net_cash_flow', 0):.2f}",
            'Completion Rate': f"{scenario['data'].get('completion_rate', 0):.1f}%",
            'Dropout Rate': f"{scenario['data'].get('dropout_rate', 0):.1f}%"
        })
    
    metrics_table = dash_table.DataTable(
        data=metrics_data,
        columns=[{"name": col, "id": col} for col in metrics_data[0].keys()],
        style_cell={'textAlign': 'center', 'padding': '10px'},
        style_header={
            'backgroundColor': 'rgb(230, 230, 230)',
            'fontWeight': 'bold'
        },
        style_data_conditional=[
            {
                'if': {'row_index': 'odd'},
                'backgroundColor': 'rgb(248, 248, 248)'
            }
        ]
    )
    
    comparison_elements.append(html.Div([
        html.H5("Key Metrics by Scenario", style={'marginTop': '30px', 'marginBottom': '15px'}),
        metrics_table
    ]))
    
    return html.Div(comparison_elements)

# Callback for the State Progression content
@app.callback(
    Output("state-progression-content", "children"),
    [Input("simulation-results-store", "data")]
)
def update_state_progression(results):
    if not results:
        return "Run a simulation to see results"
    
    # Get state metrics and config
    state_metrics = results.get('state_metrics', {})
    config = results.get('config', {})
    
    # Create data for state progression table
    progression_data = []
    
    # Default starting value is number of simulations
    total_entered = config.get('num_students', 100)
    
    for state_idx, metrics in sorted(state_metrics.items(), key=lambda x: int(x[0])):
        if isinstance(metrics, dict):
            state_name = metrics.get('name', f"State {state_idx}")
            
            # For the first state, all careers enter
            if int(state_idx) == 0:
                entered = total_entered
            else:
                # For subsequent states, the number who entered is the number who completed the previous state
                prev_state_idx = str(int(state_idx) - 1)
                if prev_state_idx in state_metrics:
                    entered = progression_data[-1].get('completed', 0)
                else:
                    entered = 0
            
            # Calculate dropouts and completions based on configured dropout rates
            dropout_rate = 0
            if "Basic Training" in state_name:
                dropout_rate = config.get('basic_training_dropout_rate', 0) * 100
            elif "Advanced Training" in state_name:
                dropout_rate = config.get('advanced_training_dropout_rate', 0) * 100
            elif "First Cruise" in state_name:
                dropout_rate = config.get('first_cruise_dropout_rate', 0) * 100
            else:
                dropout_rate = config.get('subsequent_cruise_dropout_rate', 0) * 100
            
            dropouts = round(entered * (dropout_rate / 100))
            completed = entered - dropouts
            
            progression_data.append({
                'state': state_name,
                'entered': entered,
                'completed': completed,
                'dropouts': dropouts,
                'dropout_rate': dropout_rate
            })
    
    # Create a table showing the progression
    table = dash_table.DataTable(
        id='state-progression-table',
        columns=[
            {'name': 'State', 'id': 'state'},
            {'name': 'Entered', 'id': 'entered'},
            {'name': 'Completed', 'id': 'completed'},
            {'name': 'Dropouts', 'id': 'dropouts'},
            {'name': 'Dropout Rate', 'id': 'dropout_rate', 'type': 'numeric', 'format': {'specifier': '.1f'}}
        ],
        data=[
            {
                'state': row['state'],
                'entered': row['entered'],
                'completed': row['completed'],
                'dropouts': row['dropouts'],
                'dropout_rate': row['dropout_rate']
            }
            for row in progression_data
        ],
        style_cell={'textAlign': 'center', 'padding': '10px'},
        style_header={
            'backgroundColor': 'rgb(230, 230, 230)',
            'fontWeight': 'bold'
        },
        style_data_conditional=[
            {
                'if': {'row_index': 'odd'},
                'backgroundColor': 'rgb(248, 248, 248)'
            }
        ]
    )
    
    # Create a bar chart visualization
    progression_fig = go.Figure()
    
    # Add bars for entered, completed, and dropouts
    states = [row['state'] for row in progression_data]
    
    progression_fig.add_trace(go.Bar(
        x=states,
        y=[row['entered'] for row in progression_data],
        name='Entered',
        marker_color='rgb(53, 83, 255)',
        hovertemplate="State: %{x}<br>Entered: %{y}<extra></extra>"
    ))
    
    progression_fig.add_trace(go.Bar(
        x=states,
        y=[row['completed'] for row in progression_data],
        name='Completed',
        marker_color='rgb(55, 183, 109)',
        hovertemplate="State: %{x}<br>Completed: %{y}<extra></extra>"
    ))
    
    progression_fig.add_trace(go.Bar(
        x=states,
        y=[row['dropouts'] for row in progression_data],
        name='Dropouts',
        marker_color='rgb(219, 64, 82)',
        hovertemplate="State: %{x}<br>Dropouts: %{y}<extra></extra>"
    ))
    
    # Add a line for dropout rate
    progression_fig.add_trace(go.Scatter(
        x=states,
        y=[row['dropout_rate'] for row in progression_data],
        name='Dropout Rate (%)',
        mode='lines+markers',
        yaxis='y2',
        line=dict(color='rgb(128, 0, 128)', width=2),
        marker=dict(size=8),
        hovertemplate="State: %{x}<br>Dropout Rate: %{y:.1f}%<extra></extra>"
    ))
    
    # Update layout with dual y-axes
    progression_fig.update_layout(
        title='Career Progression through States',
        xaxis_title='State',
        yaxis_title='Number of Simulations',
        yaxis2=dict(
            title='Dropout Rate (%)',
            overlaying='y',
            side='right',
            range=[0, 20],
            ticksuffix='%'
        ),
        barmode='group',
        template='plotly_white',
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1
        )
    )
    
    return html.Div([
        html.H4("State Progression Analysis", style={'textAlign': 'center', 'marginBottom': '20px'}),
        html.Div([
            html.P("This analysis shows how careers progress through the training and cruise states:"),
            html.Ul([
                html.Li("Entered: Number of simulations that entered each state"),
                html.Li("Completed: Number of simulations that successfully completed the state"),
                html.Li("Dropouts: Number of simulations with dropouts during the state"),
                html.Li("Dropout Rate: Percentage of careers that dropped out of those who entered")
            ])
        ], style={'marginBottom': '20px'}),
        dcc.Graph(figure=progression_fig, style={'marginBottom': '30px'}),
        html.H5("State Progression Data", style={'textAlign': 'center', 'marginBottom': '15px'}),
        table
    ])

# Callback for the Financial Analysis content
@app.callback(
    Output("financial-analysis-content", "children"),
    [Input("simulation-results-store", "data")]
)
def update_financial_analysis(results):
    if not results:
        return "Run a simulation to see results"
    
    # Extract financial metrics
    avg_training_cost = results.get('avg_training_cost', 0)
    avg_total_payments = results.get('avg_total_payments', 0)
    net_returns = avg_total_payments - avg_training_cost
    roi = results.get('avg_roi', 0)
    
    # Get breakeven calculation from results or calculate it
    # For simplicity, we'll calculate based on avg monthly payments
    state_metrics = results.get('state_metrics', {})
    avg_monthly_payment = 0
    active_months = 0
    
    for _, metrics in state_metrics.items():
        if isinstance(metrics, dict):
            avg_monthly_payment += metrics.get('avg_payment', 0) * metrics.get('active_months', 0)
            active_months += metrics.get('active_months', 0)
    
    avg_monthly_payment = avg_monthly_payment / active_months if active_months > 0 else 0
    
    # Calculate breakeven month
    breakeven_month = 0
    if avg_monthly_payment > 0:
        breakeven_month = round(avg_training_cost / avg_monthly_payment)
    
    # Calculate other metrics
    repayment_rate = (avg_total_payments / avg_training_cost * 100) if avg_training_cost > 0 else 0
    
    # Calculate annual IRR (roughly)
    avg_duration_years = results.get('avg_duration', 0) / 12
    annual_irr = 0
    if avg_duration_years > 0 and avg_training_cost > 0 and avg_total_payments > 0:
        annual_irr = (pow(avg_total_payments / avg_training_cost, 1/avg_duration_years) - 1) * 100
    
    # Financial summary cards
    financial_summary = html.Div([
        html.H4("Financial Summary", style={'textAlign': 'center', 'marginBottom': '20px'}),
        
        # Key metrics cards in a row
        html.Div([
            # Total Training Costs
            html.Div([
                html.H5("Total Training Costs", style={'textAlign': 'center', 'marginBottom': '10px'}),
                html.Div(f"${avg_training_cost:.2f}", style={
                    'fontSize': '24px', 
                    'textAlign': 'center',
                    'color': '#e53935'
                })
            ], style={'width': '30%', 'display': 'inline-block', 'backgroundColor': '#f1f1f1', 'padding': '15px', 'borderRadius': '5px', 'marginRight': '5%'}),
            
            # Total Payments
            html.Div([
                html.H5("Total Payments Made", style={'textAlign': 'center', 'marginBottom': '10px'}),
                html.Div(f"${avg_total_payments:.2f}", style={
                    'fontSize': '24px', 
                    'textAlign': 'center',
                    'color': '#43a047'
                })
            ], style={'width': '30%', 'display': 'inline-block', 'backgroundColor': '#f1f1f1', 'padding': '15px', 'borderRadius': '5px', 'marginRight': '5%'}),
            
            # Net Returns
            html.Div([
                html.H5("Net Returns", style={'textAlign': 'center', 'marginBottom': '10px'}),
                html.Div(f"${net_returns:.2f}", style={
                    'fontSize': '24px', 
                    'textAlign': 'center',
                    'color': '#43a047' if net_returns >= 0 else '#e53935'
                })
            ], style={'width': '30%', 'display': 'inline-block', 'backgroundColor': '#f1f1f1', 'padding': '15px', 'borderRadius': '5px'})
        ], style={'marginBottom': '20px'}),
        
        # Performance metrics in a row
        html.Div([
            # Breakeven Month
            html.Div([
                html.H5("Breakeven Month", style={'textAlign': 'center', 'marginBottom': '10px'}),
                html.Div(f"{breakeven_month}" if breakeven_month > 0 else "Not reached", style={
                    'fontSize': '24px', 
                    'textAlign': 'center'
                })
            ], style={'width': '23%', 'display': 'inline-block', 'backgroundColor': '#f1f1f1', 'padding': '15px', 'borderRadius': '5px', 'marginRight': '2%'}),
            
            # Repayment Rate
            html.Div([
                html.H5("Repayment Rate", style={'textAlign': 'center', 'marginBottom': '10px'}),
                html.Div(f"{repayment_rate:.1f}%", style={
                    'fontSize': '24px', 
                    'textAlign': 'center',
                    'color': '#43a047' if repayment_rate >= 100 else '#ff9800'
                })
            ], style={'width': '23%', 'display': 'inline-block', 'backgroundColor': '#f1f1f1', 'padding': '15px', 'borderRadius': '5px', 'marginRight': '2%'}),
            
            # Annual IRR
            html.Div([
                html.H5("Annual IRR", style={'textAlign': 'center', 'marginBottom': '10px'}),
                html.Div(f"{annual_irr:.1f}%", style={
                    'fontSize': '24px', 
                    'textAlign': 'center',
                    'color': '#43a047' if annual_irr > 0 else '#e53935'
                })
            ], style={'width': '23%', 'display': 'inline-block', 'backgroundColor': '#f1f1f1', 'padding': '15px', 'borderRadius': '5px', 'marginRight': '2%'}),
            
            # ROI
            html.Div([
                html.H5("Return on Investment", style={'textAlign': 'center', 'marginBottom': '10px'}),
                html.Div(f"{roi:.1f}%", style={
                    'fontSize': '24px', 
                    'textAlign': 'center',
                    'color': '#43a047' if roi > 0 else '#e53935'
                })
            ], style={'width': '23%', 'display': 'inline-block', 'backgroundColor': '#f1f1f1', 'padding': '15px', 'borderRadius': '5px'})
        ], style={'marginBottom': '30px'})
    ])
    
    # Create a pie chart showing the financial breakdown
    financial_fig = go.Figure()
    financial_fig.add_trace(go.Pie(
        labels=['Training Costs', 'Net Returns'],
        values=[avg_training_cost, net_returns if net_returns > 0 else 0],
        marker_colors=['#e53935', '#43a047'],
        hole=.4,
        textinfo='label+percent',
        hoverinfo='label+value',
        texttemplate='%{label}<br>%{percent}'
    ))
    
    financial_fig.update_layout(
        title='Financial Breakdown',
        showlegend=False,
        template='plotly_white',
        annotations=[{
            'text': f'${avg_total_payments:.0f}',
            'showarrow': False,
            'font_size': 20
        }]
    )
    
    # Create a bar chart showing monthly costs vs payments
    monthly_fig = go.Figure()
    
    # Aggregate state data
    state_names = []
    state_payments = []
    state_costs = []
    
    for state_idx, metrics in sorted(state_metrics.items(), key=lambda x: int(x[0])):
        if isinstance(metrics, dict):
            state_names.append(metrics.get('name', f"State {state_idx}"))
            # Calculate total payments for this state
            total_payment = metrics.get('avg_payment', 0) * metrics.get('active_months', 0)
            state_payments.append(total_payment)
            
            # Get training cost for this state
            training_cost = 0
            state_name = metrics.get('name', "")
            if "Basic Training" in state_name:
                training_cost = results.get('config', {}).get('basic_training_cost', 0)
            elif "Advanced Training" in state_name:
                training_cost = results.get('config', {}).get('advanced_training_cost', 0)
            
            state_costs.append(training_cost)
    
    monthly_fig.add_trace(go.Bar(
        x=state_names,
        y=state_costs,
        name='Training Costs',
        marker_color='#e53935',
        hovertemplate="State: %{x}<br>Training Cost: $%{y:.2f}<extra></extra>"
    ))
    
    monthly_fig.add_trace(go.Bar(
        x=state_names,
        y=state_payments,
        name='Payments Received',
        marker_color='#43a047',
        hovertemplate="State: %{x}<br>Payments: $%{y:.2f}<extra></extra>"
    ))
    
    monthly_fig.update_layout(
        title='Costs vs Payments by State',
        xaxis_title='State',
        yaxis_title='Amount ($)',
        barmode='group',
        template='plotly_white',
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1
        )
    )
    
    return html.Div([
        financial_summary,
        html.Div([
            html.Div([
                dcc.Graph(figure=financial_fig)
            ], style={'width': '50%', 'display': 'inline-block', 'verticalAlign': 'top'}),
            html.Div([
                dcc.Graph(figure=monthly_fig)
            ], style={'width': '50%', 'display': 'inline-block', 'verticalAlign': 'top'})
        ]),
        
        # Additional explanation
        html.Div([
            html.H5("Performance Metrics Explained", style={'marginTop': '20px', 'marginBottom': '10px'}),
            html.Ul([
                html.Li([
                    html.Span("Breakeven Month: ", style={'fontWeight': 'bold'}),
                    "The month when total payments received equal the training costs."
                ]),
                html.Li([
                    html.Span("Repayment Rate: ", style={'fontWeight': 'bold'}),
                    "The percentage of training costs recovered through payments."
                ]),
                html.Li([
                    html.Span("Annual IRR: ", style={'fontWeight': 'bold'}),
                    "Internal Rate of Return, annualized to show yearly performance."
                ]),
                html.Li([
                    html.Span("Return on Investment: ", style={'fontWeight': 'bold'}),
                    "Total net returns as a percentage of training costs."
                ])
            ])
        ], style={'marginTop': '20px'})
    ])

# Callback for the Monthly Metrics content
@app.callback(
    Output("monthly-metrics-content", "children"),
    [Input("simulation-results-store", "data")]
)
def update_monthly_metrics(results):
    if not results:
        return "Run a simulation to see results"
    
    state_metrics = results.get('state_metrics', {})
        
    # *** Start - Explicit Name Extraction Logic ***
    metrics_data = []
    for state_idx_str, metric_info in sorted(state_metrics.items(), key=lambda item: int(item[0])):
        # Default values
        state_name = f"State {state_idx_str}"
        avg_salary = 0
        avg_payment = 0
        active_months = 0

        # Try extracting data, focusing on getting the simple name string
        if isinstance(metric_info, dict):
            # Attempt to get the simple name string
            name_raw = metric_info.get('name')
            if isinstance(name_raw, str):
                state_name = name_raw
            elif isinstance(name_raw, dict) and isinstance(name_raw.get('name'), str):
                 state_name = name_raw['name'] # Handle potential nested dict { 'name': { 'name': 'Actual Name' } }
            
            # Get other metrics safely
            avg_salary = metric_info.get('avg_salary', 0)
            avg_payment = metric_info.get('avg_payment', 0)
            active_months = metric_info.get('active_months', 0)
            
            avg_salary = avg_salary if isinstance(avg_salary, (int, float)) else 0
            avg_payment = avg_payment if isinstance(avg_payment, (int, float)) else 0
            active_months = active_months if isinstance(active_months, (int, float)) else 0
            
        elif isinstance(metric_info, str):
             # Handle case where the entire metric_info is just the name string
             state_name = metric_info

        metrics_data.append({
            'state': state_name, # Ensure this holds the simple string
            'avg_monthly_salary': avg_salary,
            'avg_monthly_payment': avg_payment,
            'active_months': int(active_months)
        })
    # *** End - Explicit Name Extraction Logic ***

    # Create the table using the processed metrics_data
    salary_table = dash_table.DataTable(
        id='monthly-metrics-table',
        columns=[
            {'name': 'State', 'id': 'state'},
            {'name': 'Avg Monthly Salary', 'id': 'avg_monthly_salary', 'type': 'numeric', 'format': {'specifier': '$,.2f'}},
            {'name': 'Avg Monthly Payment', 'id': 'avg_monthly_payment', 'type': 'numeric', 'format': {'specifier': '$,.2f'}},
            {'name': 'Active Months', 'id': 'active_months', 'type': 'numeric'}
        ],
        data=metrics_data, # data should now contain correct 'state' strings
        style_cell={'textAlign': 'center', 'padding': '10px'},
        style_header={
            'backgroundColor': 'rgb(230, 230, 230)',
            'fontWeight': 'bold'
        },
        style_data_conditional=[
            {
                'if': {'row_index': 'odd'},
                'backgroundColor': 'rgb(248, 248, 248)'
            }
        ]
    )
    
    # Create graphs using the processed metrics_data
    states = [item['state'] for item in metrics_data]
    salaries = [item['avg_monthly_salary'] for item in metrics_data]
    payments = [item['avg_monthly_payment'] for item in metrics_data]
    active_months_list = [item['active_months'] for item in metrics_data]
    
    comparison_fig = go.Figure()
    comparison_fig.add_trace(go.Bar(x=states, y=salaries, name='Average Monthly Salary', marker_color='#2196F3'))
    comparison_fig.add_trace(go.Bar(x=states, y=payments, name='Average Monthly Payment', marker_color='#4CAF50'))
    # ... Add Scatter trace for payment percentage ...
    payment_percentages = [(p / s * 100) if s > 0 else 0 for s, p in zip(salaries, payments)]
    comparison_fig.add_trace(go.Scatter(x=states, y=payment_percentages, name='Payment Percentage', mode='lines+markers', yaxis='y2', line=dict(color='#FFC107')))
    comparison_fig.update_layout(
        title='Monthly Salary and Payment Comparison', xaxis_title='State', yaxis_title='Amount ($)', 
        yaxis2=dict(title='Payment Percentage', overlaying='y', side='right', range=[0, 30], ticksuffix='%'),
        barmode='group', template='plotly_white',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )

    months_fig = go.Figure()
    months_fig.add_trace(go.Bar(x=states, y=active_months_list, name='Active Months', marker_color='#9C27B0'))
    months_fig.update_layout(title='Active Months by State', xaxis_title='State', yaxis_title='Number of Months', template='plotly_white')

    # Return layout including the table and graphs
    return html.Div([
        html.H4("Monthly Metrics Analysis", style={'textAlign': 'center', 'marginBottom': '20px'}),
        html.P("This analysis shows the average monthly salary and payments by state:"),
        html.H5("Monthly Salary and Payment Data", style={'textAlign': 'center', 'marginBottom': '15px', 'marginTop': '20px'}),
        salary_table,
        dcc.Graph(figure=comparison_fig, style={'marginTop': '30px'}),
        dcc.Graph(figure=months_fig, style={'marginTop': '30px'}),
        # final_state_distribution can be added back if needed
    ])

# Callback for the Active Students content
@app.callback(
    Output("active-students-content", "children"),
    [Input("simulation-results-store", "data")]
)
def update_active_students(results):
    if not results:
        return "Run a simulation to see results"
    
    # Extract data (state_metrics should be pre-cleaned)
    state_metrics = results.get('state_metrics', {})
    state_distribution = results.get('state_distribution', {})
    config = results.get('config', {})
    
    # *** REMOVE the get_state_name helper function definition ***
    # def get_state_name(state_idx, metrics_dict): ... (removed)

    # Create data for active students table and visualization
    active_data = []
    
    total_students = config.get('num_students', 100)
    
    for state_idx, metrics in sorted(state_metrics.items(), key=lambda x: int(x[0])):
        # Directly access the cleaned name
        state_name = metrics.get('name', f"State {state_idx}") # Access cleaned name
        current_active = state_distribution.get(str(state_idx), 0) # Ensure string key
            
        # Calculate percentage of total
        percentage = (current_active / total_students) * 100 if total_students > 0 else 0
            
        # Get state-specific metrics safely
        avg_salary = 0
        avg_payment = 0
        if isinstance(metrics, dict):
            avg_salary = metrics.get('avg_salary', 0)
            avg_payment = metrics.get('avg_payment', 0)
            
            # Ensure numeric
            avg_salary = avg_salary if isinstance(avg_salary, (int, float)) else 0
            avg_payment = avg_payment if isinstance(avg_payment, (int, float)) else 0

        active_data.append({
            'state': state_name,  # Use cleaned name
            'active_students': current_active,
            'percentage': percentage,
            'avg_salary': avg_salary,
            'avg_payment': avg_payment
        })
    
    # Filter out states with 0 active students if desired (optional)
    # active_data = [item for item in active_data if item['active_students'] > 0]

    # Create the active students table (No changes needed here)
    # ...
    active_table = dash_table.DataTable(
        id='active-students-table',
        columns=[
            {'name': 'State', 'id': 'state'},
            {'name': 'Active Students', 'id': 'active_students', 'type': 'numeric'},
            {'name': 'Percentage', 'id': 'percentage', 'type': 'numeric', 'format': {'specifier': '.1f'}},
            {'name': 'Avg Salary', 'id': 'avg_salary', 'type': 'numeric', 'format': {'specifier': '$,.2f'}},
            {'name': 'Avg Payment', 'id': 'avg_payment', 'type': 'numeric', 'format': {'specifier': '$,.2f'}}
        ],
        data=active_data,
        style_cell={'textAlign': 'center', 'padding': '10px'},
        style_header={
            'backgroundColor': 'rgb(230, 230, 230)',
            'fontWeight': 'bold'
        },
        style_data_conditional=[
            {
                'if': {'row_index': 'odd'},
                'backgroundColor': 'rgb(248, 248, 248)'
            }
        ]
    )
    
    # Create a pie chart (No changes needed here)
    # ...
    distribution_fig = go.Figure(data=[go.Pie(
        labels=[item['state'] for item in active_data],
        values=[item['active_students'] for item in active_data],
        hole=.3,
        textinfo='label+percent',
        hovertemplate="State: %{label}<br>Students: %{value}<br>Percentage: %{percent}<extra></extra>"
    )])
    
    distribution_fig.update_layout(
        title='Distribution of Active Students by State',
        template='plotly_white'
    )
    
    # Create a bar chart (No changes needed here)
    # ...
    metrics_fig = go.Figure()
    
    metrics_fig.add_trace(go.Bar(
        x=[item['state'] for item in active_data],
        y=[item['active_students'] for item in active_data],
        name='Active Students',
        marker_color='#1976D2',
        yaxis='y',
        hovertemplate="State: %{x}<br>Students: %{y}<extra></extra>"
    ))
    
    metrics_fig.add_trace(go.Scatter(
        x=[item['state'] for item in active_data],
        y=[item['avg_salary'] for item in active_data],
        name='Avg Salary',
        mode='lines+markers',
        yaxis='y2',
        line=dict(color='#4CAF50', width=2),
        marker=dict(size=8),
        hovertemplate="State: %{x}<br>Avg Salary: $%{y:.2f}<extra></extra>"
    ))
    
    metrics_fig.add_trace(go.Scatter(
        x=[item['state'] for item in active_data],
        y=[item['avg_payment'] for item in active_data],
        name='Avg Payment',
        mode='lines+markers',
        yaxis='y2',
        line=dict(color='#FFC107', width=2),
        marker=dict(size=8),
        hovertemplate="State: %{x}<br>Avg Payment: $%{y:.2f}<extra></extra>"
    ))
    
    metrics_fig.update_layout(
        title='Active Students and Financial Metrics by State',
        xaxis_title='State',
        yaxis_title='Number of Students',
        yaxis2=dict(
            title='Amount ($)',
            overlaying='y',
            side='right'
        ),
        template='plotly_white',
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1
        )
    )
    
    # Create summary statistics (Logic updated to sum from active_data)
    active_students_total = sum(item['active_students'] for item in active_data)
    summary_stats = html.Div([
        html.Div([
            html.Div([
                html.H5("Total Active Students", style={'textAlign': 'center', 'marginBottom': '10px'}),
                html.Div(f"{active_students_total}", style={
                    'fontSize': '24px', 
                    'textAlign': 'center',
                    'color': '#1976D2'
                })
            ], style={'width': '30%', 'display': 'inline-block', 'backgroundColor': '#f1f1f1', 'padding': '15px', 'borderRadius': '5px', 'marginRight': '5%'}),
            
            html.Div([
                html.H5("Completion Rate", style={'textAlign': 'center', 'marginBottom': '10px'}),
                html.Div(f"{results.get('completion_rate', 0):.1f}%", style={
                    'fontSize': '24px', 
                    'textAlign': 'center',
                    'color': '#4CAF50'
                })
            ], style={'width': '30%', 'display': 'inline-block', 'backgroundColor': '#f1f1f1', 'padding': '15px', 'borderRadius': '5px', 'marginRight': '5%'}),
            
            html.Div([
                html.H5("Dropout Rate", style={'textAlign': 'center', 'marginBottom': '10px'}),
                html.Div(f"{results.get('dropout_rate', 0):.1f}%", style={
                    'fontSize': '24px', 
                    'textAlign': 'center',
                    'color': '#F44336'
                })
            ], style={'width': '30%', 'display': 'inline-block', 'backgroundColor': '#f1f1f1', 'padding': '15px', 'borderRadius': '5px'})
        ], style={'marginBottom': '20px'})
    ])
    
    # Return combined layout (No changes needed)
    return html.Div([
        html.H4("Active Students Analysis", style={'textAlign': 'center', 'marginBottom': '20px'}),
        summary_stats,
        html.Div([
            html.Div([dcc.Graph(figure=distribution_fig)], style={'width': '50%', 'display': 'inline-block', 'verticalAlign': 'top'}),
            html.Div([dcc.Graph(figure=metrics_fig)], style={'width': '50%', 'display': 'inline-block', 'verticalAlign': 'top'})
        ], style={'marginBottom': '30px'}),
        html.H5("Active Students by State", style={'textAlign': 'center', 'marginBottom': '15px'}),
        html.P([
            "This table shows the current distribution of students across different states, ",
            "including the number of active students, percentage of total, and average financial metrics for each state."
        ], style={'marginBottom': '15px'}),
        active_table
    ])

# Run the app
if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=10000) 