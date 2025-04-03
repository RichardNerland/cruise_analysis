import dash
from dash import dcc, html, Input, Output, State, dash_table
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import time
import random
import argparse

# Import the simplified model
from simple_cruise_model import run_simple_simulation

# Initialize the Dash app
app = dash.Dash(__name__, suppress_callback_exceptions=True)
server = app.server  # Expose the server variable for production

# Enable the app to be embedded in an iframe
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>ISA Analysis Tool</title>
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

# Dropdown options for number of students
student_options = [
    {'label': '10 students', 'value': 10},
    {'label': '50 students', 'value': 50},
    {'label': '100 students', 'value': 100},
    {'label': '200 students', 'value': 200},
    {'label': '500 students', 'value': 500}
]

# Dropdown options for number of simulations
sim_options = [
    {'label': '10 simulations (faster)', 'value': 10},
    {'label': '50 simulations', 'value': 50},
    {'label': '100 simulations (recommended)', 'value': 100}
]

# Define the preset scenarios from the original notebook
preset_scenarios = {
    'uganda_baseline': {
        'name': 'Uganda Baseline',
        'description': 'Balanced mix of BA, MA, and Assistant Track degrees.',
        'program_type': 'Uganda',
        'degrees': {'BA': 0.45, 'MA': 0.24, 'ASST_SHIFT': 0.27, 'NURSE': 0.0, 'NA': 0.04, 'TRADE': 0.0, 'ASST': 0.0}
    },
    'uganda_conservative': {
        'name': 'Uganda Conservative',
        'description': 'More assistant track degrees, fewer advanced degrees.',
        'program_type': 'Uganda',
        'degrees': {'BA': 0.32, 'MA': 0.11, 'ASST_SHIFT': 0.42, 'NURSE': 0.0, 'NA': 0.15, 'TRADE': 0.0, 'ASST': 0.0}
    },
    'uganda_optimistic': {
        'name': 'Uganda Optimistic',
        'description': 'Higher proportion of BA and MA degrees, fewer assistant track degrees.',
        'program_type': 'Uganda',
        'degrees': {'BA': 0.63, 'MA': 0.33, 'ASST_SHIFT': 0.025, 'NURSE': 0.0, 'NA': 0.015, 'TRADE': 0.0, 'ASST': 0.0}
    },
    'kenya_baseline': {
        'name': 'Kenya Baseline',
        'description': 'Nursing baseline scenario with both regular and shifted assistant track.',
        'program_type': 'Kenya',
        'degrees': {'BA': 0.0, 'MA': 0.0, 'ASST': 0.40, 'ASST_SHIFT': 0.20, 'NURSE': 0.25, 'NA': 0.15, 'TRADE': 0.0}
    },
    'kenya_conservative': {
        'name': 'Kenya Conservative',
        'description': 'Higher proportion of assistant track degrees, more dropouts.',
        'program_type': 'Kenya',
        'degrees': {'BA': 0.0, 'MA': 0.0, 'ASST': 0.33, 'ASST_SHIFT': 0.17, 'NURSE': 0.20, 'NA': 0.30, 'TRADE': 0.0}
    },
    'kenya_optimistic': {
        'name': 'Kenya Optimistic',
        'description': 'Higher proportion of nursing degrees, no dropouts.',
        'program_type': 'Kenya',
        'degrees': {'BA': 0.0, 'MA': 0.0, 'ASST': 0.27, 'ASST_SHIFT': 0.13, 'NURSE': 0.60, 'NA': 0.0, 'TRADE': 0.0}
    },
    'rwanda_baseline': {
        'name': 'Rwanda Baseline',
        'description': 'Standard trade program distribution with both regular and shifted assistant track.',
        'program_type': 'Rwanda',
        'degrees': {'BA': 0.0, 'MA': 0.0, 'ASST': 0.27, 'ASST_SHIFT': 0.13, 'NURSE': 0.0, 'NA': 0.20, 'TRADE': 0.40}
    },
    'rwanda_conservative': {
        'name': 'Rwanda Conservative',
        'description': 'Higher dropout rate for trade programs.',
        'program_type': 'Rwanda',
        'degrees': {'BA': 0.0, 'MA': 0.0, 'ASST': 0.27, 'ASST_SHIFT': 0.13, 'NURSE': 0.0, 'NA': 0.40, 'TRADE': 0.2}
    },
    'rwanda_optimistic': {
        'name': 'Rwanda Optimistic',
        'description': 'Very low dropout rate for trade programs.',
        'program_type': 'Rwanda',
        'degrees': {'BA': 0.0, 'MA': 0.0, 'ASST': 0.23, 'ASST_SHIFT': 0.12, 'NURSE': 0.0, 'NA': 0.05, 'TRADE': 0.60}
    }
}

# Define the layout of the app
app.layout = html.Div([
    html.H1("ISA Analysis Tool", style={'textAlign': 'center', 'marginBottom': '30px'}),
    
    # Tabs for different sections
    dcc.Tabs([
        dcc.Tab(label='About', children=[
            html.Div([
                html.H1("ISA Analysis Tool", style={'textAlign': 'center', 'marginBottom': '30px', 'color': '#2c3e50'}),
                
                # Background Section
                html.Div([
                    html.H2("Background", style={'color': '#2c3e50', 'borderBottom': '1px solid #eee', 'paddingBottom': '10px'}),
                    html.P([
                        "Income Share Agreements (ISAs) represent an innovative approach to educational financing where students receive funding for their education in exchange for a percentage of their future income over a defined period. ",
                        "Unlike traditional loans with fixed repayments regardless of outcomes, ISAs align incentives between funders and students—payments scale with a graduate's actual earnings success."
                    ], style={'fontSize': '16px', 'lineHeight': '1.6'}),
                    html.P([
                        html.A("Malengo", href="https://www.malengo.org", target="_blank"), 
                        " is a nonprofit organization that connects talented students from developing countries with educational opportunities abroad through ISA financing. ",
                        "The organization's mission focuses on reducing barriers to migration—a strategy with enormous potential impact, as research suggests the estimated global gains from reducing migration barriers dwarf those related to other policy restrictions by one or two orders of magnitude."
                    ], style={'fontSize': '16px', 'lineHeight': '1.6'}),
                    
                    html.H3("The Malengo Model", style={'color': '#2c3e50', 'marginTop': '20px'}),
                    html.P([
                        "Malengo currently operates a Uganda–Germany Program that prepares academically talented but financially constrained students from Uganda for admission to English-speaking Bachelor's programs at German universities. The organization:"
                    ], style={'fontSize': '16px', 'lineHeight': '1.6'}),
                    html.Ul([
                        html.Li("Provides financial support covering first-year living expenses, semester fees, travel, and application costs", style={'fontSize': '16px', 'lineHeight': '1.6'}),
                        html.Li("Selects students through a competitive process based on academic excellence, limited financial means, and motivation", style={'fontSize': '16px', 'lineHeight': '1.6'}),
                        html.Li("Supports training in both university programs and Germany's highly-regarded vocational training system (Ausbildung)", style={'fontSize': '16px', 'lineHeight': '1.6'}),
                        html.Li("Uses Income Share Agreements to create a sustainable funding model where successful graduates contribute back to support future students", style={'fontSize': '16px', 'lineHeight': '1.6'})
                    ], style={'paddingLeft': '30px'}),
                    
                    html.H3("Economic Impact", style={'color': '#2c3e50', 'marginTop': '20px'}),
                    html.P([
                        "The economic benefits of Malengo's approach are substantial:"
                    ], style={'fontSize': '16px', 'lineHeight': '1.6'}),
                    html.Ul([
                        html.Li("Individual participants can access wages up to 19 times higher than they would earn in Uganda", style={'fontSize': '16px', 'lineHeight': '1.6'}),
                        html.Li("Remittances flow back to families and communities in the home country", style={'fontSize': '16px', 'lineHeight': '1.6'}),
                        html.Li("Research suggests significant spillover effects, with migration contributing to dramatic improvements in GDP growth in developing nations", style={'fontSize': '16px', 'lineHeight': '1.6'}),
                        html.Li("German universities and vocational programs offer tuition-free high-quality education, maximizing the return on investment", style={'fontSize': '16px', 'lineHeight': '1.6'})
                    ], style={'paddingLeft': '30px'}),
                    
                    html.H3("Program Paths", style={'color': '#2c3e50', 'marginTop': '20px'}),
                    html.P([
                        "Malengo supports three primary educational paths in different countries:"
                    ], style={'fontSize': '16px', 'lineHeight': '1.6'}),
                    html.Ul([
                        html.Li([html.Strong("Uganda Program: "), "English-language Bachelor's degrees at German universities, with students typically supporting themselves through part-time work after the first year"], style={'fontSize': '16px', 'lineHeight': '1.6'}),
                        html.Li([html.Strong("Kenya Program: "), "German 'Ausbildung' programs focused on nursing and healthcare fields, combining classroom learning with practical training and providing a modest salary during the training period. Students spend the first year in intensive German language training before traveling to Germany."], style={'fontSize': '16px', 'lineHeight': '1.6'}),
                        html.Li([html.Strong("Rwanda Program: "), "German 'Ausbildung' programs in trade skills like mechatronics, solar installation, and other technical fields. Like the Kenya program, students complete a year of German language training before beginning their vocational training in Germany."], style={'fontSize': '16px', 'lineHeight': '1.6'})
                    ], style={'paddingLeft': '30px'}),
                    
                    html.H3("Language Training", style={'color': '#2c3e50', 'marginTop': '20px'}),
                    html.P([
                        "For the Kenya and Rwanda programs, Malengo invests in a full year of intensive German language training before students travel to Germany. This essential preparation:"
                    ], style={'fontSize': '16px', 'lineHeight': '1.6'}),
                    html.Ul([
                        html.Li("Brings students to the B1/B2 German proficiency level required for vocational training", style={'fontSize': '16px', 'lineHeight': '1.6'}),
                        html.Li("Represents a significant portion of the overall program investment", style={'fontSize': '16px', 'lineHeight': '1.6'}),
                        html.Li("Extends the timeline for investor returns, as payments begin only after students complete their training and find employment", style={'fontSize': '16px', 'lineHeight': '1.6'}),
                        html.Li("Dramatically increases student success rates and long-term employment prospects", style={'fontSize': '16px', 'lineHeight': '1.6'})
                    ], style={'paddingLeft': '30px'}),
                    
                    html.P([
                        "This tool helps model and analyze the financial sustainability of Income Share Agreements across these different program paths, allowing for optimization of support structures while ensuring that both students and funding partners benefit."
                    ], style={'fontSize': '16px', 'lineHeight': '1.6', 'marginTop': '15px'}),
                ], style={'marginBottom': '30px'}),
                
                # Objectives Section
                html.Div([
                    html.H2("Objectives", style={'color': '#2c3e50', 'borderBottom': '1px solid #eee', 'paddingBottom': '10px'}),
                    html.P([
                        "This simulation tool helps stakeholders understand the financial outcomes of ISA programs across various scenarios and student pathways. The tool specifically aims to:"
                    ], style={'fontSize': '16px', 'lineHeight': '1.6'}),
                    html.Ul([
                        html.Li("Model expected returns on educational investments across different program types and student outcomes", style={'fontSize': '16px', 'lineHeight': '1.6'}),
                        html.Li("Simulate how various factors (unemployment, wage penalties, international labor mobility) affect student repayment patterns", style={'fontSize': '16px', 'lineHeight': '1.6'}),
                        html.Li("Provide investors and educational program designers with data-driven insights for program structure and financial planning", style={'fontSize': '16px', 'lineHeight': '1.6'}),
                        html.Li("Create transparency around the financial mechanics of ISAs for all stakeholders", style={'fontSize': '16px', 'lineHeight': '1.6'})
                    ], style={'paddingLeft': '30px'})
                ], style={'marginBottom': '30px'}),
                
                # Methodology Section
                html.Div([
                    html.H2("Methodology", style={'color': '#2c3e50', 'borderBottom': '1px solid #eee', 'paddingBottom': '10px'}),
                    
                    html.H3("Program Simulation", style={'color': '#2c3e50', 'marginTop': '20px'}),
                    html.P([
                        "This tool simulates Income Share Agreement (ISA) outcomes for students in various educational programs, ",
                        "including university degrees, assistant training, and specific trade or nursing tracks. By modeling student earnings, ",
                        "payment thresholds, and potential dropouts or returns to home countries, it provides a comprehensive view of ",
                        "investor returns and student payment patterns under different scenarios."
                    ], style={'fontSize': '16px', 'lineHeight': '1.6'}),
                    
                    html.H3("Economic Modeling", style={'color': '#2c3e50', 'marginTop': '20px'}),
                    html.P([
                        "The simulation incorporates key economic factors including:"
                    ], style={'fontSize': '16px', 'lineHeight': '1.6'}),
                    html.Ul([
                        html.Li("Inflation (defaulted to 2% annually)", style={'fontSize': '16px', 'lineHeight': '1.6'}),
                        html.Li("Unemployment (variable, with 4-8% defaults)", style={'fontSize': '16px', 'lineHeight': '1.6'}),
                        html.Li("Immigrant wage penalties (approximately 20%)", style={'fontSize': '16px', 'lineHeight': '1.6'}),
                        html.Li("Career progression with experience-based salary growth", style={'fontSize': '16px', 'lineHeight': '1.6'}),
                        html.Li("Labor market exit probability (representing return to home countries)", style={'fontSize': '16px', 'lineHeight': '1.6'})
                    ], style={'paddingLeft': '30px'}),
                    
                    html.H3("Scenario Analysis", style={'color': '#2c3e50', 'marginTop': '20px'}),
                    html.P([
                        "The tool offers three scenario types for each educational pathway:"
                    ], style={'fontSize': '16px', 'lineHeight': '1.6'}),
                    html.Ul([
                        html.Li("Baseline: Realistic projections based on current data", style={'fontSize': '16px', 'lineHeight': '1.6'}),
                        html.Li("Conservative: Higher dropout rates, lower advanced degree completion", style={'fontSize': '16px', 'lineHeight': '1.6'}),
                        html.Li("Optimistic: Better degree completion, fewer dropouts, higher employment retention", style={'fontSize': '16px', 'lineHeight': '1.6'})
                    ], style={'paddingLeft': '30px'})
                ], style={'marginBottom': '30px'}),
                
                
                # Model Parameters Section
                html.Div([
                    html.H2("Model Parameters", style={'color': '#2c3e50', 'borderBottom': '1px solid #eee', 'paddingBottom': '10px'}),
                    
                    html.H3("Degree Tracks", style={'color': '#2c3e50', 'marginTop': '20px'}),
                    html.P([
                        "The model uses six distinct tracks, each with mean earnings, variances, and completion times that approximate real-world data:"
                    ], style={'fontSize': '16px', 'lineHeight': '1.6'}),
                    
                    html.Div([
                        html.Div([
                            html.H4("1. Bachelor's Degree (BA)", style={'color': '#2c3e50'}),
                            html.Ul([
                                html.Li("Mean earnings: $41,300/year", style={'fontSize': '16px', 'lineHeight': '1.5'}),
                                html.Li("Standard deviation: $6,000", style={'fontSize': '16px', 'lineHeight': '1.5'}),
                                html.Li("Annual Experience Growth: 3%", style={'fontSize': '16px', 'lineHeight': '1.5'}),
                                html.Li("Years to Complete: 4", style={'fontSize': '16px', 'lineHeight': '1.5'})
                            ], style={'paddingLeft': '20px'})
                        ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top'}),
                        
                        html.Div([
                            html.H4("2. Master's Degree (MA)", style={'color': '#2c3e50'}),
                            html.Ul([
                                html.Li("Mean earnings: $46,709/year", style={'fontSize': '16px', 'lineHeight': '1.5'}),
                                html.Li("Standard deviation: $6,600", style={'fontSize': '16px', 'lineHeight': '1.5'}),
                                html.Li("Annual Experience Growth: 4%", style={'fontSize': '16px', 'lineHeight': '1.5'}),
                                html.Li("Years to Complete: 6", style={'fontSize': '16px', 'lineHeight': '1.5'})
                            ], style={'paddingLeft': '20px'})
                        ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top'})
                    ], style={'marginBottom': '20px'}),
                    
                    html.Div([
                        html.Div([
                            html.H4("3. Assistant Track (ASST)", style={'color': '#2c3e50'}),
                            html.Ul([
                                html.Li("Mean earnings: $31,500/year", style={'fontSize': '16px', 'lineHeight': '1.5'}),
                                html.Li("Standard deviation: $2,800", style={'fontSize': '16px', 'lineHeight': '1.5'}),
                                html.Li("Annual Experience Growth: 0.5%", style={'fontSize': '16px', 'lineHeight': '1.5'}),
                                html.Li("Years to Complete: 3", style={'fontSize': '16px', 'lineHeight': '1.5'}),
                                html.Li("Used in Kenya/Rwanda programs for direct-entry students", style={'fontSize': '16px', 'lineHeight': '1.5'})
                            ], style={'paddingLeft': '20px'})
                        ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top'}),
                        
                        html.Div([
                            html.H4("4. Assistant Shift Track (ASST_SHIFT)", style={'color': '#2c3e50'}),
                            html.Ul([
                                html.Li("Mean earnings: $31,500/year", style={'fontSize': '16px', 'lineHeight': '1.5'}),
                                html.Li("Standard deviation: $2,800", style={'fontSize': '16px', 'lineHeight': '1.5'}),
                                html.Li("Annual Experience Growth: 0.5%", style={'fontSize': '16px', 'lineHeight': '1.5'}),
                                html.Li("Years to Complete: 6", style={'fontSize': '16px', 'lineHeight': '1.5'}),
                                html.Li("Represents students who begin in another program and then shift to assistant training", style={'fontSize': '16px', 'lineHeight': '1.5'})
                            ], style={'paddingLeft': '20px'})
                        ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top'})
                    ], style={'marginBottom': '20px'}),
                    
                    html.Div([
                        html.Div([
                            html.H4("5. Nursing Degree (NURSE)", style={'color': '#2c3e50'}),
                            html.Ul([
                                html.Li("Mean earnings: $40,000/year", style={'fontSize': '16px', 'lineHeight': '1.5'}),
                                html.Li("Standard deviation: $4,000", style={'fontSize': '16px', 'lineHeight': '1.5'}),
                                html.Li("Annual Experience Growth: 2%", style={'fontSize': '16px', 'lineHeight': '1.5'}),
                                html.Li("Years to Complete: 4", style={'fontSize': '16px', 'lineHeight': '1.5'})
                            ], style={'paddingLeft': '20px'})
                        ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top'}),
                        
                        html.Div([
                            html.H4("6. Trade Program (TRADE)", style={'color': '#2c3e50'}),
                            html.Ul([
                                html.Li("Mean earnings: $35,000/year", style={'fontSize': '16px', 'lineHeight': '1.5'}),
                                html.Li("Standard deviation: $3,000", style={'fontSize': '16px', 'lineHeight': '1.5'}),
                                html.Li("Annual Experience Growth: 2%", style={'fontSize': '16px', 'lineHeight': '1.5'}),
                                html.Li("Years to Complete: 3", style={'fontSize': '16px', 'lineHeight': '1.5'})
                            ], style={'paddingLeft': '20px'})
                        ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top'})
                    ], style={'marginBottom': '20px'}),
                    
                    html.Div([
                        html.Div([
                            html.H4("7. No Advancement (NA)", style={'color': '#2c3e50'}),
                            html.Ul([
                                html.Li("Mean earnings: $2,200/year", style={'fontSize': '16px', 'lineHeight': '1.5'}),
                                html.Li("Standard deviation: $640", style={'fontSize': '16px', 'lineHeight': '1.5'}),
                                html.Li("Annual Experience Growth: 1%", style={'fontSize': '16px', 'lineHeight': '1.5'}),
                                html.Li("Years to Complete: 4", style={'fontSize': '16px', 'lineHeight': '1.5'}),
                                html.Li("100% Probability of Returning Home", style={'fontSize': '16px', 'lineHeight': '1.5'})
                            ], style={'paddingLeft': '20px'})
                        ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top'})
                    ], style={'marginBottom': '20px'}),
                    
                    html.H3("Earnings Profiles", style={'color': '#2c3e50', 'marginTop': '20px'}),
                    html.P([
                        "We derive salary levels from public labor data and research on earnings for new graduates in high-income countries. ",
                        "The baseline is then adjusted for:"
                    ], style={'fontSize': '16px', 'lineHeight': '1.6'}),
                    html.Ul([
                        html.Li("Immigrant Wage Penalty (~20%) – Reflects both potential employer bias and the reality that newcomers to a field/country typically earn less early in their careers.", style={'fontSize': '16px', 'lineHeight': '1.6'}),
                        html.Li("Career Stage – The model targets entry-level and early-career professionals with realistic wage increases over time.", style={'fontSize': '16px', 'lineHeight': '1.6'}),
                        html.Li("Program-specific growth trajectories – Different careers have distinct salary progression patterns.", style={'fontSize': '16px', 'lineHeight': '1.6'})
                    ], style={'paddingLeft': '30px'}),
                    
                    html.H3("ISA Terms", style={'color': '#2c3e50', 'marginTop': '20px'}),
                    html.P([
                        "The ISA terms vary by program type and include:"
                    ], style={'fontSize': '16px', 'lineHeight': '1.6'}),
                    html.Ul([
                        html.Li([html.Strong("Payment Thresholds:"), " Students only make payments when their income exceeds $27,000 per year"], style={'fontSize': '16px', 'lineHeight': '1.6'}),
                        html.Li([html.Strong("Income Percentage:"), 
                                 html.Ul([
                                     html.Li("University: 14% of income above threshold", style={'fontSize': '16px', 'lineHeight': '1.5'}),
                                     html.Li("Nursing: 12% of income above threshold", style={'fontSize': '16px', 'lineHeight': '1.5'}),
                                     html.Li("Trade: 12% of income above threshold", style={'fontSize': '16px', 'lineHeight': '1.5'})
                                 ], style={'paddingLeft': '20px'})
                                ], style={'fontSize': '16px', 'lineHeight': '1.6'}),
                        html.Li([html.Strong("Payment Caps:"), 
                                 html.Ul([
                                     html.Li("University: $72,500 total repayment cap", style={'fontSize': '16px', 'lineHeight': '1.5'}),
                                     html.Li("Nursing: $49,950 total repayment cap", style={'fontSize': '16px', 'lineHeight': '1.5'}),
                                     html.Li("Trade: $45,000 total repayment cap", style={'fontSize': '16px', 'lineHeight': '1.5'})
                                 ], style={'paddingLeft': '20px'})
                                ], style={'fontSize': '16px', 'lineHeight': '1.6'}),
                        html.Li([html.Strong("Term Limit:"), " All ISAs have a 10-year maximum repayment period"], style={'fontSize': '16px', 'lineHeight': '1.6'})
                    ], style={'paddingLeft': '30px'})
                ], style={'marginBottom': '30px'}),
                
                # Implementation Section
                html.Div([
                    html.H2("Implementation", style={'color': '#2c3e50', 'borderBottom': '1px solid #eee', 'paddingBottom': '10px'}),
                    html.P([
                        "This interactive dashboard allows users to:"
                    ], style={'fontSize': '16px', 'lineHeight': '1.6'}),
                    html.Ul([
                        html.Li("Select program types (University, Nursing, or Trade) with preset or custom degree distributions", style={'fontSize': '16px', 'lineHeight': '1.6'}),
                        html.Li("Adjust economic parameters like unemployment, inflation, and labor force participation", style={'fontSize': '16px', 'lineHeight': '1.6'}),
                        html.Li("Customize ISA terms including payment percentages, thresholds, and caps", style={'fontSize': '16px', 'lineHeight': '1.6'}),
                        html.Li("Run Monte Carlo simulations to test robustness against parameter variation", style={'fontSize': '16px', 'lineHeight': '1.6'}),
                        html.Li("Save and compare multiple scenarios to identify optimal program structures", style={'fontSize': '16px', 'lineHeight': '1.6'})
                    ], style={'paddingLeft': '30px'}),
                    html.P([
                        "Users navigate through tabs to access simulations, compare results, and analyze detailed outcomes including IRR distributions, repayment patterns, and student success metrics."
                    ], style={'fontSize': '16px', 'lineHeight': '1.6'})
                ], style={'marginBottom': '30px'}),
                
                # Expected Outcomes Section
                html.Div([
                    html.H2("Expected Outcomes", style={'color': '#2c3e50', 'borderBottom': '1px solid #eee', 'paddingBottom': '10px'}),
                    html.P([
                        "The ISA Analysis Tool provides stakeholders with:"
                    ], style={'fontSize': '16px', 'lineHeight': '1.6'}),
                    html.Ul([
                        html.Li("Data-driven understanding of expected investor returns across different program structures", style={'fontSize': '16px', 'lineHeight': '1.6'}),
                        html.Li("Insights into how student outcomes vary by degree type, economic conditions, and program design", style={'fontSize': '16px', 'lineHeight': '1.6'}),
                        html.Li("Ability to test the robustness of ISA models across conservative, baseline, and optimistic scenarios", style={'fontSize': '16px', 'lineHeight': '1.6'}),
                        html.Li("Quantification of risk-return profiles to support investment decisions", style={'fontSize': '16px', 'lineHeight': '1.6'}),
                        html.Li("Transparent assessment of how ISA structures impact student affordability and investor returns", style={'fontSize': '16px', 'lineHeight': '1.6'})
                    ], style={'paddingLeft': '30px'})
                ], style={'marginBottom': '30px'}),
                
                # Future Development Section
                html.Div([
                    html.H2("Future Development", style={'color': '#2c3e50', 'borderBottom': '1px solid #eee', 'paddingBottom': '10px'}),
                    html.P([
                        "We plan to enhance the ISA Analysis Tool with:"
                    ], style={'fontSize': '16px', 'lineHeight': '1.6'}),
                    html.Ul([
                        html.Li("Integration with actual student outcome data as Malengo's programs mature", style={'fontSize': '16px', 'lineHeight': '1.6'}),
                        html.Li("More granular country-specific economic and labor market parameters", style={'fontSize': '16px', 'lineHeight': '1.6'}),
                        html.Li("Dynamic ISA term optimization to balance student affordability and investor returns", style={'fontSize': '16px', 'lineHeight': '1.6'}),
                        html.Li("Additional educational pathway models as Malengo expands program offerings", style={'fontSize': '16px', 'lineHeight': '1.6'}),
                        html.Li("Advanced risk assessment tools for portfolio-level analysis", style={'fontSize': '16px', 'lineHeight': '1.6'})
                    ], style={'paddingLeft': '30px'}),
                    html.P([
                        "These enhancements will make the tool increasingly valuable for educational financing decisions as Malengo grows its impact worldwide."
                    ], style={'fontSize': '16px', 'lineHeight': '1.6'})
                ], style={'marginBottom': '30px'}),
                
                # References Section (preserved from original)
                html.Div([
                    html.H2("Additional Resources", style={'color': '#2c3e50', 'borderBottom': '1px solid #eee', 'paddingBottom': '10px'}),
                    
                    html.H4("Graduation Rate Data Sources", style={'color': '#2c3e50', 'marginTop': '20px'}),
                    html.P([
                        "Our graduation rate assumptions are based on several key German educational research sources:"
                    ], style={'fontSize': '16px', 'lineHeight': '1.6'}),
                    html.Ul([
                        html.Li([
                            html.A("DZHW Brief 05/2022", href="https://www.dzhw.eu/pdf/pub_brief/dzhw_brief_05_2022_anhang.pdf", target="_blank"),
                            " - The table illustrates a baseline dropout rate of 40% for international students; however, Malengo students currently outperform this benchmark with 95% retention rate due to their specialized support system and rigorous selection process."
                        ], style={'fontSize': '16px', 'lineHeight': '1.6'}),
                        html.Li([
                            html.A("BIBB Data Report 2015 (Vocational Training)", href="https://www.bibb.de/datenreport/de/2015/30777.php", target="_blank"),
                            " - The table indicates that 32% of non-German students terminate vocational training prior to completing their exams, compared to 13% of students with previous university experience. Approximately 50% of these early terminations result in transfers to alternative programs rather than complete dropouts, leading to an overall dropout rate of roughly 16%. Malengo currently lacks sufficient data to assess these findings independently."
                        ], style={'fontSize': '16px', 'lineHeight': '1.6'})
                    ], style={'paddingLeft': '30px'}),
                    
                    html.H4("German Profession Names & Salary References", style={'color': '#2c3e50', 'marginTop': '20px'}),
                    html.P([
                        "Below are the German names for the professions mentioned above, along with specific job examples for each degree type:"
                    ], style={'fontSize': '16px', 'lineHeight': '1.6'}),
                    html.Ul([
                        html.Li([
                            html.Strong("Bachelor's Degree (BA) - Bachelorabschluss"), html.Br(),
                            "Example professions: Chemieingenieur/in (Chemical Engineer), Jurist/in (Lawyer), Wirtschaftsingenieur/in (Business Engineer), Informatiker/in (Computer Scientist)"
                        ], style={'fontSize': '16px', 'lineHeight': '1.6'}),
                        html.Li([
                            html.Strong("Master's Degree (MA) - Masterabschluss"), html.Br(),
                            "Example professions: Maschinenbauingenieur/in (Mechanical Engineer), Architekt/in (Architect), Betriebswirt/in (Business Administrator), Physiker/in (Physicist)"
                        ], style={'fontSize': '16px', 'lineHeight': '1.6'}),
                        html.Li([
                            html.Strong("Assistant Track (ASST) - Assistenzausbildung"), html.Br(),
                            "Example professions: Pflegehelfer/in (Nurse Assistant), Altenpflegehelfer/in (Geriatric Nurse Care), Solaranlagenmonteur/in (Solar Installer), Technische/r Assistent/in (Technical Assistant)"
                        ], style={'fontSize': '16px', 'lineHeight': '1.6'}),
                        html.Li([
                            html.Strong("Nursing Degree (NURSE) - Krankenpflegeausbildung"), html.Br(),
                            "Example professions: Krankenschwester/Krankenpfleger (Nurse), Gesundheits- und Krankenpfleger/in (Healthcare and Nursing Professional)"
                        ], style={'fontSize': '16px', 'lineHeight': '1.6'}),
                        html.Li([
                            html.Strong("Trade Program (TRADE) - Handwerksausbildung"), html.Br(),
                            "Example professions: Mechatroniker/in (Mechatronics Engineer), Klempner/in (Plumber), Elektriker/in (Electrician), Schreiner/in (Carpenter)"
                        ], style={'fontSize': '16px', 'lineHeight': '1.6'})
                    ], style={'paddingLeft': '30px'}),
                    
                    html.P([
                        "Salary reference resources:"
                    ], style={'fontSize': '16px', 'lineHeight': '1.6'}),
                    html.Ul([
                        html.Li(html.A("German Government Earnings Atlas (Entgeltatlas)", href="https://web.arbeitsagentur.de/entgeltatlas/beruf/134712", target="_blank"), style={'fontSize': '16px', 'lineHeight': '1.6'}),
                        html.Li(html.A("StepStone Salary Data for Elektroniker", href="https://www.stepstone.de/gehalt/Elektroniker-in.html", target="_blank"), style={'fontSize': '16px', 'lineHeight': '1.6'}),
                        html.Li(html.A("JobVector Salary Information", href="https://www.jobvector.de/gehalt/Elektroniker/", target="_blank"), style={'fontSize': '16px', 'lineHeight': '1.6'}),
                        html.Li(html.A("Gehalt.de Profession Data", href="https://www.gehalt.de/beruf/elektroniker-elektronikerin", target="_blank"), style={'fontSize': '16px', 'lineHeight': '1.6'})
                    ], style={'paddingLeft': '30px'}),
                ], style={'marginBottom': '30px'})
            ], style={'padding': '20px', 'maxWidth': '1200px', 'margin': '0 auto'})
        ]),
        
        dcc.Tab(label='Simulation', children=[
            html.Div([
                html.H1("ISA Analysis Dashboard", style={'textAlign': 'center', 'marginBottom': '30px'}),
                
                html.Div([
                    html.Div([
                        html.H3("Program Parameters", style={'marginBottom': '20px'}),
                        
                        html.Div([
                            html.Label("Preset Scenarios:", style={'fontWeight': 'bold', 'fontSize': '16px'}),
                            html.P("Select a pre-configured scenario to automatically set program type and degree distributions", 
                                 style={'fontSize': '0.85em', 'margin': '2px 0 10px 0'}),
                            dcc.Dropdown(
                                id="preset-scenario",
                                options=[{'label': preset_scenarios[k]['name'], 'value': k} for k in preset_scenarios.keys()],
                                value="uganda_baseline",
                                placeholder="Select a preset scenario",
                                style={'fontWeight': 'bold'}
                            ),
                            html.Div(id="preset-description", style={'color': '#666', 'fontSize': '0.9em', 'marginTop': '5px', 'fontStyle': 'italic'})
                        ], style={'marginBottom': '20px', 'backgroundColor': '#e6f7ff', 'padding': '15px', 'borderRadius': '5px', 'border': '1px solid #b3e0ff', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'}),
                        
                        html.Div([
                            html.Label("Degree Distribution:"),
                            dcc.RadioItems(
                                id="degree-distribution-type",
                                options=[
                                    {'label': 'Custom Distribution', 'value': 'custom'}
                                ],
                                value="custom",
                                labelStyle={'display': 'block', 'marginBottom': '10px'},
                            ),
                        ], style={'marginBottom': '15px'}),
                        
                        # Custom degree distribution section (visible by default now)
                        html.Div(id="custom-degree-section", children=[
                            html.Div([
                                html.Div([
                                    html.Label("Degree Parameters", style={'fontWeight': 'bold', 'fontSize': '16px', 'marginBottom': '10px'})
                                ], style={'textAlign': 'center', 'marginBottom': '15px'}),
                                
                                # Table headers
                                html.Div([
                                    html.Div([html.Label("Degree Type")], style={'width': '20%', 'display': 'inline-block', 'fontWeight': 'bold'}),
                                    html.Div([html.Label("Percentage (%)")], style={'width': '20%', 'display': 'inline-block', 'fontWeight': 'bold'}),
                                    html.Div([html.Label("Avg Salary ($)")], style={'width': '20%', 'display': 'inline-block', 'fontWeight': 'bold'}),
                                    html.Div([html.Label("Std Dev ($)")], style={'width': '20%', 'display': 'inline-block', 'fontWeight': 'bold'}),
                                    html.Div([html.Label("Growth Rate (%)")], style={'width': '20%', 'display': 'inline-block', 'fontWeight': 'bold'}),
                                ], style={'marginBottom': '10px', 'textAlign': 'center'}),
                                
                                # Bachelor's row
                                html.Div([
                                    html.Div([html.Label("Bachelor's (BA)")], style={'width': '20%', 'display': 'inline-block'}),
                                    html.Div([
                                        dcc.Input(
                                            id="ba-pct", 
                                            type="number", 
                                            value=45, 
                                            min=0, 
                                            max=100, 
                                            step=1,
                                            style={'width': '100%'}
                                        )
                                    ], style={'width': '20%', 'display': 'inline-block'}),
                                    html.Div([
                                        dcc.Input(
                                            id="ba-salary", 
                                            type="number", 
                                            value=41300, 
                                            min=0, 
                                            step=100,
                                            style={'width': '100%'}
                                        )
                                    ], style={'width': '20%', 'display': 'inline-block'}),
                                    html.Div([
                                        dcc.Input(
                                            id="ba-std", 
                                            type="number", 
                                            value=6000, 
                                            min=0, 
                                            step=100,
                                            style={'width': '100%'}
                                        )
                                    ], style={'width': '20%', 'display': 'inline-block'}),
                                    html.Div([
                                        dcc.Input(
                                            id="ba-growth", 
                                            type="number", 
                                            value=3.0, 
                                            min=0, 
                                            max=20, 
                                            step=0.1,
                                            style={'width': '100%'}
                                        )
                                    ], style={'width': '20%', 'display': 'inline-block'}),
                                ], style={'marginBottom': '10px'}),
                                
                                # Master's row
                                html.Div([
                                    html.Div([html.Label("Master's (MA)")], style={'width': '20%', 'display': 'inline-block'}),
                                    html.Div([
                                        dcc.Input(
                                            id="ma-pct", 
                                            type="number", 
                                            value=24, 
                                            min=0, 
                                            max=100, 
                                            step=1,
                                            style={'width': '100%'}
                                        )
                                    ], style={'width': '20%', 'display': 'inline-block'}),
                                    html.Div([
                                        dcc.Input(
                                            id="ma-salary", 
                                            type="number", 
                                            value=46709, 
                                            min=0, 
                                            step=100,
                                            style={'width': '100%'}
                                        )
                                    ], style={'width': '20%', 'display': 'inline-block'}),
                                    html.Div([
                                        dcc.Input(
                                            id="ma-std", 
                                            type="number", 
                                            value=6600, 
                                            min=0, 
                                            step=100,
                                            style={'width': '100%'}
                                        )
                                    ], style={'width': '20%', 'display': 'inline-block'}),
                                    html.Div([
                                        dcc.Input(
                                            id="ma-growth", 
                                            type="number", 
                                            value=4.0, 
                                            min=0, 
                                            max=20, 
                                            step=0.1,
                                            style={'width': '100%'}
                                        )
                                    ], style={'width': '20%', 'display': 'inline-block'}),
                                ], style={'marginBottom': '10px'}),
                                
                                # Assistant row
                                html.Div([
                                    html.Div([html.Label("Assistant Track (ASST)")], style={'width': '20%', 'display': 'inline-block'}),
                                    html.Div([
                                        dcc.Input(
                                            id="asst-pct", 
                                            type="number", 
                                            value=0, 
                                            min=0, 
                                            max=100, 
                                            step=1,
                                            style={'width': '100%'}
                                        )
                                    ], style={'width': '20%', 'display': 'inline-block'}),
                                    html.Div([
                                        dcc.Input(
                                            id="asst-salary", 
                                            type="number", 
                                            value=31500, 
                                            min=0, 
                                            step=100,
                                            style={'width': '100%'}
                                        )
                                    ], style={'width': '20%', 'display': 'inline-block'}),
                                    html.Div([
                                        dcc.Input(
                                            id="asst-std", 
                                            type="number", 
                                            value=2800, 
                                            min=0, 
                                            step=100,
                                            style={'width': '100%'}
                                        )
                                    ], style={'width': '20%', 'display': 'inline-block'}),
                                    html.Div([
                                        dcc.Input(
                                            id="asst-growth", 
                                            type="number", 
                                            value=0.5, 
                                            min=0, 
                                            max=20, 
                                            step=0.1,
                                            style={'width': '100%'}
                                        )
                                    ], style={'width': '20%', 'display': 'inline-block'}),
                                ], style={'marginBottom': '10px'}),
                                
                                # Assistant Shift row
                                html.Div([
                                    html.Div([html.Label("Assistant Shift (ASST_SHIFT)")], style={'width': '20%', 'display': 'inline-block'}),
                                    html.Div([
                                        dcc.Input(
                                            id="asst-shift-pct", 
                                            type="number", 
                                            value=27, 
                                            min=0, 
                                            max=100, 
                                            step=1,
                                            style={'width': '100%'}
                                        )
                                    ], style={'width': '20%', 'display': 'inline-block'}),
                                    html.Div([
                                        dcc.Input(
                                            id="asst-shift-salary", 
                                            type="number", 
                                            value=31500, 
                                            min=0, 
                                            step=100,
                                            style={'width': '100%'}
                                        )
                                    ], style={'width': '20%', 'display': 'inline-block'}),
                                    html.Div([
                                        dcc.Input(
                                            id="asst-shift-std", 
                                            type="number", 
                                            value=2800, 
                                            min=0, 
                                            step=100,
                                            style={'width': '100%'}
                                        )
                                    ], style={'width': '20%', 'display': 'inline-block'}),
                                    html.Div([
                                        dcc.Input(
                                            id="asst-shift-growth", 
                                            type="number", 
                                            value=0.5, 
                                            min=0, 
                                            max=20, 
                                            step=0.1,
                                            style={'width': '100%'}
                                        )
                                    ], style={'width': '20%', 'display': 'inline-block'}),
                                ], style={'marginBottom': '10px'}),
                                
                                # Nursing row
                                html.Div([
                                    html.Div([html.Label("Nursing (NURSE)")], style={'width': '20%', 'display': 'inline-block'}),
                                    html.Div([
                                        dcc.Input(
                                            id="nurse-pct", 
                                            type="number", 
                                            value=0, 
                                            min=0, 
                                            max=100, 
                                            step=1,
                                            style={'width': '100%'}
                                        )
                                    ], style={'width': '20%', 'display': 'inline-block'}),
                                    html.Div([
                                        dcc.Input(
                                            id="nurse-salary", 
                                            type="number", 
                                            value=40000, 
                                            min=0, 
                                            step=100,
                                            style={'width': '100%'}
                                        )
                                    ], style={'width': '20%', 'display': 'inline-block'}),
                                    html.Div([
                                        dcc.Input(
                                            id="nurse-std", 
                                            type="number", 
                                            value=4000, 
                                            min=0, 
                                            step=100,
                                            style={'width': '100%'}
                                        )
                                    ], style={'width': '20%', 'display': 'inline-block'}),
                                    html.Div([
                                        dcc.Input(
                                            id="nurse-growth", 
                                            type="number", 
                                            value=2.0, 
                                            min=0, 
                                            max=20, 
                                            step=0.1,
                                            style={'width': '100%'}
                                        )
                                    ], style={'width': '20%', 'display': 'inline-block'}),
                                ], style={'marginBottom': '10px'}),
                                
                                # Trade row
                                html.Div([
                                    html.Div([html.Label("Trade (TRADE)")], style={'width': '20%', 'display': 'inline-block'}),
                                    html.Div([
                                        dcc.Input(
                                            id="trade-pct", 
                                            type="number", 
                                            value=0, 
                                            min=0, 
                                            max=100, 
                                            step=1,
                                            style={'width': '100%'}
                                        )
                                    ], style={'width': '20%', 'display': 'inline-block'}),
                                    html.Div([
                                        dcc.Input(
                                            id="trade-salary", 
                                            type="number", 
                                            value=35000, 
                                            min=0, 
                                            step=100,
                                            style={'width': '100%'}
                                        )
                                    ], style={'width': '20%', 'display': 'inline-block'}),
                                    html.Div([
                                        dcc.Input(
                                            id="trade-std", 
                                            type="number", 
                                            value=5000, 
                                            min=0, 
                                            step=100,
                                            style={'width': '100%'}
                                        )
                                    ], style={'width': '20%', 'display': 'inline-block'}),
                                    html.Div([
                                        dcc.Input(
                                            id="trade-growth", 
                                            type="number", 
                                            value=2, 
                                            min=0, 
                                            max=20, 
                                            step=0.1,
                                            style={'width': '100%'}
                                        )
                                    ], style={'width': '20%', 'display': 'inline-block'}),
                                ], style={'marginBottom': '10px'}),
                                
                                # No Degree row
                                html.Div([
                                    html.Div([html.Label("No Degree (NA)")], style={'width': '20%', 'display': 'inline-block'}),
                                    html.Div([
                                        dcc.Input(
                                            id="na-pct", 
                                            type="number", 
                                            value=4, 
                                            min=0, 
                                            max=100, 
                                            step=1,
                                            style={'width': '100%'}
                                        )
                                    ], style={'width': '20%', 'display': 'inline-block'}),
                                    html.Div([
                                        dcc.Input(
                                            id="na-salary", 
                                            type="number", 
                                            value=2200, 
                                            min=0, 
                                            step=100,
                                            style={'width': '100%'}
                                        )
                                    ], style={'width': '20%', 'display': 'inline-block'}),
                                    html.Div([
                                        dcc.Input(
                                            id="na-std", 
                                            type="number", 
                                            value=640, 
                                            min=0, 
                                            step=100,
                                            style={'width': '100%'}
                                        )
                                    ], style={'width': '20%', 'display': 'inline-block'}),
                                    html.Div([
                                        dcc.Input(
                                            id="na-growth", 
                                            type="number", 
                                            value=1, 
                                            min=0, 
                                            max=20, 
                                            step=0.1,
                                            style={'width': '100%'}
                                        )
                                    ], style={'width': '20%', 'display': 'inline-block'}),
                                ], style={'marginBottom': '10px'}),
                                
            
                                
                                html.Div(id="degree-sum-warning", style={'color': 'red', 'marginTop': '10px', 'textAlign': 'center'})
                            ])
                        ], style={'marginBottom': '20px', 'display': 'block', 'backgroundColor': '#f1f1f1', 'padding': '15px', 'borderRadius': '5px'}),
                        
                        # Add ISA Parameter Controls
                        html.Div([
                            html.Label("ISA Parameters", style={'fontWeight': 'bold', 'fontSize': '16px', 'marginBottom': '10px'}),
                            
                            html.Div([
                                html.Div([html.Label("ISA Percentage (%)")], style={'width': '33%', 'display': 'inline-block'}),
                                html.Div([html.Label("ISA Threshold ($)")], style={'width': '33%', 'display': 'inline-block'}),
                                html.Div([html.Label("ISA Cap ($)")], style={'width': '33%', 'display': 'inline-block'}),
                            ], style={'marginBottom': '5px', 'textAlign': 'center'}),
                            
                            html.Div([
                                html.Div([
                                    dcc.Input(
                                        id="isa-percentage-input", 
                                        type="number", 
                                        value=14,  # Default to Uganda values
                                        min=0, 
                                        max=100, 
                                        step=0.1,
                                        style={'width': '100%'}
                                    )
                                ], style={'width': '33%', 'display': 'inline-block'}),
                                
                                html.Div([
                                    dcc.Input(
                                        id="isa-threshold-input", 
                                        type="number", 
                                        value=27000,
                                        min=0, 
                                        step=1000,
                                        style={'width': '100%'}
                                    )
                                ], style={'width': '33%', 'display': 'inline-block'}),
                                
                                html.Div([
                                    dcc.Input(
                                        id="isa-cap-input", 
                                        type="number", 
                                        value=72500,  # Default to Uganda values
                                        min=0, 
                                        step=1000,
                                        style={'width': '100%'}
                                    )
                                ], style={'width': '33%', 'display': 'inline-block'}),
                            ], style={'marginBottom': '20px'}),
                        ], style={'marginBottom': '20px', 'backgroundColor': '#e6f7ff', 'padding': '15px', 'borderRadius': '5px'}),
                        
                        html.Div([
                            html.Label("Number of Students:"),
                            dcc.Dropdown(
                                id="num-students",
                                options=student_options,
                                value=100
                            ),
                        ], style={'marginBottom': '20px'}),
                        
                        html.Div([
                            html.Label("Number of Simulations:"),
                            dcc.Dropdown(
                                id="num-sims",
                                options=sim_options,
                                value=50
                            ),
                        ], style={'marginBottom': '20px'}),
                        
                        html.Div([
                            html.Label("Leave Labor Force Probability (%):"),
                            dcc.Slider(
                                id="leave-labor-force-prob",
                                min=0,
                                max=50,
                                step=5,
                                value=5,
                                marks={i: f'{i}%' for i in range(0, 51, 10)},
                            ),
                        ], style={'marginBottom': '20px'}),
                        
                        html.Div([
                            html.Label("Initial Unemployment Rate (%):"),
                            dcc.Slider(
                                id="unemployment-rate",
                                min=0,
                                max=20,
                                step=1,
                                value=4,
                                marks={i: f'{i}%' for i in range(0, 21, 5)},
                            ),
                        ], style={'marginBottom': '20px'}),
                        
                        html.Div([
                            html.Label("Initial Inflation Rate (%):"),
                            dcc.Slider(
                                id="inflation-rate",
                                min=0,
                                max=10,
                                step=0.5,
                                value=2,
                                marks={i: f'{i}%' for i in range(0, 11, 2)},
                            ),
                        ], style={'marginBottom': '30px'}),
                        
                        html.Div([
                            html.Button(
                                "Run Simulation", 
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
                    
                    html.Div([
                        html.H3("Results"),
                        
                        html.Div([
                            html.Div(id="summary-stats", style={'marginBottom': '20px'}),
                            
                            dcc.Tabs([
                                dcc.Tab(label='Scenarios', children=[
                                    html.Div(id="scenario-info")
                                ]),
                                dcc.Tab(label='Payment Distribution', children=[
                                    dcc.Graph(id="payment-distribution")
                                ]),
                                dcc.Tab(label='Payment Data Table', children=[
                                    html.Div(id="payment-data-table")
                                ]),
                                dcc.Tab(label='Degree Information', children=[
                                    html.Div(id="degree-info")
                                ]),
                                dcc.Tab(label='IRR Comparison', children=[
                                    dcc.Graph(id="irr-comparison")
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
                                
                                dcc.Tab(label='Blended Scenario Monte Carlo', children=[
                                    html.Div([
                                        html.H4("Blended Scenario Monte Carlo", style={'marginBottom': '15px'}),
                                        html.P("Simulate outcomes by blending multiple scenarios with different weights:"),
                                        
                                        html.Div([
                                            html.Div([
                                                html.Label("Number of Simulations:"),
                                                dcc.Dropdown(
                                                    id="blended-monte-carlo-sims",
                                                    options=[
                                                        {'label': '100 simulations (faster)', 'value': 100},
                                                        {'label': '500 simulations', 'value': 500},
                                                        {'label': '1000 simulations (recommended)', 'value': 1000}
                                                    ],
                                                    value=500
                                                )
                                            ], style={'width': '100%', 'marginBottom': '20px'}),
                                            
                                            html.H5("Scenario 1", style={'marginBottom': '10px'}),
                                            html.Div([
                                                html.Div([
                                                    html.Label("Scenario Type:"),
                                                    dcc.Dropdown(
                                                        id="scenario1-type",
                                                        options=[
                                                            {'label': 'Baseline', 'value': 'baseline'},
                                                            {'label': 'Conservative', 'value': 'conservative'},
                                                            {'label': 'Optimistic', 'value': 'optimistic'},
                                                            {'label': 'Custom', 'value': 'custom'}
                                                        ],
                                                        value='baseline'
                                                    )
                                                ], style={'width': '48%', 'display': 'inline-block'}),
                                                
                                                html.Div([
                                                    html.Label("Weight (%):"),
                                                    dcc.Input(
                                                        id="scenario1-weight",
                                                        type="number",
                                                        min=0,
                                                        max=100,
                                                        value=50,
                                                        style={'width': '100%'}
                                                    )
                                                ], style={'width': '48%', 'display': 'inline-block', 'float': 'right'})
                                            ], style={'marginBottom': '20px'}),
                                            
                                            html.H5("Scenario 2", style={'marginBottom': '10px'}),
                                            html.Div([
                                                html.Div([
                                                    html.Label("Scenario Type:"),
                                                    dcc.Dropdown(
                                                        id="scenario2-type",
                                                        options=[
                                                            {'label': 'Baseline', 'value': 'baseline'},
                                                            {'label': 'Conservative', 'value': 'conservative'},
                                                            {'label': 'Optimistic', 'value': 'optimistic'},
                                                            {'label': 'Custom', 'value': 'custom'}
                                                        ],
                                                        value='conservative'
                                                    )
                                                ], style={'width': '48%', 'display': 'inline-block'}),
                                                
                                                html.Div([
                                                    html.Label("Weight (%):"),
                                                    dcc.Input(
                                                        id="scenario2-weight",
                                                        type="number",
                                                        min=0,
                                                        max=100,
                                                        value=30,
                                                        style={'width': '100%'}
                                                    )
                                                ], style={'width': '48%', 'display': 'inline-block', 'float': 'right'})
                                            ], style={'marginBottom': '20px'}),
                                            
                                            html.H5("Scenario 3", style={'marginBottom': '10px'}),
                                            html.Div([
                                                html.Div([
                                                    html.Label("Scenario Type:"),
                                                    dcc.Dropdown(
                                                        id="scenario3-type",
                                                        options=[
                                                            {'label': 'Baseline', 'value': 'baseline'},
                                                            {'label': 'Conservative', 'value': 'conservative'},
                                                            {'label': 'Optimistic', 'value': 'optimistic'},
                                                            {'label': 'Custom', 'value': 'custom'}
                                                        ],
                                                        value='optimistic'
                                                    )
                                                ], style={'width': '48%', 'display': 'inline-block'}),
                                                
                                                html.Div([
                                                    html.Label("Weight (%):"),
                                                    dcc.Input(
                                                        id="scenario3-weight",
                                                        type="number",
                                                        min=0,
                                                        max=100,
                                                        value=20,
                                                        style={'width': '100%'}
                                                    )
                                                ], style={'width': '48%', 'display': 'inline-block', 'float': 'right'})
                                            ], style={'marginBottom': '20px'}),
                                            
                                            html.Div(id="weight-sum-warning", style={'color': 'red', 'marginBottom': '20px'}),
                                            
                                            html.Div([
                                                html.Label("Leave Labor Force Probability (%):"),
                                                dcc.RangeSlider(
                                                    id="blended-leave-labor-force-range",
                                                    min=0,
                                                    max=20,
                                                    step=2,
                                                    value=[0, 10],
                                                    marks={i: f'{i}%' for i in range(0, 21, 5)},
                                                )
                                            ], style={'marginBottom': '20px'}),
                                            
                                            html.Div([
                                                html.Label("Wage Penalty (%):"),
                                                dcc.RangeSlider(
                                                    id="blended-wage-penalty-range",
                                                    min=-40,
                                                    max=0,
                                                    step=5,
                                                    value=[-30, -10],
                                                    marks={i: f'{i}%' for i in range(-40, 1, 10)},
                                                )
                                            ], style={'marginBottom': '20px'})
                                        ], style={'backgroundColor': '#f1f1f1', 'padding': '15px', 'borderRadius': '5px', 'marginBottom': '20px'}),
                                        
                                        html.Button(
                                            "Run Blended Monte Carlo Simulation", 
                                            id="run-blended-monte-carlo-button", 
                                            n_clicks=0,
                                            style={
                                                'backgroundColor': '#4CAF50',
                                                'color': 'white',
                                                'padding': '10px 20px',
                                                'borderRadius': '5px',
                                                'border': 'none',
                                                'fontSize': '16px',
                                                'cursor': 'pointer',
                                                'width': '100%',
                                                'marginBottom': '20px'
                                            }
                                        ),
                                        
                                        html.Div(id="blended-monte-carlo-loading", style={'color': '#888', 'textAlign': 'center'}),
                                        html.Div(id="blended-monte-carlo-results")
                                    ])
                                ])
                            ], style={'marginTop': '20px'})
                        ])
                    ], style={'width': '65%', 'display': 'inline-block', 'padding': '20px', 'boxShadow': '0 4px 8px 0 rgba(0,0,0,0.2)', 'backgroundColor': '#f9f9f9', 'borderRadius': '8px'})
                ], style={'display': 'flex', 'justifyContent': 'space-between', 'gap': '20px', 'marginBottom': '30px'}),
            ])
        ])
    ]),
    
    dcc.Store(id='simulation-results-store'),
    dcc.Store(id='saved-scenarios-store', data={})
])

# Callback to toggle the custom degree section visibility
@app.callback(
    Output("custom-degree-section", "style"),
    [Input("degree-distribution-type", "value")]
)
def toggle_custom_degree_section(distribution_type):
    if distribution_type == "custom":
        return {'marginBottom': '20px', 'display': 'block', 'backgroundColor': '#f1f1f1', 'padding': '15px', 'borderRadius': '5px'}
    else:
        return {'marginBottom': '20px', 'display': 'none'}

# Callback to validate degree distribution percentages
@app.callback(
    Output("degree-sum-warning", "children"),
    [Input("ba-pct", "value"),
     Input("ma-pct", "value"),
     Input("asst-pct", "value"),
     Input("asst-shift-pct", "value"),
     Input("nurse-pct", "value"),
     Input("na-pct", "value"),
     Input("trade-pct", "value")]
)
def validate_degree_sum(ba_pct, ma_pct, asst_pct, asst_shift_pct, nurse_pct, na_pct, trade_pct):
    total = sum(filter(None, [ba_pct, ma_pct, asst_pct, asst_shift_pct, nurse_pct, na_pct, trade_pct]))
    if total != 100:
        return html.Div(f"Warning: Degree percentages sum to {total}%, not 100%", style={'color': 'red'})
    return ""

# Callback to update sliders when a preset is selected
@app.callback(
    [Output("ba-pct", "value"),
     Output("ma-pct", "value"),
     Output("asst-pct", "value"),
     Output("nurse-pct", "value"),
     Output("na-pct", "value"),
     Output("trade-pct", "value"),
     Output("asst-shift-pct", "value"),
     Output("degree-distribution-type", "value"),
     Output("preset-description", "children")],
    [Input("preset-scenario", "value")]
)
def update_from_preset(preset_name):
    ctx = dash.callback_context
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None
    
    # Use different default presets based on program type
    if preset_name is None or preset_name not in preset_scenarios:
        preset_name = 'uganda_baseline'
    
    # Get the preset
    preset = preset_scenarios[preset_name]
    degrees = preset['degrees']
    
    # Create a more informative description that includes program type
    description = html.Div([
        html.P(f"{preset['description']}", style={'marginBottom': '5px'}),
        html.P(f"Program Type: {preset['program_type']}", style={'fontWeight': 'bold'})
    ])
    
    # Set the sliders to the preset values (multiply by 100 for percentages)
    return (
        degrees.get('BA', 0) * 100, 
        degrees.get('MA', 0) * 100, 
        degrees.get('ASST', 0) * 100,
        degrees.get('NURSE', 0) * 100,
        degrees.get('NA', 0) * 100,
        degrees.get('TRADE', 0) * 100,
        degrees.get('ASST_SHIFT', 0) * 100,
        "custom",  # Switch to custom mode
        description
    )

# Define callback for running the simulation
@app.callback(
    [Output("loading-message", "children"),
     Output("simulation-results-store", "data")],
    [Input("run-simulation", "n_clicks")],
    [State("degree-distribution-type", "value"),
     State("preset-scenario", "value"),
     State("ba-pct", "value"),
     State("ma-pct", "value"),
     State("asst-pct", "value"),
     State("asst-shift-pct", "value"),
     State("nurse-pct", "value"),
     State("na-pct", "value"),
     State("trade-pct", "value"),
     State("ba-salary", "value"),
     State("ba-std", "value"),
     State("ba-growth", "value"),
     State("ma-salary", "value"),
     State("ma-std", "value"),
     State("ma-growth", "value"),
     State("asst-salary", "value"),
     State("asst-std", "value"),
     State("asst-growth", "value"),
     State("asst-shift-salary", "value"),
     State("asst-shift-std", "value"),
     State("asst-shift-growth", "value"),
     State("nurse-salary", "value"),
     State("nurse-std", "value"),
     State("nurse-growth", "value"),
     State("na-salary", "value"),
     State("na-std", "value"),
     State("na-growth", "value"),
     State("trade-salary", "value"),
     State("trade-std", "value"),
     State("trade-growth", "value"),
     State("isa-percentage-input", "value"),
     State("isa-threshold-input", "value"),
     State("isa-cap-input", "value"),
     State("num-students", "value"),
     State("num-sims", "value"),
     State("unemployment-rate", "value"),
     State("inflation-rate", "value"),
     State("leave-labor-force-prob", "value")]
)
def run_simulation(n_clicks, degree_dist_type, preset_scenario, ba_pct, ma_pct, asst_pct, asst_shift_pct, nurse_pct, na_pct, trade_pct,
                   ba_salary, ba_std, ba_growth, ma_salary, ma_std, ma_growth, 
                   asst_salary, asst_std, asst_growth, asst_shift_salary, asst_shift_std, asst_shift_growth,
                   nurse_salary, nurse_std, nurse_growth,
                   na_salary, na_std, na_growth, trade_salary, trade_std, trade_growth,
                   isa_percentage, isa_threshold, isa_cap,
                   num_students, num_sims, unemployment_rate, inflation_rate, 
                   leave_labor_force_prob):
    if n_clicks is None:
        raise PreventUpdate
    
    # Get program type from preset scenario
    program_type = preset_scenarios[preset_scenario]['program_type'] if preset_scenario in preset_scenarios else 'Uganda'
    
    # Convert percentages to decimals
    unemployment_rate = unemployment_rate / 100.0
    inflation_rate = inflation_rate / 100.0
    leave_labor_force_prob = leave_labor_force_prob / 100.0
    isa_percentage = isa_percentage / 100.0
    
    # Determine scenario based on degree distribution type
    scenario = 'custom'  # We now always use custom since we removed the default option
    
    # Convert percentages to decimals
    ba_pct_decimal = ba_pct / 100.0
    ma_pct_decimal = ma_pct / 100.0
    asst_pct_decimal = asst_pct / 100.0
    asst_shift_pct_decimal = asst_shift_pct / 100.0 if asst_shift_pct is not None else 0
    nurse_pct_decimal = nurse_pct / 100.0
    na_pct_decimal = na_pct / 100.0
    trade_pct_decimal = trade_pct / 100.0
    
    # Run the simulation
    try:
        results = run_simple_simulation(
            program_type=program_type,
            num_students=num_students,
            num_sims=num_sims,
            ba_salary=ba_salary,
            ba_std=ba_std,
            ba_growth=ba_growth/ 100.0,
            ma_salary=ma_salary,
            ma_std=ma_std,
            ma_growth=ma_growth/ 100.0,
            asst_salary=asst_salary,
            asst_std=asst_std,
            asst_growth=asst_growth/ 100.0,
            nurse_salary=nurse_salary,
            nurse_std=nurse_std,
            nurse_growth=nurse_growth/ 100.0,
            na_salary=na_salary,
            na_std=na_std,
            na_growth=na_growth/ 100.0,
            trade_salary=trade_salary,
            trade_std=trade_std,
            trade_growth=trade_growth/ 100.0,
            isa_percentage=isa_percentage,
            isa_threshold=isa_threshold,
            isa_cap=isa_cap,
            initial_unemployment_rate=unemployment_rate,
            initial_inflation_rate=inflation_rate,
            leave_labor_force_probability=leave_labor_force_prob,
            ba_pct=ba_pct_decimal,
            ma_pct=ma_pct_decimal,
            asst_pct=asst_pct_decimal,
            nurse_pct=nurse_pct_decimal,
            na_pct=na_pct_decimal,
            trade_pct=trade_pct_decimal,
            asst_shift_pct=(asst_shift_pct or 0) / 100.0,
            scenario='custom',
            new_malengo_fee=True,
            apply_graduation_delay=True  # Enable the graduation delay feature
        )
        
        # Convert DataFrames to JSON for storage
        serializable_results = {
            'program_type': results['program_type'],
            'total_investment': results['total_investment'],
            'price_per_student': results['price_per_student'],
            'isa_percentage': results['isa_percentage'],
            'isa_threshold': results['isa_threshold'],
            'isa_cap': results['isa_cap'],
            'IRR': results['IRR'],
            'investor_IRR': results['investor_IRR'],
            'nominal_IRR': results.get('nominal_IRR', results['IRR']),  # Add nominal IRR
            'nominal_investor_IRR': results.get('nominal_investor_IRR', results['investor_IRR']),  # Add nominal investor IRR
            'average_total_payment': results['average_total_payment'],
            'average_investor_payment': results['average_investor_payment'],
            'average_malengo_payment': results['average_malengo_payment'],
            'average_nominal_total_payment': results.get('average_nominal_total_payment', results['average_total_payment']),  # Add nominal payment
            'average_duration': results['average_duration'],
            'payment_by_year': results['payment_by_year'].to_dict(),
            'investor_payment_by_year': results['investor_payment_by_year'].to_dict(),
            'malengo_payment_by_year': results['malengo_payment_by_year'].to_dict(),
            'payment_quantiles': results['payment_quantiles'],
            'investor_payment_quantiles': results.get('investor_payment_quantiles', {}),
            'employment_rate': results['employment_rate'],
            'ever_employed_rate': results.get('ever_employed_rate', 0),
            'repayment_rate': results['repayment_rate'],
            'cap_stats': results['cap_stats'],
            'leave_labor_force_probability': results['leave_labor_force_probability'],
            'custom_degrees': results['custom_degrees'],
            'degree_counts': results['degree_counts'],
            'degree_pcts': results['degree_pcts'],
            'initial_unemployment_rate': unemployment_rate,
            'initial_inflation_rate': inflation_rate,
            'apply_graduation_delay': True  # Store this information for the payment table
        }
        
        return "Simulation completed!", serializable_results
    
    except Exception as e:
        return f"Error in simulation: {str(e)}", None

# Update summary stats to include degree distribution
@app.callback(
    Output("summary-stats", "children"),
    [Input("simulation-results-store", "data")]
)
def update_summary_stats(results):
    if not results:
        return "Run a simulation to see results"
    
    # Instead of showing redundant information, just display a message directing users to the tabs
    return html.Div([
        html.H4("Simulation Complete", style={'color': '#4CAF50'}),
        html.P("Your simulation has completed successfully. Please explore the tabs below to view detailed results and analysis."),
        html.P("Each tab contains different visualizations and metrics to help you understand the simulation outcomes.")
    ], style={'padding': '15px', 'backgroundColor': '#f9f9f9', 'borderRadius': '5px', 'marginBottom': '20px'})

# Callback for payment distribution graph
@app.callback(
    Output("payment-distribution", "figure"),
    [Input("simulation-results-store", "data")]
)
def update_payment_distribution(results):
    if not results:
        return go.Figure()
    
    # Convert stored dict back to pandas Series
    payment_by_year = pd.Series(results['payment_by_year'])
    
    # Create the payment distribution graph
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=payment_by_year.index,
        y=payment_by_year.values,
        name="Average Payment by Year",
        marker_color='rgb(55, 83, 109)',
        hovertemplate="Year: %{x}<br>Average Payment: $%{y:,.0f}<extra></extra>"
    ))
    
    fig.update_layout(
        title=f"{results['program_type']} Program - Average Payments by Year",
        xaxis_title="Year",
        yaxis_title="Payment Amount ($)",
        template="plotly_white",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

# Callback for IRR distribution
@app.callback(
    Output("irr-distribution", "figure"),
    [Input("simulation-results-store", "data")]
)
def update_irr_distribution(results):
    if not results:
        return go.Figure()
    
    # Create the IRR distribution graph
    fig = go.Figure()
    
    # Extract quantile values - with fallbacks for missing data
    quantiles = {}
    
    # Prioritize nominal investor payment quantiles
    if 'nominal_investor_payment_quantiles' in results:
        quantiles = results['nominal_investor_payment_quantiles']
    elif 'nominal_payment_quantiles' in results:
        quantiles = results['nominal_payment_quantiles']
    elif 'investor_payment_quantiles' in results:
        quantiles = results['investor_payment_quantiles']
    elif 'payment_quantiles' in results:
        quantiles = results['payment_quantiles']
    else:
        # Create default quantiles if none are available
        quantiles = {
            '0': 0,
            '0.25': 0.02,
            '0.5': 0.05,
            '0.75': 0.08,
            '1.0': 0.12
        }
    
    # Create data for quantile plot
    quantile_labels = ['Minimum', '25th Percentile', 'Median', '75th Percentile', 'Maximum']
    quantile_values = [
        float(quantiles.get('0', 0)), 
        float(quantiles.get('0.25', 0.02)), 
        float(quantiles.get('0.5', 0.05)), 
        float(quantiles.get('0.75', 0.08)), 
        float(quantiles.get('1.0', 0.12))
    ]
    
    # Add a bar chart for quantiles
    fig.add_trace(go.Bar(
        x=quantile_labels,
        y=quantile_values,
        marker_color='rgb(158,202,225)',
        name='Investor IRR Quantiles'
    ))
    
    # Add a horizontal line for the overall IRR (less prominent)
    # Always prioritize nominal_investor_IRR
    irr_value = results.get('nominal_investor_IRR', results.get('investor_IRR', 0.06))  # Use nominal_investor_IRR as primary source
    fig.add_trace(go.Scatter(
        x=quantile_labels,
        y=[irr_value] * len(quantile_labels),
        mode='lines',
        name='Overall Investor IRR',
        line=dict(color='rgb(55, 83, 109)', dash='dash', width=1.5)
    ))
    
    fig.update_layout(
        title=f"{results['program_type']} Program - Nominal Investor IRR Distribution",
        xaxis_title="Percentile",
        yaxis_title="IRR",
        yaxis_tickformat='.2%',
        template="plotly_white",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

# Callback for student outcome statistics
@app.callback(
    Output("student-outcome-stats", "children"),
    [Input("simulation-results-store", "data")]
)
def update_student_outcome_stats(results):
    if not results:
        return html.Div("Run a simulation to see student outcome statistics")
    
    # Get cap stats with fallback for missing keys
    cap_stats = results.get('cap_stats', {})
    
    # If cap_stats only has percentages but not counts, calculate the counts
    num_students = 0
    for degree, count in results.get('degree_counts', {}).items():
        num_students += count
    
    if num_students == 0:
        num_students = 100  # Default value
    
    # Initialize with zeros to ensure we have values
    if 'payment_cap_count' not in cap_stats and 'payment_cap_pct' in cap_stats:
        cap_stats['payment_cap_count'] = cap_stats['payment_cap_pct'] * num_students
    
    if 'years_cap_count' not in cap_stats and 'years_cap_pct' in cap_stats:
        cap_stats['years_cap_count'] = cap_stats['years_cap_pct'] * num_students
    
    if 'no_cap_count' not in cap_stats and 'no_cap_pct' in cap_stats:
        cap_stats['no_cap_count'] = cap_stats['no_cap_pct'] * num_students
    
    # Ensure all required keys exist with default values
    for key in ['payment_cap_count', 'years_cap_count', 'no_cap_count',
                'payment_cap_pct', 'years_cap_pct', 'no_cap_pct',
                'avg_repayment_cap_hit', 'avg_repayment_years_hit', 'avg_repayment_no_cap']:
        if key not in cap_stats:
            cap_stats[key] = 0
    
    # Calculate total students in each category
    total_students = cap_stats['payment_cap_count'] + cap_stats['years_cap_count'] + cap_stats['no_cap_count']
    
    # Create a table for cap statistics
    cap_table = pd.DataFrame({
        'Category': ['Hit Payment Cap', 'Hit 10-Years Cap', 'Hit No Cap'],
        'Count': [
            f"{cap_stats['payment_cap_count']:.1f}",
            f"{cap_stats['years_cap_count']:.1f}",
            f"{cap_stats['no_cap_count']:.1f}"
        ],
        'Percentage': [
            f"{cap_stats['payment_cap_pct']*100:.1f}%",
            f"{cap_stats['years_cap_pct']*100:.1f}%", 
            f"{cap_stats['no_cap_pct']*100:.1f}%"
        ],
        'Avg Repayment': [
            f"${cap_stats['avg_repayment_cap_hit']:,.0f}",
            f"${cap_stats['avg_repayment_years_hit']:,.0f}",
            f"${cap_stats['avg_repayment_no_cap']:,.0f}"
        ]
    })
    
    # Convert to Dash table
    table = dash_table.DataTable(
        data=cap_table.to_dict('records'),
        columns=[{'name': col, 'id': col} for col in cap_table.columns],
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
    
    # Add summary statistics
    summary_stats = html.Div([
        html.H4("Student Payment Categories", style={'marginBottom': '15px', 'textAlign': 'center'}),
        html.Div([
            html.P(f"Annual Employment Rate: {results['employment_rate']*100:.1f}%", style={'fontWeight': 'bold'}),
            html.P(f"Ever Employed Rate: {results.get('ever_employed_rate', 0)*100:.1f}%", style={'fontWeight': 'bold'}),
            html.P(f"Students Making Payments: {results['repayment_rate']*100:.1f}%", style={'fontWeight': 'bold'}),
        ], style={'textAlign': 'center', 'marginBottom': '15px'}),
        table,
        html.Div([
            html.P(f"Average Cap Value When Hit: ${cap_stats.get('avg_cap_value', results.get('isa_cap', 0)):,.0f}", 
                 style={'marginTop': '15px', 'color': '#2a6496', 'fontWeight': 'bold', 'textAlign': 'center'})
        ])
    ])
    
    return summary_stats

# New callback for repayment caps chart
@app.callback(
    Output("repayment-caps-chart", "figure"),
    [Input("simulation-results-store", "data")]
)
def update_repayment_caps_chart(results):
    if not results:
        return go.Figure()
    
    # Get cap stats with fallback for missing keys
    cap_stats = results.get('cap_stats', {})
    
    # Ensure all required keys exist with default values
    for key in ['payment_cap_count', 'years_cap_count', 'no_cap_count',
                'avg_repayment_cap_hit', 'avg_repayment_years_hit', 'avg_repayment_no_cap']:
        if key not in cap_stats:
            cap_stats[key] = 0
    
    # Create data for the figure
    categories = ['Hit Payment Cap', 'Hit 10-Years Cap', 'Hit No Cap']
    counts = [cap_stats['payment_cap_count'], cap_stats['years_cap_count'], cap_stats['no_cap_count']]
    avg_repayments = [cap_stats['avg_repayment_cap_hit'], cap_stats['avg_repayment_years_hit'], cap_stats['avg_repayment_no_cap']]
    
    # Create combined student count and average repayment chart
    fig = go.Figure()
    
    # Add student count bars
    fig.add_trace(go.Bar(
        x=categories,
        y=counts,
        name='Student Count',
        marker_color='rgb(55, 83, 109)',
        yaxis='y'
    ))

    # Add average repayment line
    fig.add_trace(go.Scatter(
        x=categories,
        y=avg_repayments,
        name='Avg Repayment ($)',
        marker_color='rgb(26, 118, 255)',
        mode='lines+markers',
        marker=dict(size=12),
        line=dict(width=4),
        yaxis='y2'
    ))

    # Add cost per student line
    fig.add_trace(go.Scatter(
        x=categories,
        y=[results['price_per_student']] * 3,
        name='Cost Per Student',
        marker_color='rgb(255, 99, 71)',
        mode='lines',
        line=dict(width=2, dash='dash'),
        yaxis='y2'
    ))
    
    # Update layout with secondary y-axis
    fig.update_layout(
        title='Student Outcomes by Repayment Category',
        xaxis_title='Repayment Category',
        yaxis=dict(
            title='Number of Students',
            side='left'
        ),
        yaxis2=dict(
            title='Amount ($)',
            side='right',
            overlaying='y',
            showgrid=False
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        template='plotly_white'
    )
    
    return fig

# Callback for detailed results
@app.callback(
    Output("detailed-results", "children"),
    [Input("simulation-results-store", "data")]
)
def update_detailed_results(results):
    if not results:
        return "Run a simulation to see results"
    
    # Get cap stats with fallback for missing keys
    cap_stats = results.get('cap_stats', {})
    
    # Ensure all required keys exist with default values
    for key in ['payment_cap_pct', 'years_cap_pct', 'no_cap_pct',
                'avg_repayment_cap_hit', 'avg_repayment_years_hit', 'avg_repayment_no_cap']:
        if key not in cap_stats:
            cap_stats[key] = 0
    
    # Create a list of detailed statistics
    content_elements = [
        html.Div([
            html.H4("Repayment Statistics"),
            html.Table([
                html.Tr([html.Th("Metric"), html.Th("Value")]),
                html.Tr([
                    html.Td("Students Hitting Payment Cap"),
                    html.Td(f"{cap_stats['payment_cap_pct']*100:.1f}%")
                ]),
                html.Tr([
                    html.Td("Students Hitting Years Cap"),
                    html.Td(f"{cap_stats['years_cap_pct']*100:.1f}%")
                ]),
                html.Tr([
                    html.Td("Students Not Hitting Cap"),
                    html.Td(f"{cap_stats['no_cap_pct']*100:.1f}%")
                ])
            ], style={'width': '100%', 'marginBottom': '20px'})
        ]),
        
        html.Div([
            html.H4("Average Repayment by Cap Type"),
            html.Table([
                html.Tr([html.Th("Category"), html.Th("Average Repayment")]),
                html.Tr([
                    html.Td("Payment Cap Hit"),
                    html.Td(f"${cap_stats['avg_repayment_cap_hit']:,.0f}")
                ]),
                html.Tr([
                    html.Td("Years Cap Hit"),
                    html.Td(f"${cap_stats['avg_repayment_years_hit']:,.0f}")
                ]),
                html.Tr([
                    html.Td("No Cap Hit"),
                    html.Td(f"${cap_stats['avg_repayment_no_cap']:,.0f}")
                ]),
                html.Tr([
                    html.Td("Average Cap Value When Hit"),
                    html.Td(f"${cap_stats.get('avg_cap_value', results.get('isa_cap', 0)):,.0f}")
                ], style={'color': '#2a6496'})
            ], style={'width': '100%', 'marginBottom': '20px'})
        ]),
        
        html.Div([
            html.H4("Degree Distribution"),
            html.Table([
                html.Tr([html.Th("Degree Type"), html.Th("Percentage"), html.Th("Count")]),
                *[html.Tr([
                    html.Td(degree),
                    html.Td(f"{pct*100:.1f}%"),
                    html.Td(f"{results['degree_counts'].get(degree, 0):.1f}" if degree in results.get('degree_counts', {}) else "-"),
                ]) for degree, pct in results['degree_pcts'].items() if pct > 0]
            ], style={'width': '100%', 'marginBottom': '20px'})
        ])
    ]
    
    return html.Div(content_elements)

# Add a callback for the scenario info tab
@app.callback(
    Output("scenario-info", "children"),
    [Input("simulation-results-store", "data")]
)
def update_scenario_info(results):
    if results is None:
        return html.Div()
    
    scenario_info = [
        html.H4("Scenario Parameters"),
        html.Table([
            html.Thead(html.Tr([
                html.Th("Parameter"),
                html.Th("Value")
            ])),
            html.Tbody([
                html.Tr([html.Td("Program Type:"), html.Td(results['program_type'])]),
                html.Tr([html.Td("ISA Percentage:"), html.Td(f"{results['isa_percentage']*100:.1f}%")]),
                html.Tr([html.Td("ISA Threshold:"), html.Td(f"${results['isa_threshold']:.2f}")]),
                html.Tr([html.Td("ISA Cap:"), html.Td(f"${results['isa_cap']:.2f}")]),
                html.Tr([html.Td("Initial Unemployment:"), html.Td(f"{results['initial_unemployment_rate']*100:.1f}%")]),
                html.Tr([html.Td("Initial Inflation:"), html.Td(f"{results['initial_inflation_rate']*100:.1f}%")]),
                html.Tr([html.Td("Leave Labor Force Probability:"), html.Td(f"{results['leave_labor_force_probability']*100:.1f}%")]),
            ])
        ], className="table table-striped table-sm")
    ]
    
    return html.Div(scenario_info)

# Callback for degree info - update to show custom degree distribution if used
@app.callback(
    Output("degree-info", "children"),
    [Input("simulation-results-store", "data")]
)
def update_degree_info(results):
    if not results:
        return "Run a simulation to see results"
    
    # Get degree percentages
    degree_pcts = results.get('degree_pcts', {})
    
    # Create a list of degree information
    content_elements = [
        html.Div([
            html.H4("Degree Distribution"),
            html.Table([
                html.Tr([html.Th("Degree Type"), html.Th("Percentage"), html.Th("Count")]),
                *[html.Tr([
                    html.Td(degree),
                    html.Td(f"{pct*100:.1f}%"),
                    html.Td(f"{results['degree_counts'].get(degree, 0):.1f}" if degree in results.get('degree_counts', {}) else "-"),
                ]) for degree, pct in degree_pcts.items() if pct > 0]
            ], style={'width': '100%', 'marginBottom': '20px'})
        ])
    ]
    
    return html.Div(content_elements)

# Add callback to update ISA parameters based on preset scenario
@app.callback(
    [Output("isa-percentage-input", "value"),
     Output("isa-threshold-input", "value"),
     Output("isa-cap-input", "value")],
    [Input("preset-scenario", "value")]
)
def update_isa_params(preset_scenario):
    # Get program type from preset scenario
    if preset_scenario in preset_scenarios:
        program_type = preset_scenarios[preset_scenario]['program_type']
    else:
        program_type = 'Uganda'  # Default
    
    if program_type == 'Uganda':
        return 14, 27000, 72500
    elif program_type == 'Kenya':
        return 12, 27000, 49950
    elif program_type == 'Rwanda':
        return 12, 27000, 45000  # Changed from 10% to 12%
    else:
        return 12, 27000, 50000  # Default values

# Add callbacks for scenario comparison functionality
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
            irr_value = saved_scenarios[name].get('nominal_investor_IRR', 0) * 100
            scenario_items.append(html.Div([
                html.Button(
                    f"{name} - IRR: {irr_value:.1f}%",
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
    
    # 1. IRR Comparison Chart
    irr_data = []
    for scenario in selected_scenarios:
        irr_data.append({
            'Scenario': scenario['name'],
            'Investor IRR': scenario['data'].get('nominal_investor_IRR', 0) * 100,
            'Total IRR': scenario['data'].get('nominal_IRR', 0) * 100
        })
    
    irr_df = pd.DataFrame(irr_data)
    
    irr_fig = go.Figure()
    irr_fig.add_trace(go.Bar(
        x=irr_df['Scenario'],
        y=irr_df['Investor IRR'],
        name='Investor IRR',
        marker_color='rgb(26, 118, 255)'
    ))
    
    irr_fig.add_trace(go.Bar(
        x=irr_df['Scenario'],
        y=irr_df['Total IRR'],
        name='Total IRR (before fees)',
        marker_color='rgb(55, 83, 109)',
        opacity=0.7
    ))
    
    irr_fig.update_layout(
        title='IRR Comparison by Scenario',
        xaxis_title='Scenario',
        yaxis_title='IRR (%)',
        barmode='group',
        template='plotly_white'
    )
    
    comparison_elements.append(dcc.Graph(figure=irr_fig))
    
    # 2. Key Metrics Comparison
    metrics_data = []
    for scenario in selected_scenarios:
        metrics_data.append({
            'Scenario': scenario['name'],
            'Investor IRR': f"{scenario['data'].get('nominal_investor_IRR', 0)*100:.1f}%",
            'Avg Payment': f"${scenario['data'].get('average_nominal_total_payment', 0):,.0f}",
            'Repayment Rate': f"{scenario['data'].get('repayment_rate', 0)*100:.1f}%",
            'Employment Rate': f"{scenario['data'].get('employment_rate', 0)*100:.1f}%",
            'Ever Employed': f"{scenario['data'].get('ever_employed_rate', 0)*100:.1f}%",
            'Duration': f"{scenario['data'].get('average_duration', 0):.1f} years"
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
    
    # 3. Student Outcomes Comparison
    outcomes_data = []
    for scenario in selected_scenarios:
        outcomes_data.append({
            'Scenario': scenario['name'],
            'Employment Rate': f"{scenario['data'].get('employment_rate', 0)*100:.1f}%",
            'Ever Employed': f"{scenario['data'].get('ever_employed_rate', 0)*100:.1f}%",
            'Repayment Rate': f"{scenario['data'].get('repayment_rate', 0)*100:.1f}%",
            'Payment Cap %': f"{scenario['data'].get('cap_stats', {}).get('payment_cap_pct', 0)*100:.1f}%",
            'Years Cap %': f"{scenario['data'].get('cap_stats', {}).get('years_cap_pct', 0)*100:.1f}%",
            'No Cap %': f"{scenario['data'].get('cap_stats', {}).get('no_cap_pct', 0)*100:.1f}%"
        })
    
    outcomes_table = dash_table.DataTable(
        data=outcomes_data,
        columns=[{"name": col, "id": col} for col in outcomes_data[0].keys()],
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
        html.H5("Student Outcomes by Scenario", style={'marginTop': '30px', 'marginBottom': '15px'}),
        outcomes_table
    ]))
    
    # 4. Degree Distribution Comparison
    degree_data = []
    for scenario in selected_scenarios:
        degree_pcts = scenario['data'].get('degree_pcts', {})
        degree_data.append({
            'Scenario': scenario['name'],
            'BA': f"{degree_pcts.get('BA', 0)*100:.1f}%",
            'MA': f"{degree_pcts.get('MA', 0)*100:.1f}%",
            'ASST': f"{degree_pcts.get('ASST', 0)*100:.1f}%",
            'NURSE': f"{degree_pcts.get('NURSE', 0)*100:.1f}%",
            'NA': f"{degree_pcts.get('NA', 0)*100:.1f}%",
            'TRADE': f"{degree_pcts.get('TRADE', 0)*100:.1f}%"
        })
    
    degree_table = dash_table.DataTable(
        data=degree_data,
        columns=[{"name": col, "id": col} for col in degree_data[0].keys()],
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
        html.H5("Degree Distribution by Scenario", style={'marginTop': '30px', 'marginBottom': '15px'}),
        degree_table
    ]))
    
    return html.Div(comparison_elements)


# Monte Carlo simulation tab was removed while keeping the Blended Monte Carlo simulation tab

    
    return "", html.Div(viz_elements)

@app.callback(
    Output("weight-sum-warning", "children"),
    [Input("scenario1-weight", "value"),
     Input("scenario2-weight", "value"),
     Input("scenario3-weight", "value")]
)
def validate_weight_sum(weight1, weight2, weight3):
    if weight1 is None or weight2 is None or weight3 is None:
        return "Please enter weights for all scenarios."
    
    total = weight1 + weight2 + weight3
    
    if total != 100:
        return f"Warning: Weights sum to {total}%, not 100%. Please adjust the weights."
    
    return ""

@app.callback(
    [Output("blended-monte-carlo-loading", "children"),
     Output("blended-monte-carlo-results", "children")],
    [Input("run-blended-monte-carlo-button", "n_clicks")],
    [State("blended-monte-carlo-sims", "value"),
     State("scenario1-type", "value"),
     State("scenario2-type", "value"),
     State("scenario3-type", "value"),
     State("scenario1-weight", "value"),
     State("scenario2-weight", "value"),
     State("scenario3-weight", "value"),
     State("blended-leave-labor-force-range", "value"),
     State("blended-wage-penalty-range", "value"),
     State("preset-scenario", "value"),
     State("ba-pct", "value"),
     State("ma-pct", "value"),
     State("asst-pct", "value"),
     State("nurse-pct", "value"),
     State("na-pct", "value"),
     State("trade-pct", "value"),
     State("ba-salary", "value"),
     State("ma-salary", "value"),
     State("asst-salary", "value"),
     State("nurse-salary", "value"),
     State("na-salary", "value"),
     State("trade-salary", "value"),
     State("ba-growth", "value"),  # Add growth parameters
     State("ma-growth", "value"),
     State("asst-growth", "value"),
     State("nurse-growth", "value"),
     State("na-growth", "value"),
     State("trade-growth", "value"),
     State("isa-percentage-input", "value"),
     State("isa-threshold-input", "value"),
     State("isa-cap-input", "value"),
     State("num-students", "value"),
     State("inflation-rate", "value")]
)
def run_blended_monte_carlo(n_clicks, num_sims, 
                           scenario1_type, scenario2_type, scenario3_type,
                           scenario1_weight, scenario2_weight, scenario3_weight,
                           leave_labor_force_range, wage_penalty_range,
                           preset_scenario, ba_pct, ma_pct, asst_pct, nurse_pct, na_pct, trade_pct,
                           ba_salary, ma_salary, asst_salary, nurse_salary, na_salary, trade_salary,
                           ba_growth, ma_growth, asst_growth, nurse_growth, na_growth, trade_growth,
                           isa_percentage, isa_threshold, isa_cap,
                           num_students, inflation_rate):
    if n_clicks == 0:
        return "", html.Div("Click 'Run Blended Monte Carlo Simulation' to see results")
    
    # Get program type from preset scenario
    if preset_scenario in preset_scenarios:
        program_type = preset_scenarios[preset_scenario]['program_type']
    else:
        program_type = 'Uganda'  # Default
        
    # Start with a loading message
    loading_message = "Running Blended Monte Carlo simulation... This may take a moment."
    
    # Validate weights
    if scenario1_weight is None or scenario2_weight is None or scenario3_weight is None:
        return "", html.Div("Please enter weights for all scenarios.", style={'color': 'red'})
    
    # Set default values for None parameters
    if isa_percentage is None:
        if program_type == 'Uganda':
            isa_percentage = 14
        elif program_type == 'Kenya':
            isa_percentage = 12
        elif program_type == 'Rwanda':
            isa_percentage = 12  # Changed from 10 to 12
        else:
            isa_percentage = 12
    
    if isa_threshold is None:
        isa_threshold = 27000
    
    if isa_cap is None:
        if program_type == 'Uganda':
            isa_cap = 72500
        elif program_type == 'Kenya':
            isa_cap = 49950
        elif program_type == 'Rwanda':
            isa_cap = 45000
        else:
            isa_cap = 50000
    
    if inflation_rate is None:
        inflation_rate = 2
    
    # Set default values for degree percentages if None
    if ba_pct is None: ba_pct = 0
    if ma_pct is None: ma_pct = 0
    if asst_pct is None: asst_pct = 0
    if nurse_pct is None: nurse_pct = 0
    if na_pct is None: na_pct = 0
    if trade_pct is None: trade_pct = 0
    
    # Set default values for salary parameters if None
    if ba_salary is None: ba_salary = 41300
    if ma_salary is None: ma_salary = 46709
    if asst_salary is None: asst_salary = 31500
    if nurse_salary is None: nurse_salary = 40000
    if na_salary is None: na_salary = 2200
    if trade_salary is None: trade_salary = 35000
    
    # Set default values for growth parameters if None
    if ba_growth is None: ba_growth = 3  # 3%
    if ma_growth is None: ma_growth = 4  # 4%
    if asst_growth is None: asst_growth = 0.5  # 0.5%
    if nurse_growth is None: nurse_growth = 2  # 2%
    if na_growth is None: na_growth = 1  # 1%
    if trade_growth is None: trade_growth = 2  # 2%
    
    # Normalize weights to sum to 1
    total_weight = scenario1_weight + scenario2_weight + scenario3_weight
    if total_weight != 100:
        return "", html.Div(f"Weights must sum to 100%. Current sum: {total_weight}%", style={'color': 'red'})
    
    # Normalize weights to probabilities
    weights = [scenario1_weight / 100.0, scenario2_weight / 100.0, scenario3_weight / 100.0]
    scenarios = [scenario1_type, scenario2_type, scenario3_type]
    
    # Prepare base parameters for custom scenario
    base_params = {
        'program_type': program_type,
        'num_students': num_students or 100,  # Default to 100 if None
        'num_sims': 5,  # Use minimal simulations per run for speed
        'ba_pct': (ba_pct or 0) / 100.0,
        'ma_pct': (ma_pct or 0) / 100.0,
        'asst_pct': (asst_pct or 0) / 100.0,
        'nurse_pct': (nurse_pct or 0) / 100.0,
        'na_pct': (na_pct or 0) / 100.0,
        'trade_pct': (trade_pct or 0) / 100.0,
        'ba_salary': ba_salary or 41300,
        'ma_salary': ma_salary or 46709,
        'asst_salary': asst_salary or 31500,
        'nurse_salary': nurse_salary or 40000,
        'na_salary': na_salary or 2200,
        'trade_salary': trade_salary or 35000,
        'ba_growth': (ba_growth or 0) / 100.0,
        'ma_growth': (ma_growth or 0) / 100.0,
        'asst_growth': (asst_growth or 0) / 100.0,
        'nurse_growth': (nurse_growth or 0) / 100.0,
        'na_growth': (na_growth or 0) / 100.0,
        'trade_growth': (trade_growth or 0) / 100.0,
        'isa_percentage': (isa_percentage or 12) / 100.0,
        'isa_threshold': isa_threshold or 27000,
        'isa_cap': isa_cap or 50000,
        'initial_inflation_rate': (inflation_rate or 2) / 100.0
    }
    
    # Unpack ranges
    min_leave_labor_force, max_leave_labor_force = leave_labor_force_range
    min_wage_penalty, max_wage_penalty = wage_penalty_range
    
    # Run Monte Carlo simulations
    results = []
    np.random.seed(42)
    
    for i in range(num_sims):
        # Randomly select a scenario based on weights
        selected_scenario = np.random.choice(scenarios, p=weights)
        
        # Generate random parameters for this simulation
        sim_params = base_params.copy()
        sim_params['scenario'] = selected_scenario
        
        # Apply leave labor force probability
        sim_params['leave_labor_force_probability'] = np.random.uniform(min_leave_labor_force, max_leave_labor_force) / 100.0
        
        # Apply wage penalty to all salary parameters
        # Shift the wage penalty by 20% (e.g., -20% becomes 0%, -40% becomes -20%)
        raw_wage_penalty = np.random.uniform(min_wage_penalty, max_wage_penalty) / 100.0
        adjusted_wage_penalty = raw_wage_penalty + 0.2  # Shift by 20%
        penalty_factor = 1 + adjusted_wage_penalty
        
        sim_params['ba_salary'] = ba_salary * penalty_factor
        sim_params['ma_salary'] = ma_salary * penalty_factor
        sim_params['asst_salary'] = asst_salary * penalty_factor
        sim_params['nurse_salary'] = nurse_salary * penalty_factor
        sim_params['na_salary'] = na_salary * penalty_factor
        sim_params['trade_salary'] = trade_salary * penalty_factor
        
        # Add a random seed for each simulation
        sim_params['random_seed'] = np.random.randint(1, 10000)
        
        try:
            # Run the simulation with the varied parameters
            sim_result = run_simple_simulation(**sim_params)
            
            # Extract key metrics
            results.append({
                'investor_irr': sim_result.get('nominal_investor_IRR', 0) * 100,  # Convert to percentage
                'total_irr': sim_result.get('nominal_IRR', 0) * 100,
                'avg_payment': sim_result.get('average_nominal_total_payment', 0),
                'duration': sim_result.get('average_duration', 0),
                'repayment_rate': sim_result.get('repayment_rate', 0) * 100,
                'employment_rate': sim_result.get('employment_rate', 0) * 100,
                'ever_employed_rate': sim_result.get('ever_employed_rate', 0) * 100,
                'leave_labor_force': sim_params.get('leave_labor_force_probability', 0) * 100,
                'wage_penalty': raw_wage_penalty * 100,
                'adjusted_wage_penalty': adjusted_wage_penalty * 100,
                'scenario': selected_scenario
            })
        except Exception as e:
            # If simulation fails, skip this iteration
            continue
    
    if not results:
        return "", html.Div("No valid simulation results. Try different parameters.")
    
    # Convert results to DataFrame for analysis
    results_df = pd.DataFrame(results)
    
    # Create tabs for different visualizations
    tabs = []
    
    # Scenario labels for display
    scenario_labels = {
        'baseline': 'Baseline',
        'conservative': 'Conservative',
        'optimistic': 'Optimistic',
        'custom': 'Custom'
    }
    
    # Tab 1: Simulation Parameters
    param_summary = html.Div([
        html.H4("Blended Monte Carlo Parameters", className="mt-4"),
        html.Table([
            html.Thead(html.Tr([
                html.Th("Parameter"),
                html.Th("Value")
            ])),
            html.Tbody([
                html.Tr([html.Td("Number of Simulations"), html.Td(f"{len(results_df)}")]),
                html.Tr([html.Td("Program Type"), html.Td(f"{program_type}")]),
                html.Tr([html.Td("Scenario 1"), html.Td(f"{scenario_labels.get(scenario1_type, scenario1_type)} ({scenario1_weight}%)")]),
                html.Tr([html.Td("Scenario 2"), html.Td(f"{scenario_labels.get(scenario2_type, scenario2_type)} ({scenario2_weight}%)")]),
                html.Tr([html.Td("Scenario 3"), html.Td(f"{scenario_labels.get(scenario3_type, scenario3_type)} ({scenario3_weight}%)")]),
                html.Tr([html.Td("Leave Labor Force Range"), html.Td(f"{min_leave_labor_force}% to {max_leave_labor_force}%")]),
                html.Tr([html.Td("Wage Penalty Range"), html.Td(f"{min_wage_penalty}% to {max_wage_penalty}%")]),
                html.Tr([html.Td("Wage Penalty Adjustment"), html.Td("+20% (applied to all wage penalties)")])
            ])
        ], className="table table-striped table-sm"),
        
        # Add scenario distribution pie chart
        html.H4("Scenario Distribution", className="mt-4"),
        dcc.Graph(figure=go.Figure(data=[go.Pie(
            labels=[scenario_labels.get(s, s) for s in results_df['scenario'].value_counts().index],
            values=results_df['scenario'].value_counts().values,
            hole=.3,
            marker_colors=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        )]).update_layout(
            title='Distribution of Selected Scenarios',
            template='plotly_white'
        ))
    ])
    
    tabs.append(dcc.Tab(label="Simulation Parameters", children=[param_summary]))
    
    # Tab 2: IRR Distribution
    # Create IRR histogram
    irr_fig = go.Figure()
    irr_fig.add_trace(go.Histogram(
        x=results_df['investor_irr'],
        name='Investor IRR',
        marker_color='rgb(26, 118, 255)',
        opacity=0.75,
        nbinsx=30
    ))
    
    # Add vertical lines for key statistics
    median_irr = results_df['investor_irr'].median()
    mean_irr = results_df['investor_irr'].mean()
    p10_irr = results_df['investor_irr'].quantile(0.1)
    p90_irr = results_df['investor_irr'].quantile(0.9)
    
    irr_fig.add_vline(x=median_irr, line_dash="dash", line_color="black", annotation_text=f"Median: {median_irr:.1f}%")
    irr_fig.add_vline(x=mean_irr, line_dash="solid", line_color="red", annotation_text=f"Mean: {mean_irr:.1f}%")
    irr_fig.add_vline(x=p10_irr, line_dash="dot", line_color="orange", annotation_text=f"P10: {p10_irr:.1f}%")
    irr_fig.add_vline(x=p90_irr, line_dash="dot", line_color="green", annotation_text=f"P90: {p90_irr:.1f}%")
    
    irr_fig.update_layout(
        title='Blended Monte Carlo Distribution of Nominal Investor IRR',
        xaxis_title='Investor IRR (%)',
        yaxis_title='Frequency',
        template='plotly_white',
        bargap=0.1
    )
    
    # IRR statistics table
    irr_stats = html.Div([
        html.H4("Nominal IRR Distribution Statistics", className="mt-4"),
        html.Table([
            html.Thead(html.Tr([
                html.Th("Statistic"),
                html.Th("Value")
            ])),
            html.Tbody([
                html.Tr([html.Td("Mean IRR"), html.Td(f"{results_df['investor_irr'].mean():.2f}%")]),
                html.Tr([html.Td("Median IRR"), html.Td(f"{results_df['investor_irr'].median():.2f}%")]),
                html.Tr([html.Td("Standard Deviation"), html.Td(f"{results_df['investor_irr'].std():.2f}%")]),
                html.Tr([html.Td("10th Percentile"), html.Td(f"{results_df['investor_irr'].quantile(0.1):.2f}%")]),
                html.Tr([html.Td("25th Percentile"), html.Td(f"{results_df['investor_irr'].quantile(0.25):.2f}%")]),
                html.Tr([html.Td("75th Percentile"), html.Td(f"{results_df['investor_irr'].quantile(0.75):.2f}%")]),
                html.Tr([html.Td("90th Percentile"), html.Td(f"{results_df['investor_irr'].quantile(0.9):.2f}%")]),
                html.Tr([html.Td("Minimum IRR"), html.Td(f"{results_df['investor_irr'].min():.2f}%")]),
                html.Tr([html.Td("Maximum IRR"), html.Td(f"{results_df['investor_irr'].max():.2f}%")]),
                html.Tr([html.Td("Probability of IRR > 5%"), html.Td(f"{(results_df['investor_irr'] > 5).mean()*100:.1f}%")]),
                html.Tr([html.Td("Probability of IRR > 8%"), html.Td(f"{(results_df['investor_irr'] > 8).mean()*100:.1f}%")]),
                html.Tr([html.Td("Probability of IRR < 0%"), html.Td(f"{(results_df['investor_irr'] < 0).mean()*100:.1f}%")])
            ])
        ], className="table table-striped table-sm")
    ])
    
    irr_tab_content = html.Div([
        dcc.Graph(figure=irr_fig),
        irr_stats
    ])
    
# #     tabs.append(dcc.Tab(label="Nominal IRR Distribution", children=[irr_tab_content]))
    
    # Tab 3: Scenario Comparison
    # IRR by Scenario Box Plot
    scenario_box = go.Figure()
    
    for scenario in results_df['scenario'].unique():
        scenario_data = results_df[results_df['scenario'] == scenario]
        scenario_box.add_trace(go.Box(
            y=scenario_data['investor_irr'],
            name=scenario_labels.get(scenario, scenario),
            boxpoints='all',
            jitter=0.3,
            pointpos=-1.8
        ))
    
    scenario_box.update_layout(
        title='IRR Distribution by Scenario',
        yaxis_title='Investor IRR (%)',
        template='plotly_white'
    )
    
    # Scenario statistics table
    scenario_stats = html.Div([
        html.H4("Scenario Comparison", className="mt-4"),
        html.Table([
            html.Thead(html.Tr([
                html.Th("Scenario"),
                html.Th("Count"),
                html.Th("Mean IRR"),
                html.Th("Median IRR"),
                html.Th("Min IRR"),
                html.Th("Max IRR")
            ])),
            html.Tbody([
                html.Tr([
                    html.Td(scenario_labels.get(scenario, scenario)),
                    html.Td(f"{len(results_df[results_df['scenario'] == scenario])}"),
                    html.Td(f"{results_df[results_df['scenario'] == scenario]['investor_irr'].mean():.2f}%"),
                    html.Td(f"{results_df[results_df['scenario'] == scenario]['investor_irr'].median():.2f}%"),
                    html.Td(f"{results_df[results_df['scenario'] == scenario]['investor_irr'].min():.2f}%"),
                    html.Td(f"{results_df[results_df['scenario'] == scenario]['investor_irr'].max():.2f}%")
                ]) for scenario in results_df['scenario'].unique()
            ])
        ], className="table table-striped table-sm")
    ])
    
    scenario_tab_content = html.Div([
        dcc.Graph(figure=scenario_box),
        scenario_stats
    ])
    
    tabs.append(dcc.Tab(label="Scenario Comparison", children=[scenario_tab_content]))
    
    # Tab 4: Parameter Relationships
    # Scatter Plot of Leave Labor Force vs IRR colored by Scenario
    llf_scatter_fig = go.Figure()
    
    for scenario in results_df['scenario'].unique():
        scenario_data = results_df[results_df['scenario'] == scenario]
        llf_scatter_fig.add_trace(go.Scatter(
            x=scenario_data['leave_labor_force'],
            y=scenario_data['investor_irr'],
            mode='markers',
            name=scenario_labels.get(scenario, scenario),
            text=[f"IRR: {irr:.1f}%<br>Leave Labor Force: {llf:.1f}%<br>Scenario: {scenario_labels.get(s, s)}" 
                  for irr, llf, s in zip(
                      scenario_data['investor_irr'], 
                      scenario_data['leave_labor_force'],
                      scenario_data['scenario']
                  )],
            hoverinfo='text'
        ))
    
    llf_scatter_fig.update_layout(
        title='Relationship Between Leave Labor Force Probability and Investor IRR by Scenario',
        xaxis_title='Leave Labor Force Probability (%)',
        yaxis_title='Investor IRR (%)',
        template='plotly_white'
    )
    
    # Scatter Plot of Wage Penalty vs IRR colored by Scenario
    wp_scatter_fig = go.Figure()
    
    for scenario in results_df['scenario'].unique():
        scenario_data = results_df[results_df['scenario'] == scenario]
        wp_scatter_fig.add_trace(go.Scatter(
            x=scenario_data['wage_penalty'],
            y=scenario_data['investor_irr'],
            mode='markers',
            name=scenario_labels.get(scenario, scenario),
            text=[f"IRR: {irr:.1f}%<br>Wage Penalty: {wp:.1f}%<br>Scenario: {scenario_labels.get(s, s)}" 
                  for irr, wp, s in zip(
                      scenario_data['investor_irr'], 
                      scenario_data['wage_penalty'],
                      scenario_data['scenario']
                  )],
            hoverinfo='text'
        ))
    
    wp_scatter_fig.update_layout(
        title='Relationship Between Wage Penalty and Investor IRR by Scenario',
        xaxis_title='Wage Penalty (%)',
        yaxis_title='Investor IRR (%)',
        template='plotly_white'
    )
    
    parameter_tab_content = html.Div([
        dcc.Graph(figure=llf_scatter_fig),
        dcc.Graph(figure=wp_scatter_fig)
    ])
    
    tabs.append(dcc.Tab(label="Parameter Relationships", children=[parameter_tab_content]))
    
    # Tab 5: Student Outcomes
    # Create student outcomes visualization
    outcomes_fig = go.Figure()
    
    # Add employment rate vs IRR scatter plot
    outcomes_fig.add_trace(go.Scatter(
        x=results_df['employment_rate'],
        y=results_df['investor_irr'],
        mode='markers',
        marker=dict(
            size=8,
            color=results_df['scenario'].map({s: i for i, s in enumerate(results_df['scenario'].unique())}),
            colorscale='Viridis',
            showscale=False,
            opacity=0.7
        ),
        text=[f"IRR: {irr:.1f}%<br>Employment Rate: {emp:.1f}%<br>Scenario: {scenario_labels.get(s, s)}" 
              for irr, emp, s in zip(
                  results_df['investor_irr'], 
                  results_df['employment_rate'],
                  results_df['scenario']
              )],
        hoverinfo='text',
        name='Employment Rate vs IRR'
    ))
    
    outcomes_fig.update_layout(
        title='Relationship Between Employment Rate and Investor IRR',
        xaxis_title='Employment Rate (%)',
        yaxis_title='Investor IRR (%)',
        template='plotly_white'
    )
    
    # Create repayment rate vs IRR scatter plot
    repayment_fig = go.Figure()
    
    repayment_fig.add_trace(go.Scatter(
        x=results_df['repayment_rate'],
        y=results_df['investor_irr'],
        mode='markers',
        marker=dict(
            size=8,
            color=results_df['scenario'].map({s: i for i, s in enumerate(results_df['scenario'].unique())}),
            colorscale='Viridis',
            showscale=False,
            opacity=0.7
        ),
        text=[f"IRR: {irr:.1f}%<br>Repayment Rate: {rep:.1f}%<br>Scenario: {scenario_labels.get(s, s)}" 
              for irr, rep, s in zip(
                  results_df['investor_irr'], 
                  results_df['repayment_rate'],
                  results_df['scenario']
              )],
        hoverinfo='text',
        name='Repayment Rate vs IRR'
    ))
    
    repayment_fig.update_layout(
        title='Relationship Between Repayment Rate and Investor IRR',
        xaxis_title='Repayment Rate (%)',
        yaxis_title='Investor IRR (%)',
        template='plotly_white'
    )
    
    # Create student outcomes summary table
    outcomes_stats = html.Div([
        html.H4("Student Outcomes by Scenario", className="mt-4"),
        html.Table([
            html.Thead(html.Tr([
                html.Th("Scenario"),
                html.Th("Employment Rate"),
                html.Th("Ever Employed Rate"),
                html.Th("Repayment Rate")
            ])),
            html.Tbody([
                html.Tr([
                    html.Td(scenario_labels.get(scenario, scenario)),
                    html.Td(f"{results_df[results_df['scenario'] == scenario]['employment_rate'].mean():.1f}%"),
                    html.Td(f"{results_df[results_df['scenario'] == scenario]['ever_employed_rate'].mean():.1f}%"),
                    html.Td(f"{results_df[results_df['scenario'] == scenario]['repayment_rate'].mean():.1f}%")
                ]) for scenario in results_df['scenario'].unique()
            ])
        ], className="table table-striped table-sm")
    ])
    
    outcomes_tab_content = html.Div([
        dcc.Graph(figure=outcomes_fig),
        dcc.Graph(figure=repayment_fig),
        outcomes_stats
    ])
    
    tabs.append(dcc.Tab(label="Student Outcomes", children=[outcomes_tab_content]))
    
    # Combine all tabs
    viz_elements = html.Div([
        dcc.Tabs(tabs)
    ])
    
    return "", viz_elements

# Callback for payment data table
@app.callback(
    Output("payment-data-table", "children"),
    [Input("simulation-results-store", "data")]
)
def update_payment_data_table(results):
    if not results:
        return html.Div("Run a simulation to see results")
    
    # Convert stored dict back to pandas Series
    payment_by_year = pd.Series(results['payment_by_year'])
    malengo_payment_by_year = pd.Series(results['malengo_payment_by_year'])
    
    # Define the number of years in the simulation right at the start
    num_years = len(payment_by_year)
    
    # Get number of students and degree completion times
    num_students = 0
    for degree in results.get('degree_counts', {}):
        num_students += results['degree_counts'][degree]
    
    if num_students == 0:
        num_students = 100  # Default if no data available
        print(f"WARNING: No students found in degree_counts, defaulting to {num_students} students")
        
        # If we need to create default degree distribution, create it here
        if not results.get('degree_counts'):
            results['degree_counts'] = {'BA': 100}
            results['degree_pcts'] = {'BA': 1.0}
    
    # Get degree distribution and completion times
    degree_pcts = results.get('degree_pcts', {})
    degree_counts = results.get('degree_counts', {})
    
    # Define degree nominal completion times
    base_degree_times = {
        'BA': 4,
        'MA': 6,
        'ASST': 3,
        'NURSE': 3,
        'TRADE': 3,
        'NA': 4
    }
    
    # Adjust for program type - add 1 year for language training in Kenya/Rwanda
    if results['program_type'] in ['Kenya', 'Rwanda']:
        for degree in base_degree_times:
            base_degree_times[degree] += 1
    
    # Check if graduation delay is applied
    apply_graduation_delay = results.get('apply_graduation_delay', False)
    
    # Model graduation over time with the degree-specific delay distribution
    def calculate_graduation_distribution(degree_count, base_years, degree_type=''):
        graduation_counts = {}
        
        # Special distribution for Masters, Nurse, and Trade degrees
        if degree_type in ['MA', 'NURSE', 'TRADE']:
            # 75% on time, 20% +1 year, 2.5% +2 years, 2.5% +3 years
            for year in range(base_years, base_years + 4):
                if year == base_years:
                    # 75% graduate on time
                    graduation_counts[year] = degree_count * 0.75
                elif year == base_years + 1:
                    # 20% graduate 1 year late
                    graduation_counts[year] = degree_count * 0.20
                elif year == base_years + 2:
                    # 2.5% graduate 2 years late
                    graduation_counts[year] = degree_count * 0.025
                elif year == base_years + 3:
                    # 2.5% graduate 3 years late
                    graduation_counts[year] = degree_count * 0.025
        else:
            # Default distribution for BA, ASST, NA degrees
            # 50% on time, 25% +1 year, 12.5% +2 years, 6.25% +3 years, 6.25% +4 years
            for year in range(base_years, base_years + 5):
                if year == base_years:
                    # 50% graduate on time
                    graduation_counts[year] = degree_count * 0.5
                elif year == base_years + 1:
                    # 25% graduate 1 year late
                    graduation_counts[year] = degree_count * 0.25
                elif year == base_years + 2:
                    # 12.5% graduate 2 years late
                    graduation_counts[year] = degree_count * 0.125
                elif year == base_years + 3:
                    # 6.25% graduate 3 years late
                    graduation_counts[year] = degree_count * 0.0625
                elif year == base_years + 4:
                    # 6.25% graduate 4 years late
                    graduation_counts[year] = degree_count * 0.0625
                    
        return graduation_counts
    
    # Track graduations by year for each degree type
    degree_graduations = {}
    
    # Find the minimum graduation time across all degrees to determine when first graduations occur
    min_grad_time = min([base_years for degree, base_years in base_degree_times.items() 
                        if degree in degree_counts and degree_counts[degree] > 0])
    
    if apply_graduation_delay:
        # Use the delay distribution
        for degree, count in degree_counts.items():
            if degree in base_degree_times and count > 0:
                # Calculate graduation distribution for this degree
                degree_graduations[degree] = calculate_graduation_distribution(
                    count, base_degree_times[degree], degree
                )
    else:
        # Use simple graduation at exact base times
        for degree, count in degree_counts.items():
            if degree in base_degree_times and count > 0:
                degree_graduations[degree] = {base_degree_times[degree]: count}
    
    # Calculate total graduations per year
    graduations_by_year = {}
    
    # Initialize graduations_by_year with 0 for all years
    for year in range(num_years):
        graduations_by_year[year] = 0
    
    # Fill in actual graduations from degree_graduations
    for degree, grad_schedule in degree_graduations.items():
        for year, count in grad_schedule.items():
            if year < num_years:  # Make sure we don't exceed the simulation years
                graduations_by_year[year] += count
    
    # Initialize tracking arrays with correct length
    in_school = [0] * num_years
    graduated = [0] * num_years
    graduations_this_year = [0] * num_years
    repaying = [0] * num_years
    not_repaying_graduated = [0] * num_years
    
    # Current number of students in school (starts with all students)
    students_in_school = num_students
    cumulative_graduated = 0
    
    # First year all students are in school
    in_school[0] = num_students
    
    # Get employment and repayment rates
    employment_rate = results['employment_rate']
    repayment_rate = results['repayment_rate']
    
    # Get cap stats
    cap_stats = results.get('cap_stats', {})
    
    # If cap_stats only has percentages but not counts, calculate the counts from percentages
    if 'payment_cap_pct' in cap_stats and 'payment_cap_count' not in cap_stats:
        cap_stats['payment_cap_count'] = cap_stats['payment_cap_pct'] * num_students
        cap_stats['years_cap_count'] = cap_stats['years_cap_pct'] * num_students
        cap_stats['no_cap_count'] = cap_stats['no_cap_pct'] * num_students
    
    # If cap_stats is missing repayment values, add default values
    if 'avg_repayment_cap_hit' not in cap_stats:
        avg_payment = results.get('average_total_payment', 0)
        cap_stats['avg_repayment_cap_hit'] = results.get('isa_cap', avg_payment * 1.5)
        cap_stats['avg_repayment_years_hit'] = avg_payment * 0.8
        cap_stats['avg_repayment_no_cap'] = avg_payment * 0.5
    
    
    # Process graduation and payments for each year after year 0
    for year in range(1, num_years):
        # Get graduations for this year
        year_graduations = graduations_by_year.get(year, 0)
        
        # Ensure graduations never exceed students in school
        year_graduations = min(year_graduations, students_in_school)
        
        # Update students in school and cumulative graduations
        students_in_school -= year_graduations
        cumulative_graduated += year_graduations
        
        # Ensure bounds are maintained
        students_in_school = max(0, students_in_school)
        cumulative_graduated = min(cumulative_graduated, num_students)
        
        # Calculate students who are repaying (based on repayment rate)
        # Students start repaying in the same year they graduate
        students_repaying = cumulative_graduated * repayment_rate
        
        # Calculate students who have graduated but are not repaying
        students_not_repaying = cumulative_graduated - students_repaying
        
        # Store current year data in tracking arrays
        in_school[year] = int(students_in_school)
        graduated[year] = int(cumulative_graduated)
        graduations_this_year[year] = int(year_graduations)
        repaying[year] = int(students_repaying)
        not_repaying_graduated[year] = int(students_not_repaying)
    
    # Estimate average earnings
    # Based on the average payment amount and the ISA percentage
    isa_percentage = results['isa_percentage']
    isa_threshold = results['isa_threshold']
    
    # Calculate average earnings for different groups
    avg_earnings_repaying = []
    avg_repayment_if_repaying = []
    
    # Process earnings data
    for year in range(num_years):
        try:
            year_idx = year
            # Use iloc or get() instead of direct indexing to avoid Series.__getitem__ warning
            payment = payment_by_year.get(year, 0) if isinstance(payment_by_year, dict) else payment_by_year.iloc[year] if year < len(payment_by_year) else 0
            
            # Skip years where no one has graduated yet
            if graduated[year_idx] <= 0:
                avg_earnings_repaying.append(0)
                avg_repayment_if_repaying.append(0)
                continue
            
            # Calculate active repaying students (those who are repaying)
            active_repaying = repaying[year_idx]
            
            # For students who have graduated, estimate their average earnings
            if payment > 0 and active_repaying > 0:
                # Average payment per active repaying student
                payment_per_student = payment / active_repaying
                
                # Set average repayment for those who are repaying
                avg_repayment_if_repaying.append(payment_per_student)
                
                # Only calculate earnings for those who are repaying 
                # Since payment = isa_percentage * earnings, we just divide the payment by the ISA percentage
                estimated_earnings_repaying = payment_per_student / isa_percentage
                avg_earnings_repaying.append(estimated_earnings_repaying)
            else:
                avg_earnings_repaying.append(0)
                avg_repayment_if_repaying.append(0)
        except Exception as e:
            # Handle any errors (like KeyError) gracefully
            print(f"Error processing year {year}: {str(e)}")
            avg_earnings_repaying.append(0)
            avg_repayment_if_repaying.append(0)
    
    # Ensure all arrays have the same length
    # Make sure in_school has the right length (should be num_years)
    if len(in_school) > num_years:
        in_school = in_school[:num_years]
    elif len(in_school) < num_years:
        in_school = in_school + [0] * (num_years - len(in_school))
    
    # Make sure all other arrays have the right length
    graduated = graduated[:num_years] if len(graduated) > num_years else graduated + [0] * (num_years - len(graduated))
    graduations_this_year = graduations_this_year[:num_years] if len(graduations_this_year) > num_years else graduations_this_year + [0] * (num_years - len(graduations_this_year))
    repaying = repaying[:num_years] if len(repaying) > num_years else repaying + [0] * (num_years - len(repaying))
    not_repaying_graduated = not_repaying_graduated[:num_years] if len(not_repaying_graduated) > num_years else not_repaying_graduated + [0] * (num_years - len(not_repaying_graduated))
    
    # Make sure avg_repayment_if_repaying has the right length
    avg_repayment_if_repaying = avg_repayment_if_repaying[:num_years] if len(avg_repayment_if_repaying) > num_years else avg_repayment_if_repaying + [0] * (num_years - len(avg_repayment_if_repaying))
    
    # Update the DataFrame with only reliable columns
    payment_df = pd.DataFrame({
        'Year': range(num_years),
        'Students in School': in_school,
        'Graduations': graduations_this_year,
        'Total Graduated': graduated,
        'Students Repaying': repaying,
        'Graduated Not Repaying': not_repaying_graduated,
        'Avg Repayment (Repaying)': [int(x) for x in avg_repayment_if_repaying],
        'Total Payment ($)': [payment_by_year.get(i, 0) if isinstance(payment_by_year, dict) 
                              else payment_by_year.iloc[i] if i < len(payment_by_year) else 0 
                              for i in range(num_years)],
        'Malengo Fee ($)': [malengo_payment_by_year.get(i, 0) if isinstance(malengo_payment_by_year, dict)
                            else malengo_payment_by_year.iloc[i] if i < len(malengo_payment_by_year) else 0
                            for i in range(num_years)]
    })
    
    # Table with only reliable columns
    table = dash_table.DataTable(
        data=payment_df.to_dict('records'),
        columns=[
            {'name': 'Year', 'id': 'Year', 'type': 'numeric'},
            {'name': 'Students in School', 'id': 'Students in School', 'type': 'numeric'},
            {'name': 'Graduations', 'id': 'Graduations', 'type': 'numeric'},
            {'name': 'Total Graduated', 'id': 'Total Graduated', 'type': 'numeric'},
            {'name': 'Students Repaying', 'id': 'Students Repaying', 'type': 'numeric'},
            {'name': 'Graduated Not Repaying', 'id': 'Graduated Not Repaying', 'type': 'numeric'},
            {'name': 'Avg Repayment (Repaying) ($)', 'id': 'Avg Repayment (Repaying)', 'type': 'numeric', 'format': dash_table.FormatTemplate.money(0)},
            {'name': 'Total Payment ($)', 'id': 'Total Payment ($)', 'type': 'numeric', 'format': dash_table.FormatTemplate.money(0)},
            {'name': 'Malengo Fee ($)', 'id': 'Malengo Fee ($)', 'type': 'numeric', 'format': dash_table.FormatTemplate.money(0)}
        ],
        style_table={'overflowX': 'auto'},
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
        ],
        sort_action='native',
        filter_action='native',
        page_size=25,  # Show all years without pagination
    )
    
    # Update the return statement to include clearer explanation text
    return html.Div([
        html.H4(f"{results['program_type']} Program - Detailed Payment & Student Data", style={'textAlign': 'center', 'marginBottom': '20px'}),
        html.Div([
            html.P(f"Total Investment: ${results['total_investment']:,.2f}", style={'fontWeight': 'bold'}),
            html.P(f"Average Total Payment: ${results.get('average_nominal_total_payment', results.get('average_total_payment', 0)):,.2f}", style={'fontWeight': 'bold'}),
            html.P(f"Average Duration: {results['average_duration']:.2f} years", style={'fontWeight': 'bold'}),
            html.P(f"Annual Employment Rate: {results['employment_rate']*100:.1f}%", style={'fontWeight': 'bold'}),
            html.P(f"Students Making Payments: {results['repayment_rate']*100:.1f}%", style={'fontWeight': 'bold'})
        ], style={'marginBottom': '20px', 'textAlign': 'center'}),
        html.P("Students begin repaying in the same year they graduate (assuming they find employment).",
              style={'marginBottom': '15px', 'fontStyle': 'italic', 'fontSize': '0.9em', 'textAlign': 'center'}),
        html.P("The 'Graduations' column shows how many students graduate each year, incorporating the delay distribution.",
              style={'marginBottom': '5px', 'fontStyle': 'italic', 'fontSize': '0.9em'}),
        html.P("'Students Repaying' represents graduates who have found employment and have income above the threshold.",
              style={'marginBottom': '5px', 'fontStyle': 'italic', 'fontSize': '0.9em'}),

        table
    ])

# Function to create IRR comparison chart
def create_irr_comparison(results):
    # Extract IRR values with fallbacks
    real_irr = results.get('IRR', 0) * 100
    nominal_irr = results.get('nominal_IRR', 0) * 100
    real_investor_irr = results.get('investor_IRR', 0) * 100
    nominal_investor_irr = results.get('nominal_investor_IRR', 0) * 100
    
    # Create bar chart comparing real and nominal IRRs
    fig = go.Figure()
    
    # Add Total IRR bars
    fig.add_trace(go.Bar(
        x=['Total IRR (before fees)'],
        y=[real_irr],
        name='Real',
        marker_color='rgb(55, 83, 109)',
        text=[f"{real_irr:.2f}%"],
        textposition='auto'
    ))
    
    fig.add_trace(go.Bar(
        x=['Total IRR (before fees)'],
        y=[nominal_irr],
        name='Nominal',
        marker_color='rgb(26, 118, 255)',
        text=[f"{nominal_irr:.2f}%"],
        textposition='auto'
    ))
    
    # Add Investor IRR bars
    fig.add_trace(go.Bar(
        x=['Investor IRR'],
        y=[real_investor_irr],
        name='Real',
        marker_color='rgb(55, 83, 109)',
        text=[f"{real_investor_irr:.2f}%"],
        textposition='auto',
        showlegend=False
    ))
    
    fig.add_trace(go.Bar(
        x=['Investor IRR'],
        y=[nominal_investor_irr],
        name='Nominal',
        marker_color='rgb(26, 118, 255)',
        text=[f"{nominal_investor_irr:.2f}%"],
        textposition='auto',
        showlegend=False
    ))
    
    fig.update_layout(
        title='Real vs Nominal IRR Comparison',
        xaxis_title='IRR Type',
        yaxis_title='IRR (%)',
        barmode='group',
        template='plotly_white',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

# Callback for IRR comparison
@app.callback(
    Output("irr-comparison", "figure"),
    [Input("simulation-results-store", "data")]
)
def update_irr_comparison(results):
    if not results:
        return go.Figure()
    
    return create_irr_comparison(results)

# Run the app
if __name__ == "__main__":
    # Use this for local development
    app.run(debug=False, host='0.0.0.0', port=10000) 