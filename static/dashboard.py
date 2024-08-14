import json
import pandas as pd
from dash import Dash, html, dcc
import plotly.express as px
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
from flask import Flask

# Initialize the Flask server
server = Flask(__name__)

# Initialize the Dash app, connecting it to the Flask server
app = Dash(__name__, server=server, external_stylesheets=[dbc.themes.FLATLY])

# Function to load JSON data
def load_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

# Load the JSON data
json_path = r"C:\Users\simo2\source\repos\FassehTest\FassehTest\results.json"
data = load_json(json_path)

# Normalize the JSON data into a DataFrame
df = pd.json_normalize(data)

# Ensure the DataFrame contains the necessary columns
assert 'model' in df.columns and 'conf' in df.columns and 'classes' in df.columns, "DataFrame must contain 'model', 'conf' and 'classes' columns."

# Explode the 'classes' and 'conf' columns to separate rows
df = df.explode(['classes', 'conf'])

# Replace specific class names with Arabic equivalents
df['classes'].replace({
    'Happy': '”⁄Ìœ',
    'Sad': 'Õ“Ì‰',
    'Stress': '„ Ê —',
    'Natural': 'ÿ»Ì⁄Ì'
}, inplace=True)

# Define colors for charts
dark_colors = ['#2c3e50', '#8e44ad', '#1abc9c', '#16a085']  # Adjusted color for '”⁄Ìœ'

# Function to create charts for each model
def generate_charts(model):
    filtered_df = df[df['model'] == model]
    if filtered_df.empty:
        return None

    if model == "Face_Expression":
        # Define specific colors for each emotion
        color_map = {
            'ÿ»Ì⁄Ì': '#1f77b4',  # Blue
            '”⁄Ìœ': '#ff7f0e',   # Orange
            'Õ“Ì‰': '#2ca02c'    # Green
        }
        
        # Generate bar chart for Face Expression with specific colors for each emotion
        fig = px.bar(
            filtered_df,
            x='classes',
            y='conf',
            title=f'Face Expression Distribution for {model}',
            labels={'conf': '«·Àﬁ…', 'classes': '«·„‘«⁄—'},
            color='classes',
            color_discrete_map=color_map  # Apply the specific color map
        )
        
        # Customize bar chart appearance
        fig.update_traces(
            marker=dict(line=dict(width=2, color='DarkSlateGrey')),  # Add border to bars
            textposition='none',  # Remove text labels
            width=0.5  # Adjust bar width
        )
        
        # Update layout to enhance visual appearance
        fig.update_layout(
            plot_bgcolor='#f0f0f0',  # Background color of the plot area
            paper_bgcolor='#faf1e2',  # Background color of the paper
            font=dict(size=14),  # Font size for the entire chart
            title_font_size=20,  # Font size for the title
            xaxis_tickangle=-45  # Rotate x-axis labels for better readability
        )
        
        return fig
    elif model in ["Stand", "Confidence_Score", "Eyes_Gaze"]:
        # Generate gauge chart for Stand, Confidence_Score, and Eyes_Gaze
        avg_conf = filtered_df['conf'].mean() * 100  # Convert to percentage
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=avg_conf,
            title={'text': f"{model} Confidence"},
            gauge={'axis': {'range': [0, 100]}, 'bar': {'color': dark_colors[0]}}  # Darker gauge bar
        ))
        return fig
    else:
        return None

# Generate charts for all models
charts = {}
for model in df['model'].unique():
    chart = generate_charts(model)
    if chart:
        charts[model] = chart

# Define layout with dropdown and dynamic content
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H1("Model Confidence Dashboard", className="text-center mb-4"), width=12)
    ]),
    dbc.Row([
        dbc.Col(
            dcc.Dropdown(
                id='model-dropdown',
                options=[{'label': model, 'value': model} for model in charts.keys()],
                value=list(charts.keys())[0],
                clearable=False,
                style={'margin-bottom': '20px'}
            ),
            width=6
        )
    ], justify='center'),
    html.Div(id='charts-container')
], fluid=True, style={'backgroundColor': '#faf1e2'})

@app.callback(
    Output('charts-container', 'children'),
    [Input('model-dropdown', 'value')]
)
def update_charts(model):
    chart = charts.get(model)
    if chart is None:
        return "No data available for this model."

    if isinstance(chart, go.Figure):  # Use go.Figure instead of px.Figure
        return dcc.Graph(figure=chart, config={'displayModeBar': False})
    else:
        return dbc.Row([dbc.Col(chart, width=12)])

# Running the server
if __name__ == "_main_":
    app.run_server(debug=True, port=8050)
