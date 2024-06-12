# This is a testfile for the dash F1-Application. None of the current data represents any real Tracker Data but is just a test for how to update graphs etc.

# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.
import os
import json
import pandas as pd
import numpy as np
import datetime as dt
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, html, dcc, Input, Output, State
from dash import dash_table
from dash.exceptions import PreventUpdate
import dash_daq as daq
import dash_bootstrap_components as dbc
from plotly.subplots import make_subplots
from itertools import islice
from datetime import timedelta
import json

#https://results.elections.europa.eu/en/tools/download-datasheets/
path = r''
with open(os.path.join(path, r'assets\europe2.geojson')) as f:
    eu_map_json = json.load(f)

app = Dash(__name__, external_stylesheets=[dbc.themes.DARKLY, "assets/base.css"])
# Global vars
currentgraphzoom = {"xaxis.autorange": True, "yaxis.autorange": True}
turnout_df = pd.read_csv(os.path.join(path, 'turnout-eu.csv'), sep=';')
turnout_country_df = pd.read_csv(os.path.join(path, r'turnout-country.csv'), sep=';')
eu_df = pd.read_csv(os.path.join(path, r'eu.csv'), sep=';')
eu_df_groups = pd.read_csv(os.path.join(path, r'groups.csv'), sep=';')


def generate_table(dataframe, max_rows=10, topleft="", headings=None):
    """
    Generates a HTML Table with max_rows out of a Pandas Dataframe.
    :param dataframe: DataFrame which should be displayed in HTML.
    :param max_rows: Maximal amount of rows which will be displayed in the HTML.
    :return: Generated HTML Table.
    """
    return html.Table([
        html.Thead(
            html.Tr([html.Th(topleft)] + [html.Th(headings[idx] if headings else col) for idx, col in enumerate(dataframe.columns)])
        ),
        html.Tbody([
            html.Tr([html.Td(dataframe.iloc[i].name)] + [
                html.Td(dataframe.iloc[i][col]) for col in dataframe.columns],
                className="table-success" if i == 0 else None) for i in range(min(len(dataframe), max_rows))
        ])
    ],  className="table table-hover")


switchtheme = {
    'dark': True,
    'detail': '#003399',
    'primary': '#FFCC00',
    'secondary': '#003399',
}

SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "18rem",
    "padding": "2rem 1rem",
    "background-color": "#003399", #632D47
}

CONTENT_STYLE = {
    "margin-left": "20rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
}

sidebar = html.Div(
    [
        html.Img(src=app.get_asset_url("images/eu_flag.png"), style={"width": "100%"}),
        html.H2("European\nElection \nResults", className="display-4"),
        html.Hr(),
        html.P(
            "Voting results 2024:", className="lead"
        ),
        dbc.Nav(
            [
                dbc.NavLink("Home", href="/", active="exact"),
                dbc.NavLink("European Union", href="/europe_2024", active="exact"),
                dbc.NavLink("Memberstates", href="/ms_none", active="exact"),
                dbc.NavLink("Factions", href="/factions", active="exact"),
                dbc.NavLink("About", href="/about", active="exact"),
            ],
            vertical=True,
            pills=True,
        ),
    ],
    style=SIDEBAR_STYLE)


def home_page():
    cards = [dbc.Card(
    [
        dbc.CardImg(src=app.get_asset_url("images/eu_page.jpg"), top=True),
        dbc.CardBody(
            [
                html.H4("European Results", className="EU"),
                html.P(
                    "View European Turnout Results"
                    "and see how votes are distributed on a european level.",
                    className="card-text",
                ),
                dbc.Button("View", href="/europe_2024", color="primary"),
            ]
        )
    ],
    style={"width": "18rem"},
    ),
    dbc.Card(
    [
        dbc.CardImg(src=app.get_asset_url("images/factions.jpg"), top=True),
        dbc.CardBody(
            [
                html.H4("Factions", className="EU"),
                html.P(
                    "View Factions and their Acronyms"
                    "",
                    className="card-text",
                ),
                dbc.Button("View", href="/factions", color="primary"),
            ]
        )
    ],
    style={"width": "18rem"},
    ),
    dbc.Card(
    [
        dbc.CardImg(src=app.get_asset_url("images/memberstates.jpg"), top=True),
        dbc.CardBody(
            [
                html.H4("Memberstate Results", className="EU"),
                html.P(
                    "View Country Turnout and Results"
                    "and see how votes are distributed on a memberstate level.",
                    className="card-text",
                ),
                dbc.Button("View", href="/ms_none", color="primary"),
            ]
        )
    ],
    style={"width": "18rem"},
    ),
    dbc.Card(
    [
        dbc.CardImg(src=app.get_asset_url("images/parliament.png"), top=True),
        dbc.CardBody(
            [
                html.H4("Sources", className="EU"),
                html.P(
                    "Sources "
                    "and author of this notebook",
                    className="card-text",
                ),
                dbc.Button("View", href="/about", color="primary"),
            ]
        )
    ],
    style={"width": "18rem"},
    )]
    return html.Div([
        html.H1("Welcome to the European Election Results Dashboard"),
        html.P("This dashboard is a test for the visualization of the European Election Results."),
        html.P("Please select a card or the pages on the sidebar to get started."),
        dbc.Row([cards[0], cards[1], cards[2], cards[3]])
    ])

def page_european_results(year):
    min_year = min(turnout_df.YEAR)
    max_year = max(turnout_df.YEAR)
    year_slider = dcc.RangeSlider(min=min_year, max=max_year, step=5, value=[min_year, max_year],
                                 tooltip={"always_visible": True, "style": {"color": "LightSteelBlue", "fontSize": "15px"}, "placement": "bottom"},
                                 marks={i: str(i) for i in range(min_year, max_year, 5)},
                                 id='year-slider')
    content = [ dbc.Row(dbc.Col(html.H1(children='Results aross Europe: ' + str(year)))),
                dbc.Row([dbc.Col(daq.BooleanSwitch(id='previous_years_switch', on=False, label="View previous years", labelPosition="bottom"), width=2),
                         dbc.Col(html.Div(year_slider), width=10, id='hide_year_slider')]),
                dbc.Row([dcc.Graph(id='turnout-rate-plot1', 
                                   config={'modeBarButtonsToRemove': ['autoScale2d'],
                                          'modeBarButtonsToAdd': ['drawline', 'drawopenpath', 'eraseshape'],
                                          'displaylogo': False})]),
                dbc.Row([html.P("Turnout Rate per country:", style={'font-size': 20})]),
                dbc.Row([dcc.Graph(id="map")]),
                html.Div(children="Test", id="eu_vote_results")]

    return content


def page_memberstate_results(memberstate):
    memberstates = turnout_country_df.COUNTRY_ID.unique()
    buttons = [dbc.Button(memberstate, href=f"ms_{memberstate}", color="primary") for memberstate in memberstates]
    if memberstate not in memberstates:
        return [html.H1("Choose a member state to view the results:"), 
                html.Div("\n"),
                html.Div(buttons),
                html.Div("\n"),
                dbc.Row(dbc.Button("Back to EU", href="/europe_2024", color="primary"))]
    year = 2024 # placeholder for further extension
    # Create the choropleth map
    voters = turnout_country_df[(turnout_country_df.YEAR == year) & (turnout_country_df.COUNTRY_ID == memberstate.upper())].RATE.values[0]
    fig = go.Figure(go.Pie(labels=["Voters", "Did not vote"],
                    marker_colors=('crimson', 'lightgrey'),
                    hoverinfo='label+percent',
                    values=[voters, 100-voters], hole=.3, name="Turnout Rate"))
    fig.update_layout(template="plotly_dark", plot_bgcolor='rgba(0, 0, 0, 0)', paper_bgcolor='rgba(0, 0, 0, 0)', margin={"r":0,"t":0,"l":0,"b":0}, font=dict(size=20))
    
    try:
        df = pd.read_csv(os.path.join(path, 'memberstates', f'results-parties-{memberstate.lower()}.csv'), sep=';')
        df_b = pd.read_csv(os.path.join(path, 'breakdown', f'{memberstate.lower()}.csv'), sep=';')
        merged_df = pd.merge(df, df_b, on='PARTY_ID', suffixes=('_party', '_breakdown'))
    except FileNotFoundError:
        return [html.H1("No data available for this memberstate"), dbc.Button("Back to EU", href="/europe_2024", color="primary")]
    fig2 = go.Figure(go.Bar(x=merged_df.GROUP_ID, y=merged_df.VOTES_PERCENT, text=merged_df.SEATS_TOTAL, textposition='auto',
                        name='Voters',
                        orientation='v', showlegend=False, 
                        hoverinfo='x+y+text'))
    fig2.update_layout(template="plotly_dark", plot_bgcolor='rgba(0, 0, 0, 0)', paper_bgcolor='rgba(0, 0, 0, 0)', margin={"r":0,"t":0,"l":0,"b":0},
                       font=dict(size=20), barcornerradius=15)
    
    content = [ dbc.Row(dbc.Col(html.H1(children=f'Results in {str(memberstate).upper()}: ' + str(year)))),
                #dbc.Row([dbc.Col(daq.BooleanSwitch(id='show_assosiations', on=False, label="View EU factions", labelPosition="bottom"), width=2)]),
                dbc.Row([dcc.Graph(figure=fig,id='turnout-rate-plot2', config={'modeBarButtonsToRemove': ['autoScale2d'],
                                          'modeBarButtonsToAdd': ['drawline', 'drawopenpath', 'eraseshape'],
                                          'displaylogo': False})]),
                dbc.Row([html.P(f"Percentage of votes in {str(memberstate).upper()}:", style={'font-size': 20})]),
                dbc.Row([dcc.Graph(figure=fig2)]),
                dbc.Row(dbc.Button("Back to EU", href="/europe_2024", color="primary"))]

    return content

def page_about():
    return html.Div([
            html.H1("About"),
            html.P("This is a test dashboard for the visualization of the European Election Results."),
            html.P("The data is real and taken from:"),
            html.A("European Election Results", href="https://results.elections.europa.eu/en/tools/download-datasheets/"),
            html.P("The dashboard is created with Dash and Plotly."),
            html.P("Made by: Thomas Landl and Adrian Vitzthum-Lettner")])

def page_factions():
    data = eu_df_groups.to_dict('records')
    return [
        html.H1("Factions"),
        html.Div([
        dash_table.DataTable(
            columns=[
                {'name': 'Group', 'id': 'ID', 'type': 'text'},
                {'name': 'Language', 'id': 'LANGUAGE_ID', 'type': 'text'},
                {'name': 'Acronym', 'id': 'ACRONYM', 'type': 'text'},
                {'name': 'Label Expectancy', 'id': 'LABEL', 'type': 'text'}
            ],
            data=data,
            filter_action='native',
            style_table={
                'height': 400,
                'color': 'white',  # Set text color to white
                'background-color': '#222323',  # Set background color to dark grey
                'border': 'none'  # Remove border
            },
            style_data_conditional=[
                {
                    'if': {'column_editable': True},
                    'background-color': '#222323', 
                    'color': 'white',  
                    'border': 'none'  
                },
                {
                    'if': {'column_editable': False},
                    'background-color': '#222323', 
                    'color': 'white',
                    'border': 'none' 
                }
            ],
            style_header={
                'background-color': '#222323',  
                'border': 'none'  
            },
            style_cell={
                'textAlign': 'left',
                'whiteSpace': 'normal',
                'overflow': 'hidden',
                'textOverflow': 'ellipsis'
            },
            style_cell_conditional=[
                {'if': {'column_id': 'ID'}, 'width': '30px'},  # Adjusting width for specific columns
                {'if': {'column_id': 'LANGUAGE_ID'}, 'width': '150px'},
                {'if': {'column_id': 'ACRONYM'}, 'width': '80px'},
                {'if': {'column_id': 'LABEL'}, 'width': '120px'}
            ]
        )])
    ]



# MAIN APP LAYOUT
content = html.Div(id="page-content", style=CONTENT_STYLE)
app.layout = html.Div(children=[dcc.Location(id="url"), sidebar, daq.DarkThemeProvider(theme=switchtheme, children=content)])
app.config.suppress_callback_exceptions = True


@app.callback(Output('page-content', 'children'),
              Input('url', 'pathname'))
def render_page_content(pathname):
    global page
    if pathname == "/":
        return home_page()
    if pathname == "/about":
        return page_about()
    elif pathname == "/factions":
        return page_factions()
    elif pathname.split("_")[0] == "/europe":
        return page_european_results(pathname.split("_")[1])
    elif pathname.split("_")[0] == "/ms":
        return page_memberstate_results(pathname.split("_")[1])
    # If the user tries to reach a different page, return a 404 message
    return [html.H1("404: Not found", className="text-danger"),
            html.Hr(),
            html.P(f"The pathname {pathname} was not recognised...")]


@app.callback(Output('turnout-rate-plot1', 'figure'),
              Input('previous_years_switch', 'on'),
              Input('year-slider', 'value'),
              Input('url', 'pathname'))
def update_turnout_plot1(prev_years, limits, pathname):
    if not pathname.startswith("/europe"):
        raise PreventUpdate
    if prev_years and limits:
        fig = go.Figure(data=[go.Bar(x=turnout_df.YEAR[(turnout_df.YEAR <= limits[1]) & (turnout_df.YEAR >= limits[0])], marker_color='crimson', y=turnout_df.RATE, text=turnout_df.RATE, textposition='auto', name="Turnout Rate")],
                        layout=dict(template="plotly_dark", barcornerradius=15, title="Turnout Rate over the selected years"))
        fig.update_layout(plot_bgcolor='rgba(0, 0, 0, 0)', paper_bgcolor='rgba(0, 0, 0, 0)', font=dict(size=15))
    else:
        voters = turnout_df[turnout_df.YEAR == int(pathname.split("_")[1])].RATE.values[0]
        countries = turnout_country_df[turnout_country_df.YEAR == int(pathname.split("_")[1])].sort_values("RATE").COUNTRY_ID.values.tolist()
        country_voters = turnout_country_df[turnout_country_df.YEAR == int(pathname.split("_")[1])].sort_values("RATE").RATE.values
        fig = make_subplots(rows=1, cols=2, subplot_titles=("Turnout Rate", "Turnout Rate per Country"), specs=[[{'type':'domain'}, {'type':'xy'}]])
        fig.add_trace(go.Bar(x=country_voters, y=countries, text=country_voters, textposition='auto',
                        base=0,
                        marker_color='crimson',
                        name='Voters',
                        orientation='h', showlegend=False,
                        hoverinfo='y+text'), row=1, col=2)
        
        fig.add_trace(go.Pie(
                    labels=["Voters", "Did not vote"],
                    marker_colors=('crimson', 'lightgrey'),
                    hoverinfo='label+percent',
                    values=[voters, 100-voters], domain=dict(x=[0.0, 0.5]), hole=.3, name="Turnout Rate"), row=1, col=1)
        fig.update_layout(template="plotly_dark", plot_bgcolor='rgba(0, 0, 0, 0)', paper_bgcolor='rgba(0, 0, 0, 0)', height=800, width=1400, font=dict(size=15))
    return fig

@app.callback(Output('hide_year_slider', 'children'),
              Input('url', 'pathname'),
              Input('previous_years_switch', 'on'))
def hide_year_slider(pathname, prev_years):
    if not pathname.startswith("/europe"):
        raise PreventUpdate
    if prev_years:
        min_year = min(turnout_df.YEAR)
        max_year = max(turnout_df.YEAR)
        year_slider = dcc.RangeSlider(min=min_year, max=max_year, step=5, value=[min_year, max_year],
                                 tooltip={"always_visible": True, "style": {"color": "white", "fontSize": "15px"}, "placement": "bottom"},
                                 marks={i: str(i) for i in range(min_year, max_year, 5)},
                                 id='year-slider', className = "slider")
        return year_slider
    else:
        return ""
    
@app.callback(Output('url', 'pathname'),
              State('previous_years_switch', 'on'),
              Input('turnout-rate-plot1', 'clickData'),
              Input('map', 'clickData'))
def when_clicking_on_turnoutplot(prev_years, data_clicked, map_clicked):
    if data_clicked is None and map_clicked is None:
        raise PreventUpdate
    if map_clicked:
        if "location" in map_clicked['points'][0]:
            return f"/ms_{map_clicked['points'][0]['location']}"
        else:  
            raise PreventUpdate
    if prev_years:
        if data_clicked:
            year = data_clicked["points"][0]["x"]
            return f"/europe_{year}"
    else:
        if not prev_years and data_clicked:
            if "y" in data_clicked["points"][0]:
                return f"/ms_{str(data_clicked['points'][0]['y']).lower()}"
            else:
                raise PreventUpdate
            
    
@app.callback(Output("map", "figure"),
              Input('previous_years_switch', 'on'),
              Input('year-slider', 'value'),
              Input('url', 'pathname'))
def display_choropleth(prev_years, limits, pathname):
    if not pathname.startswith("/europe"):
        raise PreventUpdate
    if prev_years and limits:
        countries = turnout_country_df[(turnout_country_df.YEAR <= limits[1]) & (turnout_country_df.YEAR >= limits[0])].groupby('COUNTRY_ID')['RATE'].mean().reset_index()
        
        # Create the choropleth map
        fig = px.choropleth_mapbox(
            countries,
            geojson=eu_map_json,
            locations='COUNTRY_ID',
            featureidkey='properties.ISO2',
            color='RATE',
            color_continuous_scale="Bluered",
            range_color=(0, 100),
            mapbox_style="carto-positron",
            zoom=3,
            center={"lat": 54.5260, "lon": 15.2551},
            opacity=0.5,
            labels={'RATE': 'Turnout Rate\n(avg.)'},
            hover_data={'COUNTRY_ID': False, 'RATE': True}
        )
    else:
        countries = turnout_country_df[turnout_country_df.YEAR == int(pathname.split("_")[1])]
        # Create the choropleth map
        fig = px.choropleth_mapbox(
            countries,
            geojson=eu_map_json,
            locations='COUNTRY_ID',
            featureidkey='properties.ISO2',
            color='RATE',
            color_continuous_scale="Bluered",
            range_color=(0, 100),
            mapbox_style="carto-positron",
            zoom=3,
            center={"lat": 54.5260, "lon": 15.2551},
            opacity=0.5,
            labels={'RATE': 'Turnout Rate'},
            hover_data={'COUNTRY_ID': False, 'RATE': True}
        )

    fig.update_layout(template="plotly_dark", plot_bgcolor='rgba(0, 0, 0, 0)', paper_bgcolor='rgba(0, 0, 0, 0)', margin={"r":0,"t":0,"l":0,"b":0}, font=dict(size=15))
    return fig
    
@app.callback(Output("eu_vote_results", "children"),
              Input('previous_years_switch', 'on'),
              Input('url', 'pathname'))
def display_vote_results(prev_years, pathname):
    if prev_years or not pathname.startswith("/europe_2024") :
        return "No further data available"
    fig = go.Figure(data=[go.Pie(
        labels=eu_df.GROUP_ID.tolist(),
        values=(eu_df.SEATS_PERCENT_EU).tolist(),
        text=eu_df.SEATS_TOTAL,
        hole=0.5,
        pull=[0.05] * len(eu_df.SEATS_PERCENT_EU),
        rotation=0,
        direction='clockwise',
        hoverinfo='label+text')])
    fig.update_layout(template="plotly_dark", plot_bgcolor='rgba(0, 0, 0, 0)', paper_bgcolor='rgba(0, 0, 0, 0)', margin=dict(t=0, b=0, l=0, r=0), font=dict(size=18))

    fig.update_traces(
        showlegend=False,
        textinfo='label+percent',
        textposition='outside',
    )

    return [dbc.Row([html.H1("Vote Results")]), dbc.Row([dcc.Graph(figure=fig)]), generate_table(eu_df, 10, "EU-Parliament", ["Group", "Seats", "Seats (%)", "Status", "Update Time"])]

if __name__ == '__main__':
    app.run_server(debug=False)
