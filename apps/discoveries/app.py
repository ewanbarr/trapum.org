# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

import dash
import os
import glob
import yaml
import datetime
import base64
#import pandas as pd
import flask
from flask import jsonify
import dash_bootstrap_components as dbc
#import plotly.express as px
import plotly.graph_objects as go
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State

PRODUCTION = True

# --- Constants ---
if PRODUCTION:
    BASE_URL = "/discoveries/"
else:
    BASE_URL = "/"
IMAGE_ROUTE = "/images/" 
IMAGE_DIR = os.path.abspath("../../images/")
YAML_DIR = os.path.abspath("./pulsars/")
TRAPUM_LOGO_LARGE = os.path.join(
    IMAGE_ROUTE,
    "TRAPUM_logo_colour_CMYK_reversed_A_lock-up_04_cropped.png")
APP_TITLE = "Discoveries"

# --- Style ---

SUPPORTED_THEMES = [dbc.themes.LUX, dbc.themes.VAPOR]
THEME = dbc.themes.CYBORG
external_stylesheets = [THEME]
if THEME == dbc.themes.LUX:
    PLOTLY_THEME = "seaborn"
    PLOTLY_TRANSPARENT = False
    DASH_TABLE_STYLE = dict(
        style_cell={'padding': '10px'},
        style_header={
            'backgroundColor': 'black',
            'color': 'white'
        },
        style_data={
            'backgroundColor': 'white',
            'color': 'black'
        },
        style_filter={
            'backgroundColor': 'rgba(200, 200, 200)',
            'color': 'black'
        })
    DEFAULT_GRADIENT = "linear-gradient(rgba(255, 20, 180, 0.6), rgba(255, 255, 255, 1))"
    PULSAR_IMAGE_STYLE = {
            "width": "100%",
            "display": "block",
            "margin-left": "auto",
            "margin-right": "auto",
            }
elif THEME == dbc.themes.VAPOR:
    PLOTLY_THEME = "plotly_dark"
    PLOTLY_TRANSPARENT = True
    DASH_TABLE_STYLE = dict(
        style_cell={'padding': '10px'},
        style_header={
            'backgroundColor': 'rgba(111, 71, 190)',
            'color': 'white',
        },
        style_data={
            'backgroundColor': 'rgb(27, 12, 52)',
            'color': 'white'
        },
        style_filter={
            'backgroundColor': 'rgba(111, 71, 190)',
            'color': 'white'
        })
    DEFAULT_GRADIENT = "linear-gradient(rgba(255, 20, 180, 0.6), rgba(0, 0, 0, 0))"
    PULSAR_IMAGE_STYLE = {
        "width": "100%",
        "display": "block",
        "margin-left": "auto",
        "margin-right": "auto",
        "border": "3px solid black",
        "border-radius": "2%"
        }
elif THEME == dbc.themes.CYBORG:
    PLOTLY_THEME = "plotly_dark"
    PLOTLY_TRANSPARENT = True
    DASH_TABLE_STYLE = dict(
        style_cell={'padding': '10px'},
        style_header={
            'backgroundColor': 'rgba(51, 160, 212)',
            'color': 'white',
            'border': '1px solid white'
            #'border': '1px solid black'
        },
        style_data={
            'backgroundColor': 'rgba(6, 6, 6)',
            'color': 'rgba(220, 220, 220)',
            'border': '1px solid rgba(51, 160, 212)'
        },
        style_filter={
            'backgroundColor': 'rgba(51, 160, 212, 0.6)',
            'color': 'black',
            #'border': '1px solid black'
            'border': '1px solid rgba(51, 160, 212)'
        })
    DEFAULT_GRADIENT = "linear-gradient(rgba(255, 20, 180, 0.6), rgba(0, 0, 0, 0))"
    PULSAR_IMAGE_STYLE = {
        "width": "100%",
        "display": "block",
        "margin-left": "auto",
        "margin-right": "auto",
        "border": "3px solid black",
        "border-radius": "2%"
        }
else:
    raise Exception("Unsupported theme")


# --- App instantiation ---

app = dash.Dash(__name__, requests_pathname_prefix=BASE_URL, external_stylesheets=external_stylesheets)
app.title = APP_TITLE
#app.config.update({"requests_pathname_prefix": BASE_URL})
server = app.server

# --- Helper functions

def load_data(path):
    mapping = {}
    pulsars = glob.glob(os.path.join(path, "*.yaml"))
    for pulsar in pulsars:
        with open(pulsar) as f:
            data = yaml.safe_load(f.read())
        mapping[data["pulsar_parameters"]["name"]] = data
    return mapping

def format_data(records):
    columns = [
        {"name": "PSRJ", "id": "name", "hideable": True},
        {"name": "Period (ms)", "id": "period", "hideable": True},
        {"name": "DM (pc cm^-3)", "id": "dm", "hideable": True},
        {"name": "Binary", "id": "binary", "hideable": True},
        {"name": "Disc. date", "id": "discovery_date", "hideable": True},
        {"name": "Obs. date", "id": "observation_date", "hideable": True},
        {"name": "Disc. band", "id": "discovery_band", "hideable": True},
        {"name": "Discovery S/N", "id": "discovery_snr", "hideable": True},
        {"name": "Project", "id": "project", "hideable": True}
    ]

    data = []
    for pulsar, record in records.items():
        formatted_record = {
            "name": record["pulsar_parameters"]["name"],
            "period": record["pulsar_parameters"]["period"],
            "dm": record["pulsar_parameters"]["dm"],
            "binary": record["pulsar_parameters"]["binary"],
            "discovery_date": record["discovery_parameters"]["discovery_date"],
            "observation_date": record["discovery_parameters"]["observation_date"],
            "discovery_band": record["discovery_parameters"]["discovery_band"],
            "discovery_snr": record["discovery_parameters"]["discovery_snr"],
            "project": record["discovery_parameters"]["project"].upper()
        }
        data.append(formatted_record)

    return columns, data

"""
def format_data(records):
    columns = [{"name": "PSRJ", "id": "name", "hideable": True},
               {"name": "Period (ms)", "id": "period", "hideable": True},
               {"name": "DM (pc cm^-3)", "id": "dm", "hideable": True},
               {"name": "Binary", "id": "binary", "hideable": True},
               {"name": "Disc. date", "id": "discovery_date", "hideable": True},
               {"name": "Obs. date", "id": "observation_date", "hideable": True},
               {"name": "Disc. band", "id": "discovery_band", "hideable": True},
               # {"name": "R.A. (J2000)", "id": "ra"},
               # {"name": "Dec (J2000)", "id": "dec"},
               {"name": "Discovery S/N", "id": "discovery_snr", "hideable": True},
               {"name": "Project", "id": "project", "hideable": True}]
    data = []
    for pulsar, record in records.items():
        formatted_record = {
            "name": record["pulsar_parameters"]["name"],
            "period": record["pulsar_parameters"]["period"],
            "dm": record["pulsar_parameters"]["dm"],
            "binary": record["pulsar_parameters"]["binary"],
            "discovery_date": record["discovery_parameters"]["discovery_date"],
            "observation_date": record["discovery_parameters"]["observation_date"],
            "discovery_band": record["discovery_parameters"]["discovery_band"],
            "discovery_snr": record["discovery_parameters"]["discovery_snr"],
            "project": record["discovery_parameters"]["project"].upper()
        }
        data.append(formatted_record)
    return columns, pd.DataFrame(data)
"""

def make_plot_control(id_, label, params, default):
    return dbc.Row([
        html.Label(label),
        dcc.Dropdown(
            id=id_,
            options=params,
            value=default,
            multi=False,
            clearable=False,
            style={"color": "black"}
            )
        ], justify="center")


class PulsarDetailGenerator:
    def __init__(self, record):
        self._record = record
        self._img_style = PULSAR_IMAGE_STYLE

    def format_image(self, image):
        return html.Img(
            src=os.path.join(IMAGE_ROUTE, image),
            style=self._img_style)

    def format_pulsar_paramters(self):
        pp = self._record["pulsar_parameters"]
        return html.Div(children=[
            html.H3("Pulsar Parameters"),
            html.Br(),
            html.P(["Period: {} ms".format(pp["period"])]),
            html.P(dcc.Markdown(
                "Dispersion measure: {} pc cm<sup>-3</sup>".format(pp["dm"])
                , dangerously_allow_html=True)),
            html.P("Binary: {}".format(pp["binary"])),
            html.Br()
            ])

    def format_discovery_parameters(self):
        dp = self._record["discovery_parameters"]
        return html.Div(children=[
            html.H3("Discovery details"),
            html.Br(),
            html.P("Discovery date: {}".format(dp["discovery_date"])),
            html.P("Observation date: {}".format(dp["observation_date"])),
            html.P("Observation band: {}".format(dp["discovery_band"])),
            html.P("Discovery S/N: {}".format(dp["discovery_snr"])),
            html.P("Pipeline: {}".format(dp["pipeline"])),
            html.P("Project: {}".format(dp["project"])),
            html.Br()
            ])

    def format_associations(self):
        assocs = self._record["associations"]
        div = html.Div(children=[])
        if len(self._record["associations"]) > 0:
            div.children.append(html.H3("Associations"))
            div.children.append(html.Br())
        for assoc in self._record["associations"]:
            div.children.append(html.P("{}, {}".format(
                assoc["name"], assoc["type"])))
        div.children.append(html.Br())
        return div

    def format_additional_content(self):
        container = html.Div(children=[])
        for content in self._record.get("additional_content", []):
            div = html.Div(children=[html.H3(content["title"]), html.Br()])
            for image in content["images"]:
                div.children.append(self.format_image(image))
            div.children.append(html.P(content["body"]))
            div.children.append(html.Br())
            container.children.append(div)
        return container

    def generate(self):
        plot = self._record["discovery_parameters"]["discovery_plot"]
        return dcc.Loading(
            id="pulsar-modal-load",
            type="default",
            children=[
                html.Div(children=[
                    self.format_image(plot),
                    html.Br(),
                    self.format_pulsar_paramters(),
                    self.format_discovery_parameters(),
                    self.format_associations(),
                    self.format_additional_content()
                    ], style={"margin-left": "2%", "margin-right": "2%"})
                ]
            )



def update_plot(xaxis, yaxis, zaxis, logscales):
    print("Plot callback triggered with n_clicks {}".format(n_clicks))
    fig = px.scatter(
        df,
        x=xaxis,
        y=yaxis,
        color=zaxis,
        log_x=True if "logx" in logscales else False,
        log_y=True if "logy" in logscales else False,
        labels={
            xaxis: lable_lookup[xaxis],
            yaxis: lable_lookup[yaxis],
            zaxis: lable_lookup[zaxis]
            },
        template=PLOTLY_THEME,
        hover_data={"name": True}
        )
    if PLOTLY_TRANSPARENT:
        fig.update_layout({
            'plot_bgcolor': 'rgba(0, 0, 0, 0)',
            'paper_bgcolor': 'rgba(0, 0, 0, 0)',
            })
    return [dcc.Graph(id='pulsar-scatter-plot', figure=fig)]

# --- API --

@app.server.route('/api/')
def route1():
    return jsonify(records)


# --- Callbacks ---
"""
@app.callback(Output('pulsar-scatter-plot-col', 'children'),
              Input('update-plot-button-state', 'n_clicks'),
              State('x-axis-selector', 'value'),
              State('y-axis-selector', 'value'),
              State('z-axis-selector', 'value'),
              State('logscale-selector', 'value'))
def update_plot(n_clicks, xaxis, yaxis, zaxis, logscales):
    print("Plot callback triggered with n_clicks {}".format(n_clicks))
    fig = px.scatter(
        df,
        x=xaxis,
        y=yaxis,
        color=zaxis,
        log_x=True if "logx" in logscales else False,
        log_y=True if "logy" in logscales else False,
        labels={
            xaxis: lable_lookup[xaxis],
            yaxis: lable_lookup[yaxis],
            zaxis: lable_lookup[zaxis]
            },
        template=PLOTLY_THEME,
        hover_data={"name": True},
        )
    fig.update_traces(marker={
        "size": 12,
        "line": {
            "width": 2,
            "color": "DarkSlateGrey"
            }
        })
    if PLOTLY_TRANSPARENT:
        fig.update_layout({
            'plot_bgcolor': 'rgba(0, 0, 0, 0)',
            'paper_bgcolor': 'rgba(0, 0, 0, 0)',
            'xaxis': {"gridcolor": "rgba(51, 160, 212)"}
            })
    return [dcc.Graph(
        id='pulsar-scatter-plot',
        figure=fig,
        style={'height': '60vh'})]
"""

"""
@app.callback(
    Output('pulsar-scatter-plot-col', 'children'),
    Input('update-plot-button-state', 'n_clicks'),
    State('x-axis-selector', 'value'),
    State('y-axis-selector', 'value'),
    State('z-axis-selector', 'value'),
    State('logscale-selector', 'value')
)
def update_plot(n_clicks, xaxis, yaxis, zaxis, logscales):
    # Convert list of dicts into x, y, color arrays
    x_vals = [d[xaxis] for d in data]
    y_vals = [d[yaxis] for d in data]
    color_vals = [d[zaxis] for d in data]

    fig = px.scatter(
        x=x_vals,
        y=y_vals,
        color=color_vals,
        log_x=True if "logx" in logscales else False,
        log_y=True if "logy" in logscales else False,
        labels={
            xaxis: label_lookup[xaxis],
            yaxis: label_lookup[yaxis],
            zaxis: label_lookup[zaxis]
        },
        template=PLOTLY_THEME,
        hover_data={"name": [d["name"] for d in data]}
    )

    fig.update_traces(marker={
        "size": 12,
        "line": {"width": 2, "color": "DarkSlateGrey"}
    })

    if PLOTLY_TRANSPARENT:
        fig.update_layout({
            'plot_bgcolor': 'rgba(0, 0, 0, 0)',
            'paper_bgcolor': 'rgba(0, 0, 0, 0)',
            'xaxis': {"gridcolor": "rgba(51, 160, 212)"}
        })

    return [dcc.Graph(id='pulsar-scatter-plot', figure=fig, style={'height': '60vh'})]
"""

@app.callback(
    Output('pulsar-scatter-plot-col', 'children'),
    Input('update-plot-button-state', 'n_clicks'),
    State('x-axis-selector', 'value'),
    State('y-axis-selector', 'value'),
    State('z-axis-selector', 'value'),
    State('logscale-selector', 'value')
)
def update_plot(n_clicks, xaxis, yaxis, zaxis, logscales):
    fig = go.Figure()

    # Group data by zaxis
    groups = {}
    for row in data:
        key = row[zaxis]
        groups.setdefault(key, []).append(row)

    for key, rows in groups.items():
        fig.add_trace(go.Scatter(
            x=[r[xaxis] for r in rows],
            y=[r[yaxis] for r in rows],
            mode="markers",
            name=str(key),
            text=[r["name"] for r in rows],
            customdata=[[r["name"]] for r in rows],
            marker=dict(size=12, line=dict(width=2, color="DarkSlateGrey"))
        ))

    fig.update_layout(
        xaxis_type="log" if "logx" in logscales else "linear",
        yaxis_type="log" if "logy" in logscales else "linear",
        template=PLOTLY_THEME,
        plot_bgcolor='rgba(0,0,0,0)' if PLOTLY_TRANSPARENT else None,
        paper_bgcolor='rgba(0,0,0,0)' if PLOTLY_TRANSPARENT else None
    )

    return [dcc.Graph(id='pulsar-scatter-plot', figure=fig, style={'height': '60vh'})]

def make_pulsar_display_modal(pulsar_name):
    print("Pulsar NAME: ", pulsar_name)
    record = records[pulsar_name]
    generator = PulsarDetailGenerator(record)
    header = dbc.ModalHeader([dbc.ModalTitle(pulsar_name)])
    body = dbc.ModalBody(children=generator.generate())
    footer = dbc.ModalFooter([])
    children = [header, body, footer]
    return children


@app.callback(Output("pulsar-table-detail-modal", "is_open"),
              Output("pulsar-table-detail-modal", "children"),
              Input("pulsar-table", "active_cell"),
              State("pulsar-table", "derived_viewport_data"))
def display_table_click_data(active_cell, derived_viewport_data):
    if active_cell is None:
        return dash.no_update
    pulsar_name = derived_viewport_data[active_cell['row']]['name']
    return True, make_pulsar_display_modal(pulsar_name)


@app.callback(Output("pulsar-graph-detail-modal", "is_open"),
              Output("pulsar-graph-detail-modal", "children"),
              Input('pulsar-scatter-plot', 'clickData'),
              State('pulsar-scatter-plot', 'figure'))
def display_graph_click_data(clickData, figure):
    if clickData is None:
        return dash.no_update
    else:
        pulsar_name = clickData["points"][0]["customdata"][0]
        return True, make_pulsar_display_modal(pulsar_name)

if not PRODUCTION:
    @app.server.route('{}/<subdir>/<image_path>.png'.format(IMAGE_ROUTE))
    def serve_image(subdir, image_path):
        print("request to serve", image_path, "from", subdir)
        image_name = "{}.png".format(image_path)
        return flask.send_from_directory(
            os.path.join(IMAGE_DIR, subdir), image_name)


records = load_data(YAML_DIR)
cols, data = format_data(records)

dropdown_cols = [{"label": i["name"], "value": i["id"]} for i in cols]
label_lookup = {i["id"]: i["name"] for i in cols}

table = dash_table.DataTable(
    id='pulsar-table',
    data=data,  # pass the list of dicts directly
    columns=cols,
    sort_action="native",
    sort_mode="multi",
    filter_action="native",
    column_selectable="single",
    page_action="native",
    page_current=0,
    page_size=30,
    hidden_columns=["discovery_snr", "discovery_band", "observation_date"],
    style_as_list_view=False,
    **DASH_TABLE_STYLE
)

"""
cols, df = format_data(records)
dropdown_cols = [{"label": i["name"], "value": i["id"]} for i in cols]
lable_lookup = {i["id"]: i["name"] for i in cols}
table = dash_table.DataTable(
        id='pulsar-table', data=df.to_dict('records'),
        columns=cols,
        sort_action="native",
        sort_mode="multi",
        filter_action="native",
        column_selectable="single",
        page_action="native",
        page_current=0,
        page_size=30,
        hidden_columns=["discovery_snr", "discovery_band", "observation_date"],
        style_as_list_view=False,
        **DASH_TABLE_STYLE)
"""

def simple_dict_display(title, data, font_size):
    key_style = {
        "text-align": "right",
        "color": "white",
        "font-size": font_size,
        "font-weight": "bold"
    }
    value_style = {
        "text-align": "left",
        "color": "white",
        "font-size": font_size,
    }
    title_style = {
        "text-align": "center"
    }

    cols = []
    for key, value in data.items():
        cols.append(dbc.Col([
            html.Span(f"{key}: ", style=key_style),
            html.Span(f"{value}", style=value_style)
            ]))
    return dbc.Row(cols, justify="centre", style={
            "margin-left": "auto",
            "margin-right": "auto"})


def generate_discovery_stats(data):
    # Count discoveries per project
    by_project = {}
    for row in data:
        proj = row["project"]
        by_project[proj] = by_project.get(proj, 0) + 1

    date = datetime.datetime.strftime(datetime.datetime.now(), "%Y-%m-%d %H:%M")
    stats_panel = dbc.Container([
        simple_dict_display("Total", {"TOTAL DISCOVERIES": len(data)}, font_size=22),
        simple_dict_display("Projects", by_project, font_size=16),
        simple_dict_display("Date", {"LAST UPDATED": date}, font_size=12)
    ], style={"width": "60%"})

    return stats_panel

"""
def generate_discovery_stats(data):
    by_project = data.groupby("project").count()["name"].to_dict()
    date = datetime.datetime.strftime(datetime.datetime.now(), "%Y-%m-%d %H:%M")
    stats_panel = dbc.Container([
        simple_dict_display("Total", {"TOTAL DISCOVERIES": len(data)}, font_size=22),
        simple_dict_display("Projects", by_project, font_size=16),
        simple_dict_display("Date", {"LAST UPDATED": date}, font_size=12)
        ], style={"width": "60%"})
    return stats_panel
"""

plot_controls = [
    make_plot_control("x-axis-selector", "x-axis", dropdown_cols, "period"),
    make_plot_control("y-axis-selector", "y-axis", dropdown_cols, "dm"),
    make_plot_control("z-axis-selector", "z-axis", dropdown_cols, "project"),
    html.Br(),
    dbc.Row(
        dcc.Checklist(
            id="logscale-selector",
            options=[
                {'label': 'log-x', 'value': "logx"},
                {'label': 'log-y', 'value': "logy"}
                ],
            value=["logx"],
            labelStyle={'display': 'block'},
            inputStyle={"margin-right": "15px"}
        )),
    html.Br(),
    dbc.Row(dbc.Button(id='update-plot-button-state',
                       n_clicks=0, children='Update'))
    ]

graph_layout = dbc.Container([
    dbc.Modal(
        id="pulsar-graph-detail-modal",
        size="xl",
        centered=True,
        is_open=False,
        scrollable=True),
    dbc.Row([
        dbc.Col([],
            width=10,
            id='pulsar-scatter-plot-col'),
        dbc.Col(plot_controls, width=2)
        ], justify="center", align="center", style={"padding": "10px"}),
    ])

table_layout = dbc.Container([
    dbc.Modal(
        id="pulsar-table-detail-modal",
        size="xl",
        centered=True,
        is_open=False,
        scrollable=True),
    dbc.Row(children=[table], style={"margin-bottom": "10px"})
    ])

header = dbc.Container(children=[
    html.Img(src=TRAPUM_LOGO_LARGE,
             style={"width": "60%",
                    "display": "block",
                    "align": "left",
                    "margin-left": "auto",
                    "margin-right": "auto"}),

    ], style={"background-color": "transparent"})

footer = dbc.Container(children=[
    dbc.Row(children=[
        html.P("ebarr@mpifr-bonn.mpg.de")
        ])
    ])


tabs = html.Div(
    [
        dbc.Tabs(
            [
                dbc.Tab(label="Overview", tab_id="tab-overview"),
                dbc.Tab(label="Discoveries", tab_id="tab-discoveries"),
                dbc.Tab(label="Publications", tab_id="tab-publications"),
                dbc.Tab(label="Working Groups", tab_id="tab-working-groups"),
                dbc.Tab(label="Members and Institutions", tab_id="tab-members"),
                dbc.Tab(label="Associated Projects", tab_id="tab-projects"),
                dbc.Tab(label="Outreach", tab_id="tab-outreach"),
               ],
            id="tabs",
            active_tab="tab-overview",
        ),
        html.Div(id="content", style={"background-color": "transparent"}),
    ]
)

discoveries_tabs = html.Div(
    [
        dbc.Tabs(
            [
                dbc.Tab(label="Graph", tab_id="tab-disc-graph"),
                dbc.Tab(label="Table", tab_id="tab-disc-table")
               ],
            id="discoveries-tabs",
            active_tab="tab-disc-graph",
        ),
        html.Div(id="discoveries-content",
                 style={
                     "background-color": "transparent",
                     "padding": "10px"
                     }),
    ], style={"padding": "10px"}
)

LAYOUTS = {
    "tab-overview": [],
    "tab-discoveries": [discoveries_tabs],
    "tab-publications": [],
    "tab-working-groups": [],
    "tab-members": [],
    "tab-projects": [],
    "tab-outreach": [],
}

"""
@app.callback(Output("content", "children"), [Input("tabs", "active_tab")])
def tab_content(active_tab):
    return LAYOUTS.get(active_tab, dash.no_update)
"""

@app.callback(Output("discoveries-content", "children"), [Input("discoveries-tabs", "active_tab")])
def tab_content(active_tab):
    if active_tab == "tab-disc-graph":
        return graph_layout
    else:
        return table_layout


app.layout = dbc.Container(children=[
    header,
    generate_discovery_stats(data),
    dbc.Container([discoveries_tabs]),
    footer,
    html.Br()
    ], fluid=True)
"""
, style={
        "background-color": "transparent",
        "height": "100%",
        "overflow": "auto",
        "position": "absolute",
        "bottom": 0,
        "top": 0,
        "left": 0,
        "right": 0
        })
"""

if __name__ == '__main__':
    app.run_server(debug=False)
