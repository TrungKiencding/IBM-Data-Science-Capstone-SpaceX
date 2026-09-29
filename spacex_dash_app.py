# Plotly Dash dashboard for SpaceX launch records. Run: python spacex_dash_app.py
import pandas as pd, dash, plotly.express as px
from dash import html, dcc
from dash.dependencies import Input, Output

spacex_df = pd.read_csv("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/datasets/spacex_launch_dash.csv")
max_payload, min_payload = spacex_df["Payload Mass (kg)"].max(), spacex_df["Payload Mass (kg)"].min()
sites = [{"label": "All Sites", "value": "ALL"}] + [{"label": s, "value": s} for s in spacex_df["Launch Site"].unique()]

app = dash.Dash(__name__)
app.layout = html.Div([
    html.H1("SpaceX Launch Records Dashboard", style={"textAlign": "center", "color": "#503D36", "font-size": 40}),
    dcc.Dropdown(id="site-dropdown", options=sites, value="ALL", placeholder="Select a Launch Site here", searchable=True),
    html.Br(),
    html.Div(dcc.Graph(id="success-pie-chart")),
    html.Br(),
    html.P("Payload range (Kg):"),
    dcc.RangeSlider(id="payload-slider", min=0, max=10000, step=1000, marks={0: "0", 2500: "2500", 5000: "5000", 7500: "7500", 10000: "10000"},
                    value=[min_payload, max_payload]),
    html.Div(dcc.Graph(id="success-payload-scatter-chart")),
])

@app.callback(Output("success-pie-chart", "figure"), Input("site-dropdown", "value"))
def get_pie_chart(site):
    if site == "ALL":
        return px.pie(spacex_df, values="class", names="Launch Site", title="Total Successful Launches by Site")
    counts = spacex_df[spacex_df["Launch Site"] == site]["class"].value_counts().reset_index()
    counts.columns = ["class", "count"]
    return px.pie(counts, values="count", names="class", title=f"Total Launch Outcomes for {site}")

@app.callback(Output("success-payload-scatter-chart", "figure"), [Input("site-dropdown", "value"), Input("payload-slider", "value")])
def get_scatter_chart(site, payload):
    df = spacex_df[spacex_df["Payload Mass (kg)"].between(payload[0], payload[1])]
    if site != "ALL":
        df = df[df["Launch Site"] == site]
    return px.scatter(df, x="Payload Mass (kg)", y="class", color="Booster Version Category",
                      title=f"Correlation between Payload and Success for {'all Sites' if site == 'ALL' else site}")

if __name__ == "__main__":
    app.run(debug=False)
