import dash
from dash import html, dcc, Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load initial dataset
data_path = "DeployWithRender/exported_data.parquet"  # Replace with your dataset path
df = pd.read_parquet(data_path)
df["CustomerID"] = df["CustomerID"].astype(str)
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

# Prepare data for clustering
rfm_data = df[["CustomerID", "log_Recency", "log_Frequency", "log_MonetaryValue"]].copy()
rfm_data.set_index("CustomerID", inplace=True)

# Scale data for clustering
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm_data)

# Apply KMeans clustering
kmeans = KMeans(n_clusters=4, random_state=42)
rfm_data["Cluster"] = kmeans.fit_predict(rfm_scaled)

# Add cluster descriptions
cluster_descriptions = {
    0: "Recent Buyers with High Frequency and High Monetary Value.",
    1: "Moderate Recency, Moderate Frequency, Moderate Monetary Value.",
    2: "Infrequent Buyers with Low Monetary Value.",
    3: "Occasional Buyers with Moderate Monetary Value."
}

# Initialize Dash app with Bootstrap (dark theme)
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

# Function to generate 3D scatter plot for clusters
def generate_cluster_plot(new_data_point=None):
    fig = px.scatter_3d(
        rfm_data,
        x="log_Recency",
        y="log_Frequency",
        z="log_MonetaryValue",
        color="Cluster",
        title="Customer Clustering (3D)",
        labels={
            "log_Recency": "Log Recency",
            "log_Frequency": "Log Frequency",
            "log_MonetaryValue": "Log Monetary Value",
        },
        hover_data=["Cluster"],
        template="plotly_dark"
    )
    if new_data_point:
        fig.add_scatter3d(
            x=[new_data_point["log_Recency"]],
            y=[new_data_point["log_Frequency"]],
            z=[new_data_point["log_MonetaryValue"]],
            mode="markers",
            marker=dict(size=10, color="red", symbol="circle"),
            name="Predicted Customer"
        )
    return fig

# Layout of the app
app.layout = html.Div(
    style={"backgroundColor": "#121212", "padding": "20px"},
    children=[
        dbc.Container(
            [
                dbc.Row(
                    dbc.Col(html.H1("Customer Insights and Recommendations", className="text-center mb-4", style={"color": "#00BCD4"}), width=12)
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.H5("Inputs", style={"color": "#00BCD4"}),
                                dbc.Form(
                                    [
                                        dbc.Label("Select Product", style={"color": "white"}),
                                        dcc.Dropdown(
                                            id="product-dropdown",
                                            options=[{"label": desc, "value": desc} for desc in df["Description"].unique()],
                                            placeholder="Select a Product",
                                            style={"color": "black"}
                                        ),
                                        dbc.Label("Enter Quantity", style={"color": "white"}),
                                        dbc.Input(id="quantity-input", type="number", placeholder="Enter quantity", className="mb-3"),
                                        dbc.Label("Select Country", style={"color": "white"}),
                                        dcc.Dropdown(
                                            id="country-dropdown",
                                            options=[{"label": country, "value": country} for country in df["Country"].unique()],
                                            placeholder="Select a Country",
                                            style={"color": "black"}
                                        ),
                                        dbc.Button("Submit", id="submit-button", color="primary", className="mt-3"),
                                    ]
                                ),
                            ],
                            width=4,
                        ),
                        dbc.Col(
                            [
                                html.H5("RFM Metrics for the User", style={"color": "#00BCD4"}),
                                html.Div(id="rfm-metrics", className="border p-3 mb-3", style={"backgroundColor": "#1E1E1E", "color": "white", "border-radius": "5px"}),
                                html.H5("Cluster Descriptions", style={"color": "#00BCD4"}),
                                html.Ul(
                                    [
                                        html.Li(
                                            html.Span([
                                                html.Strong(f"{key}: "),
                                                value
                                            ]),
                                            style={"color": "white", "padding": "5px"}
                                        ) for key, value in cluster_descriptions.items()
                                    ],
                                    className="border p-3",
                                    style={
                                        "backgroundColor": "#1E1E1E",
                                        "border-radius": "5px",
                                        "list-style-position": "inside"  # قرار دادن بولت‌ها داخل کادر
                                    }
                                ),
                            ],
                            width=8,
                        ),
                    ],
                    className="mb-4",
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.H5("Total Value", style={"color": "#00BCD4"}),
                                html.Div(id="total-value", className="border p-3 mb-3", style={"backgroundColor": "#1E1E1E", "color": "white", "border-radius": "5px"}),
                            ],
                            width=4,
                        ),
                        dbc.Col(
                            [
                                html.H5("Similar Products", style={"color": "#00BCD4"}),
                                html.Div(id="similar-products", className="border p-3 mb-3", style={"backgroundColor": "#1E1E1E", "color": "white", "border-radius": "5px"}),
                            ],
                            width=4,
                        ),
                        dbc.Col(
                            [
                                html.H5("Predicted Cluster", style={"color": "#00BCD4"}),
                                html.Div(id="predicted-cluster", className="border p-3 mb-3", style={"backgroundColor": "#1E1E1E", "color": "white", "border-radius": "5px"}),
                            ],
                            width=4,
                        ),
                    ],
                    className="mb-4",
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.H5("3D Cluster Visualization", style={"color": "#00BCD4"}),
                                dcc.Graph(id="cluster-graph", style={"height": "600px", "width": "100%"}),
                            ],
                            width=12,
                        ),
                    ],
                    className="mt-4",
                ),
            ],
            fluid=True,
        ),
    ]
)

# Callback to calculate order summary, similar products, RFM metrics, and prediction
@app.callback(
    [Output("cluster-graph", "figure"),
     Output("predicted-cluster", "children"),
     Output("similar-products", "children"),
     Output("total-value", "children"),
     Output("rfm-metrics", "children")],
    Input("submit-button", "n_clicks"),
    [State("product-dropdown", "value"),
     State("quantity-input", "value"),
     State("country-dropdown", "value")]
)
def handle_order(n_clicks, product, quantity, country):
    if n_clicks and n_clicks > 0 and product and quantity and country:
        # Calculate Total Cost
        product_data = df[df["Description"] == product]
        unit_price = product_data["UnitPrice"].iloc[0]
        total_cost = unit_price * quantity

        # Predict RFM for the order
        recency = 1  # Example value
        frequency = quantity
        monetary = total_cost

        new_data = pd.DataFrame([[recency, frequency, monetary]],
                                columns=["log_Recency", "log_Frequency", "log_MonetaryValue"])
        new_data_scaled = scaler.transform(new_data)
        predicted_cluster = kmeans.predict(new_data_scaled)[0]

        # Generate updated cluster plot
        cluster_plot = generate_cluster_plot(new_data_point={
            "log_Recency": recency,
            "log_Frequency": frequency,
            "log_MonetaryValue": monetary
        })

        # Find similar products
        similar_products = df[df["Description"].str.contains(product.split()[0], na=False)]
        similar_html = html.Ul([html.Li(f"{row['Description']} - ${row['UnitPrice']:.2f}") 
                                for _, row in similar_products.head(5).iterrows()])

        # Display total value
        total_value_html = html.P(f"Total Value: ${total_cost:.2f}", className="lead text-center")

        prediction_html = html.P(
            f"Predicted Cluster: {predicted_cluster}", className="lead text-center"
        )

        # RFM Metrics
        rfm_metrics_html = html.P(
            f"Recency: {recency}, Frequency: {frequency}, Monetary: ${monetary:.2f}",
            className="lead text-center"
        )

        return cluster_plot, prediction_html, similar_html, total_value_html, rfm_metrics_html

    # Default outputs in case no action is taken
    return generate_cluster_plot(), "", "", "", ""

# Run the app
if __name__ == "__main__":
    app.run_server(debug=True)
