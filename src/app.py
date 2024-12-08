import dash
from dash import html, dcc, Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import numpy as np

# Load the dataset
data_path = "exported_data.parquet"  # Update this path if needed
df = pd.read_parquet(data_path)
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

# Prepare data for clustering (This is for the background cluster visualization)
# We assume the dataset contains columns log_Recency, log_Frequency, log_MonetaryValue precomputed.
rfm_data = df.groupby("CustomerID").agg({
    "log_Recency": "first",
    "log_Frequency": "first",
    "log_MonetaryValue": "first"
}).copy()

# Scale data for clustering
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm_data)
kmeans = KMeans(n_clusters=4, random_state=42, n_init='auto')
rfm_data["Cluster"] = kmeans.fit_predict(rfm_scaled)

# Define cluster names and descriptions
cluster_names = {
    0: "High-Value Frequent Buyer",
    1: "Moderate-Value Regular Customer",
    2: "Low-Value Infrequent Buyer",
    3: "Occasional Moderate-Spend Customer"
}

cluster_descriptions = {
    0: "Highly engaged and valuable customers who buy frequently and spend a lot recently.",
    1: "Moderately active customers with average frequency and monetary values, representing a balanced middle group.",
    2: "Customers who have not spent much recently, buy infrequently, and may need re-engagement.",
    3: "Occasional buyers who spend moderately but are less recent or frequent than the most engaged segments."
}

all_products = df["Description"].dropna().unique()
product_options = [{"label": p, "value": p} for p in sorted(all_products)]

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

def generate_cluster_plot(highlight_point=None):
    """Generate a 3D cluster plot using the pre-calculated cluster data.
       If highlight_point is provided (RFM of the user), plot it as a red marker."""
    fig = px.scatter_3d(
        rfm_data,
        x="log_Recency",
        y="log_Frequency",
        z="log_MonetaryValue",
        color="Cluster",
        labels={
            "log_Recency": "Log Recency",
            "log_Frequency": "Log Frequency",
            "log_MonetaryValue": "Log Monetary Value",
        },
        hover_data=["Cluster"],
        template="plotly_dark"
    )
    if highlight_point is not None:
        fig.add_scatter3d(
            x=[highlight_point[0]],
            y=[highlight_point[1]],
            z=[highlight_point[2]],
            mode="markers",
            marker=dict(size=10, color="red", symbol="circle"),
            name="Your RFM"
        )
    return fig

def calculate_rfm_for_inputs(records):
    """Calculate RFM values (log transformed) from the user-entered items.
       R: Based on the most recent purchase date
       F: Based on total quantity of all items
       M: Based on total monetary value (sum of quantity * unit_price)"""
    if not records:
        return None, "No items entered."

    # Filter out rows without valid UnitPrice
    df_clean = df.dropna(subset=["UnitPrice"])

    total_quantity = 0
    total_value = 0
    all_dates = []

    for r in records:
        product = r["product"]
        quantity = r["quantity"]
        date_str = r["date"]
        if not product or not date_str or quantity <= 0:
            return None, "Invalid input records."

        chosen_date = pd.to_datetime(date_str)
        all_dates.append(chosen_date)
        
        product_info = df_clean[df_clean["Description"] == product]
        if product_info.empty:
            return None, f"No valid price data for {product}"
        unit_price = product_info["UnitPrice"].mean()

        total_quantity += quantity
        total_value += (unit_price * quantity)
        
    if total_value <= 0:
        return None, "Total value is zero or negative."
    if total_quantity <= 0:
        return None, "Total quantity is zero or negative."

    today = pd.to_datetime(datetime.today().date())
    recency_days = (today - max(all_dates)).days + 1

    # Log transform the R, F, M values
    log_recency = np.log(recency_days + 1)
    log_frequency = np.log(total_quantity + 1)
    log_monetary = np.log(total_value + 1)

    if np.isnan(log_recency) or np.isnan(log_frequency) or np.isnan(log_monetary):
        return None, "Invalid RFM values."

    return (log_recency, log_frequency, log_monetary), None

def recommend_products_based_on_cooccurrence(selected_product):
    """Recommend products based on co-occurrence.
       We find customers who purchased the selected product,
       then find other products they also purchased, and rank by frequency."""
    cust_with_product = df[df["Description"] == selected_product]["CustomerID"].unique()
    if len(cust_with_product) == 0:
        return []
    
    cust_purchases = df[df["CustomerID"].isin(cust_with_product)]
    product_counts = cust_purchases["Description"].value_counts()
    product_counts = product_counts.drop(selected_product, errors='ignore')
    top_products = product_counts.head(5)
    recs = top_products.reset_index()
    recs.columns = ["Description", "Count"]
    return recs.values.tolist()

app.layout = html.Div(
    style={"backgroundColor": "#121212", "padding": "60px", "paddingBottom":"80px"},
    children=[
        # Store for user input items
        dcc.Store(id="items-store", storage_type="memory"),
        # Store for user's RFM data and cluster
        dcc.Store(id="rfm-store", storage_type="memory"),

        dbc.Container(
            [
                dbc.Row(
                    dbc.Col(
                        html.H1("Customer Insights and Recommendations",
                                className="text-center",
                                style={"color": "#00BCD4", "marginBottom":"75px"}  
                               ), 
                    width=12)
                ),

                dbc.Row(
                    [
                        # First Column: User input
                        dbc.Col(
                            [
                                html.H5("Add Products", style={"color": "#00BCD4", "marginBottom":"20px"}),
                                html.Div(
                                    [
                                        # Product selection
                                        dcc.Dropdown(
                                            id="product-dropdown",
                                            options=product_options,
                                            placeholder="Select a product",
                                            style={"color": "black", "marginBottom":"20px"}
                                        ),

                                        # Quantity input
                                        dbc.Label("Quantity", style={"color": "white"}),
                                        dbc.Input(id="quantity-input", type="number", placeholder="Enter quantity", 
                                                  className="mb-3"),

                                        # Date input
                                        dbc.Label("Date", style={"color": "white", "marginBottom":"10px"}),
                                        dcc.DatePickerSingle(
                                            id="date-picker",
                                            placeholder="Select a date",
                                            display_format='YYYY-MM-DD',
                                            style={"color":"black", "marginBottom":"20px", "marginTop":"5px"}
                                        ),

                                        # Add Item button aligned right
                                        html.Div(
                                            dbc.Button("Add Item", id="add-item-button", color="primary"),
                                            style={"textAlign":"right","marginBottom":"20px"}
                                        )
                                    ],
                                    style={"border":"1px solid #00BCD4", "padding":"15px", "borderRadius":"5px", "marginBottom":"30px"}
                                ),

                                html.H5("Entered Items", style={"color": "#00BCD4", "marginBottom":"20px"}),
                                html.Div(id="items-table",
                                         style={"backgroundColor": "#1E1E1E","color": "white",
                                                "border":"1px solid #00BCD4","border-radius": "5px", 
                                                "padding":"10px", "marginBottom":"30px"}),

                                # Button to calculate RFM and get recommendations
                                dbc.Button("Calculate RFM & Recommendations", id="calculate-button", color="secondary", className="mb-3"),
                                
                                html.Div(id="results-div", 
                                         style={"backgroundColor": "#1E1E1E","color": "white",
                                                "border":"1px solid #00BCD4","border-radius": "5px", 
                                                "padding":"10px", "marginTop":"20px"})
                            ],
                            width=4
                        ),

                        # Second & Third Columns: 3D cluster visualization
                        dbc.Col(
                            [
                                html.H5("3D Cluster Visualization", className="text-center", 
                                        style={"color": "#00BCD4", "marginBottom":"40px"}),
                                dcc.Graph(id="cluster-graph", 
                                          style={"height": "600px", "width": "100%", "marginBottom":"60px"})
                            ],
                            width=8
                        )
                    ],
                    className="mb-4",
                    align="start"
                ),
            ],
            fluid=True,
        ),
    ]
)

@app.callback(
    [
        Output("items-store", "data"),
        Output("product-dropdown", "value"),
        Output("quantity-input", "value"),
        Output("date-picker", "date")
    ],
    Input("add-item-button", "n_clicks"),
    State("product-dropdown", "value"),
    State("quantity-input", "value"),
    State("date-picker", "date"),
    State("items-store", "data")
)
def add_item(n_clicks, product, quantity, date, current_data):
    """Add an item to the user's list. After adding, clear the input fields."""
    if current_data is None:
        current_data = []
    if n_clicks and product and quantity and date:
        current_data.append({"product": product, "quantity": quantity, "date": date})
    # Clear inputs after adding item
    return current_data, None, None, None

@app.callback(
    Output("items-table", "children"),
    Input("items-store", "data")
)
def show_items_table(data):
    """Display the list of entered items with product, quantity, date, and price."""
    if not data:
        return html.P("No items added yet.", className="text-center")
    df_clean = df.dropna(subset=["UnitPrice"])
    rows = []
    for i, item in enumerate(data, start=1):
        product_info = df_clean[df_clean["Description"] == item["product"]]
        if not product_info.empty:
            unit_price = product_info["UnitPrice"].mean()
        else:
            unit_price = 0.0

        # محاسبه قیمت کل این ردیف
        total_item_cost = unit_price * item["quantity"]

        rows.append(html.Tr([
            html.Td(i),
            html.Td(item["product"]),
            html.Td(item["quantity"]),
            html.Td(item["date"]),
            html.Td(f"${unit_price:.2f}"),              # Unit price
            html.Td(f"${total_item_cost:.2f}")          # Total price
        ]))
    table = dbc.Table(
        [html.Thead(html.Tr([html.Th("#"), html.Th("Product"), html.Th("Quantity"), html.Th("Date"), html.Th("Price"), html.Th("Total") ])),
         html.Tbody(rows)],
        bordered=True, dark=True, hover=True, responsive=True, striped=True
    )
    return table

@app.callback(
    [Output("results-div", "children"),
     Output("rfm-store", "data")],
    Input("calculate-button", "n_clicks"),
    State("items-store", "data")
)
def calculate_and_recommend(n_clicks, data):
    """When the user clicks 'Calculate', compute RFM, predict cluster, and provide recommendations."""
    if n_clicks and data:
        rfm_values, error_msg = calculate_rfm_for_inputs(data)
        if error_msg is not None:
            return html.P(error_msg, className="lead"), None

        log_recency, log_frequency, log_monetary = rfm_values
        new_point = pd.DataFrame([[log_recency, log_frequency, log_monetary]],
                                 columns=["log_Recency", "log_Frequency", "log_MonetaryValue"])

        # Scale the user's RFM point using the existing scaler
        new_point_scaled = scaler.transform(new_point)
        # Predict cluster using the existing kmeans model
        predicted_cluster = kmeans.predict(new_point_scaled)[0]
        cluster_name = cluster_names.get(predicted_cluster, "Unknown")
        cluster_desc = cluster_descriptions.get(predicted_cluster, "")

        # Recommend products based on the last product added
        last_product = data[-1]["product"]
        recommendations = recommend_products_based_on_cooccurrence(last_product)

        if len(recommendations) == 0:
            recs_html = html.P("No similar products found.", className="lead text-center")
        else:
            recs_html = html.Ul(
                [html.Li(f"{desc} (count: {count})") for desc, count in recommendations],
                style={"list-style-position": "inside"}
            )

        results_content = html.Div([
            html.P("Your RFM:", className="lead"),
            html.P(f"Recency (log): {log_recency:.2f}, Frequency (log): {log_frequency:.2f}, Monetary (log): {log_monetary:.2f}", className="lead"),
            html.Hr(),
            html.P(f"Predicted Cluster: {cluster_name}", className="lead"),
            html.P(f"Description: {cluster_desc}", className="lead"),
            html.Hr(),
            html.H5("Recommended Products:", style={"color":"#00BCD4"}),
            recs_html
        ])

        # Store user's RFM for highlighting in the plot
        rfm_store_data = {
            "log_recency": log_recency,
            "log_frequency": log_frequency,
            "log_monetary": log_monetary
        }

        return results_content, rfm_store_data
    return "", None

@app.callback(
    Output("cluster-graph", "figure"),
    [Input("items-store", "data"),
     Input("rfm-store", "data")]
)
def update_graph(_, rfm_data_user):
    """Update the 3D cluster graph with the user's RFM point highlighted if available."""
    highlight_point = None
    if rfm_data_user:
        highlight_point = (rfm_data_user["log_recency"], rfm_data_user["log_frequency"], rfm_data_user["log_monetary"])
    return generate_cluster_plot(highlight_point=highlight_point)

if __name__ == "__main__":
    app.run_server(debug=True)
