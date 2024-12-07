import dash
from dash import html, dcc, Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import numpy as np

data_path = "exported_data.parquet"  # Replace with your dataset path
df = pd.read_parquet(data_path)
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
df["CustomerID"] = df["CustomerID"].astype(int)

# Prepare RFM data
rfm_data = df.groupby("CustomerID").agg({
    "log_Recency": "first",
    "log_Frequency": "first",
    "log_MonetaryValue": "first"
}).copy()

# Scale data for clustering
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm_data)
kmeans = KMeans(n_clusters=4, random_state=42)
rfm_data["Cluster"] = kmeans.fit_predict(rfm_scaled)

# Cluster info
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

def generate_cluster_plot(highlight_customer=None):
    fig = px.scatter_3d(
        rfm_data,
        x="log_Recency",
        y="log_Frequency",
        z="log_MonetaryValue",
        color="Cluster",
        title="",
        labels={
            "log_Recency": "Log Recency",
            "log_Frequency": "Log Frequency",
            "log_MonetaryValue": "Log Monetary Value",
        },
        hover_data=["Cluster"],
        template="plotly_dark"
    )
    if highlight_customer is not None and highlight_customer in rfm_data.index:
        c_data = rfm_data.loc[highlight_customer]
        fig.add_scatter3d(
            x=[c_data["log_Recency"]],
            y=[c_data["log_Frequency"]],
            z=[c_data["log_MonetaryValue"]],
            mode="markers",
            marker=dict(size=10, color="red", symbol="circle"),
            name="Selected Customer"
        )
    return fig

def calculate_rfm(selected_date_str, quantity, unit_price):
    if selected_date_str is None:
        return None, "Please select a date."
    
    chosen_date = pd.to_datetime(selected_date_str)
    today = pd.to_datetime(datetime.today().date())
    if chosen_date > today:
        return None, "Selected date is in the future. Please select a past or current date."

    recency_days = (today - chosen_date).days + 1
    if quantity <= 0:
        return None, "Quantity must be positive."
    if pd.isna(unit_price):
        return None, "Invalid price data for selected product."

    monetary_value = unit_price * quantity
    if monetary_value <= 0:
        return None, "Calculated monetary value is zero or negative, cannot predict."

    log_recency = np.log(recency_days+1)
    log_frequency = np.log(quantity+1)
    log_monetary = np.log(monetary_value+1)

    if np.isnan(log_recency) or np.isnan(log_frequency) or np.isnan(log_monetary):
        return None, "Invalid RFM values (NaN encountered), cannot predict cluster."

    return (log_recency, log_frequency, log_monetary), None

app.layout = html.Div(
    style={"backgroundColor": "#121212", "padding": "60px", "paddingBottom":"80px"},  
    children=[
        dbc.Container(
            [
                dbc.Row(
                    dbc.Col(
                        html.H1("Customer Insights and Recommendations",
                                className="text-center",
                                style={"color": "#00BCD4", "marginBottom":"100px"}  # Increase bottom margin for more space
                               ), 
                    width=12)
                ),

                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.H5("Select Customer", style={"color": "#00BCD4", "marginBottom":"30px"}),
                                dcc.Dropdown(
                                    id="customer-dropdown",
                                    options=[{"label": cid, "value": cid} for cid in df["CustomerID"].unique()],
                                    placeholder="Select a Customer ID",
                                    style={"color": "black", "marginBottom":"60px"}
                                ),

                                html.H5("Customer Profile", style={"color": "#00BCD4", "marginBottom":"20px"}),
                                html.Div(id="customer-profile", className="border p-3 mb-5",
                                         style={"backgroundColor": "#1E1E1E", "color": "white", "border-radius": "5px"}),

                                html.H5("Customer Purchase History", style={"color": "#00BCD4", "marginBottom":"20px"}),
                                html.Div(id="customer-history", className="border p-3",
                                         style={"backgroundColor": "#1E1E1E", "color": "white", 
                                                "border-radius": "5px", "maxHeight": "300px", 
                                                "overflowY": "scroll", "marginBottom":"40px"})
                            ],
                            width=4
                        ),

                        dbc.Col(
                            [
                                html.H5("3D Cluster Visualization", className="text-center", 
                                        style={"color": "#00BCD4", "marginBottom":"40px"}),
                                dcc.Graph(id="cluster-graph", 
                                          style={"height": "600px", "width": "100%", "marginBottom":"60px"})
                            ],
                            width=4
                        ),

                        dbc.Col(
                            [
                                html.H5("Select a Product", style={"color": "#00BCD4", "marginBottom":"20px"}),
                                dcc.Dropdown(
                                    id="past-purchase-dropdown",
                                    options=product_options,
                                    placeholder="Select a product",
                                    style={"color": "black", "marginBottom":"40px"}
                                ),
                                dbc.Label("Enter Quantity", style={"color": "white", "marginBottom":"10px"}),
                                dbc.Input(id="quantity-input", type="number", placeholder="Enter quantity", 
                                          className="mb-4"),
                                dbc.Label("Select Date", style={"color": "white", "marginTop":"10px", "marginBottom":"10px"}),
                                dcc.DatePickerSingle(
                                    id="date-picker",
                                    placeholder="Select a date",
                                    display_format='YYYY-MM-DD',
                                    style={"color":"black", "marginBottom":"40px", "marginTop":"5px"}
                                ),
                                # Center the submit button and add spacing
                                html.Div(
                                    dbc.Button("Submit", id="submit-button", color="primary"),
                                    style={"textAlign":"center", "marginBottom":"60px"}
                                ),

                                html.H5("Recommended Products", style={"color": "#00BCD4", "marginBottom":"20px"}),
                                html.Div(id="recommended-products", className="border p-3",
                                         style={"backgroundColor": "#1E1E1E", "color": "white", 
                                                "border-radius": "5px", "marginBottom":"40px"})
                            ],
                            width=4
                        ),
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
    [Output("cluster-graph", "figure"),
     Output("customer-history", "children"),
     Output("customer-profile", "children")],
    Input("customer-dropdown", "value")
)
def update_customer_info(customer_id):
    if customer_id and customer_id in rfm_data.index:
        c_rfm = rfm_data.loc[customer_id]

        recency = c_rfm["log_Recency"]
        frequency = c_rfm["log_Frequency"]
        monetary = c_rfm["log_MonetaryValue"]

        customer_purchases = df[df["CustomerID"] == customer_id].sort_values("InvoiceDate")
        total_value = (customer_purchases["Quantity"] * customer_purchases["UnitPrice"]).sum()
        purchase_count = customer_purchases.shape[0]

        purchase_list = []
        for _, row in customer_purchases.iterrows():
            purchase_list.append(
                html.Li(
                    f"{row['InvoiceDate'].date()} - {row['Description']} - Qty: {row['Quantity']} - Price: ${row['UnitPrice']:.2f}"
                )
            )
        history_html = html.Ul(purchase_list, style={"list-style-position": "inside"})

        profile_content = html.Div(
            [
                html.P(f"Recency: {recency:.2f}, Frequency: {frequency:.2f}, Monetary Value: {monetary:.2f}", className="lead"),
                html.P(f"Total Value: ${total_value:.2f}, Purchase Count: {purchase_count}", className="lead"),
            ],
            style={"lineHeight": "1.8"}
        )

        cluster_plot = generate_cluster_plot(highlight_customer=customer_id)
        return cluster_plot, history_html, profile_content

    return generate_cluster_plot(), "", ""

@app.callback(
    Output("recommended-products", "children"),
    Input("submit-button", "n_clicks"),
    State("past-purchase-dropdown", "value"),
    State("quantity-input", "value"),
    State("date-picker", "date"),
    State("customer-dropdown", "value")
)
def recommend_and_predict(n_clicks, selected_product, quantity, selected_date, customer_id):
    if n_clicks and n_clicks > 0 and selected_product and quantity and selected_date and customer_id in rfm_data.index:
        product_info = df[df["Description"] == selected_product]
        if product_info.empty:
            return html.P("No data available for selected product", className="lead text-center")

        valid_prices = product_info["UnitPrice"].dropna()
        if valid_prices.empty:
            return html.P("No valid price data for selected product", className="lead text-center")
        unit_price = valid_prices.iloc[0]

        rfm_values, error_msg = calculate_rfm(selected_date, quantity, unit_price)
        if error_msg is not None:
            return html.P(error_msg, className="lead text-center")

        log_recency, log_frequency, log_monetary = rfm_values

        new_point = pd.DataFrame([[log_recency, log_frequency, log_monetary]],
                                 columns=["log_Recency", "log_Frequency", "log_MonetaryValue"])
        if new_point.isna().any().any():
            return html.P("NaN detected in input data, cannot predict.", className="lead text-center")

        new_point_scaled = scaler.transform(new_point)
        if np.isnan(new_point_scaled).any():
            return html.P("NaN detected after scaling, cannot predict.", className="lead text-center")

        predicted_cluster = kmeans.predict(new_point_scaled)[0]
        cluster_name = cluster_names.get(predicted_cluster, "Unknown")
        cluster_desc = cluster_descriptions.get(predicted_cluster, "")

        # Recommendation logic
        keyword = selected_product.split()[0]
        similar_products = df[df["Description"].str.contains(keyword, case=False, na=False)]
        similar_products = similar_products[similar_products["Description"] != selected_product]

        if similar_products.empty:
            return html.Div([
                html.P(f"Predicted Cluster: {cluster_name}", className="lead"),
                html.P(f"Description: {cluster_desc}", className="lead"),
                html.P("No similar products found.", className="lead text-center")
            ])

        similar_counts = similar_products["Description"].value_counts().reset_index()
        similar_counts.columns = ["Description", "Count"]
        top_similar = similar_counts.head(5)

        if top_similar.empty:
            recs_html = html.P("No similar products found.", className="lead text-center")
        else:
            recs_html = html.Ul(
                [html.Li(f"{row['Description']} (purchased {row['Count']} times)") for _, row in top_similar.iterrows()],
                style={"list-style-position": "inside"}
            )

        return html.Div([
            html.P(f"Predicted Cluster: {cluster_name}", className="lead"),
            html.P(f"Description: {cluster_desc}", className="lead"),
            recs_html
        ])

    return ""

if __name__ == "__main__":
    app.run_server(debug=True)
