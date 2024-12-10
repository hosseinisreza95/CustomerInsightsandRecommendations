# Customer Insights and Recommendations

![Dashboard Screenshot](https://github.com/hosseinisreza95/CustomerInsightsandRecommendations/blob/main/screenshot/pic1.png)

## Overview

This project implements a customer insights and recommendations system using RFM (Recency, Frequency, Monetary) analysis. The application allows users to analyze customer behaviors, visualize customer segmentation using clustering, and provide product recommendations based on purchasing patterns.

The web application is built using Python, Dash, and Plotly, and it employs machine learning techniques like K-Means clustering for customer segmentation.

---

## Features

- **Customer Segmentation**: Analyze customer behavior and group them into clusters using RFM values.
- **Interactive 3D Visualization**: Explore customer clusters in a 3D scatter plot.
- **Product Recommendations**: Suggest products based on customer purchase history and co-occurrence analysis.
- **User Input Handling**: Add items to simulate purchases and calculate RFM metrics for real-time insights.

---

## Technologies Used

- **Python**: Core programming language.
- **Dash**: Framework for building interactive web applications.
- **Plotly**: Used for 3D visualization.
- **scikit-learn**: For K-Means clustering and data preprocessing.
- **Pandas**: Data manipulation and analysis.
- **Dash Bootstrap Components**: For responsive and styled UI.

---

## Application Links

- **Live Application**: [Customer Insights App](https://customerinsightsandrecommendations.onrender.com/)

---

## How It Works

### Data Processing

- The dataset is loaded from a parquet file and preprocessed.
- RFM metrics (`Recency`, `Frequency`, `Monetary`) are calculated and log-transformed for better scaling.
- Customers are segmented using **K-Means clustering**.

### User Interaction

1. **Add Products**: 
   - Users can input purchased products, quantities, and dates.
   - The system validates inputs and calculates RFM metrics for the provided data.

2. **3D Cluster Visualization**:
   - The application displays pre-computed customer clusters.
   - User's RFM values are highlighted in the plot for easy comparison.

3. **Product Recommendations**:
   - Based on the last product added, the app recommends top co-occurring products.

---

## Installation and Setup

### Prerequisites

### Prerequisites

- Python 3.8 or above
- Required libraries listed in `requirements.txt`

### Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/hosseinisreza95/CustomerInsightsandRecommendations.git
   cd CustomerInsightsandRecommendations


## File Structure

- `src/app.py`: Main application script.
- `exported_data.parquet`: Example dataset used for customer insights.
- `requirements.txt`: List of Python dependencies.

---

## Example Usage

1. **Launch the Application**:
   Open the app and interact with the input fields.

2. **Add Items**:
   Enter product details, quantity, and purchase date.

3. **Analyze Insights**:
   Calculate RFM values and view cluster assignments.

4. **Get Recommendations**:
   Receive tailored product recommendations.

---

## Screenshots

### 3D Cluster Visualization
![3D Cluster]([https://your-image-url-placeholder.com](https://github.com/hosseinisreza95/CustomerInsightsandRecommendations/blob/main/screenshot/pic2.png))

### User Inputs and Recommendations
![Inputs and Recommendations]([https://your-image-url-placeholder.com](https://github.com/hosseinisreza95/CustomerInsightsandRecommendations/blob/main/screenshot/pic3.png))

---

## Acknowledgements

This project is part of a learning and research initiative. Special thanks to all contributors and supporters.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

---

## Contact

For queries, suggestions, or feedback, please reach out to [Reza Hosseini](https://github.com/hosseinisreza95).


