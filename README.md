# Visualizing Darknet Traffic & PCA Analysis Using Streamlit

## Introduction 
For my project, I chose to analyze the darknet and explore the connection between privacy tools—such as VPNs and Tor—and how they are utilized. The darknet is commonly associated with anonymity, but I wanted to take a closer look and examine how these privacy tools function in practice. My goal was to investigate whether their usage is tied to specific types of activity and to gain a deeper understanding of the patterns and behaviors surrounding them.

**Dataset Used:**
https://www.kaggle.com/datasets/peterfriedrich1/cicdarknet2020-internet-traffic/data

## Darknet Network visualization
The first visualization represents a network graph of darknet activity, where nodes represent different categories of users **(e.g., Non-Tor, NonVPN, VPN, Tor)**, and edges show their connections. This graph helps illustrate the structure of darknet traffic and how different privacy tools interact within the network. By applying network visualization techniques using **networkx and bokeh**, I was able to highlight the main hubs of activity and their relationships.
### Why this visual
- Represent how darknet traffic flows between different privacy tools.
- Helps understand the relationship between the privacy tools used and the types of traffic or activities they facilitate.
    - Certain tools are associated with specific usage patterns or behaviors.
##  PCA Analysis of Darknet Traffic
The second visualization applies Principal Component Analysis (PCA) to the darknet dataset. PCA helps reduce the dimensionality of the data while retaining its most important patterns. By identifying the optimal principal components, I was able to visualize how different privacy tools—such as VPNs and  group together based on traffic characteristics. The interactive 2D and 3D scatter plots allow for an exploration of trends and variations in the data.
### Why this visual
- Reduces high-dimensional darknet traffic data into a more interpretable form while preserving key patterns.
- Identifies how privacy tools cluster based on shared traffic attributes, revealing potential similarities or outliers in usage.


## Critical Analysis
One of the initial limitations I encountered was my intention to conduct a time series analysis of darknet traffic. However, after examining the dataset, I realized that the temporal coverage was inconsistent. While data was available for 2015 and 2016, it was not evenly distributed across months. The dataset contained only a few days of data per month, making it difficult to establish meaningful trends over time.This lack of consistent and continuous timestamps made it impractical to apply traditional time-series methods, as there were large gaps and irregularities in data collection. Because of this, I had to shift my focus away from a temporal trend analysis and instead prioritize network and PCA-based approaches that could extract insights from static snapshots of darknet activity. 
### Potential Improvements & Future Directions
- **Expanding the Dataset** – If more complete time-series data becomes available, applying longitudinal analysis could help track shifts in darknet activity over time.
- **Other Dimensionality Reduction approaches** – Exploring alternatives like t-SNE (t-Distributed Stochastic Neighbor Embedding) or UMAP (Uniform Manifold Approximation and Projection) could better capture complex relationships in the data.
- **Equal Weighting of Privacy Tools & Activities** – All privacy tools (VPN, Tor, Non-VPN, Non-Tor) were treated as having equal importance, even though in reality, their usage frequency and purpose may differ significantly.
    - May oversimplify darknet behavior, as some privacy tools may be more closely associated with specific types of activity than others.
## Files
### **`visual.py`**
- Handles the **visualization of darknet network traffic** using **Streamlit, Bokeh, and Plotly**.  
- Generates **interactive network graphs and PCA-based plots** to analyze privacy tool usage patterns.

### **`exploration.ipynb`**
- Contains the **initial data exploration and analysis** of the darknet dataset.
    - Find optimal components for PCA
    - Initial visual of PCA analysis used in streamlit

### **`Darknet.csv`**
Dataset used for analysis(Found on kaggle link)

## Installation
This project runs on Streamlit, a Python framework for interactive web apps. To install the necessary dependencies, you can refer to the requirements.txt file and install them with:
```bash
pip install -r requirements.txt
```
If you already have some packages installed, you can manually install missing ones as needed. Once everything is set up, run the application with:
```bash
streamlit run visual.py
```
