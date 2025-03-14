import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import networkx as nx
from bokeh.models import Circle, MultiLine, Plot, Range1d, HoverTool
from bokeh.models import TapTool, NodesAndLinkedEdges, EdgesAndLinkedNodes
from bokeh.models import GraphRenderer, StaticLayoutProvider, Rect, LabelSet
from bokeh.models.graphs import from_networkx, EdgesAndLinkedNodes, NodesAndLinkedEdges
from bokeh.plotting import figure, show
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from bokeh.palettes import Blues8
from bokeh.io import output_file

st.set_page_config(layout="wide") 

st.title("PCA & Darknet Network Visualization")
col1, col2 = st.columns([0.3, 0.7])  

with col1:  
    st.markdown("""
        **Dataset:** [Darknet.csv](https://www.kaggle.com/datasets/peterfriedrich1/cicdarknet2020-internet-traffic/data)  
        **Author:** Sana Fessuh
    """, unsafe_allow_html=True)
st.sidebar.header("PCA Visualization Settings")

df = pd.read_csv("Darknet.csv", delimiter=',', on_bad_lines='skip')
df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%d/%m/%Y %I:%M:%S %p', errors='coerce')
df['Date'] = df['Timestamp'].dt.date
main_hubs = ["Non-Tor", "NonVPN", "VPN", "Tor"]

df_filtered = df[df["Label"].isin(main_hubs)]
filtered_df_2 = df_filtered[["Label", "Label.1"]]

airline_graph = nx.from_pandas_edgelist(filtered_df_2, source="Label", target="Label.1")
pos = nx.spring_layout(airline_graph, scale=10, k=3)
edge_highlight_color = "yellow"
node_attributes = {}
for node in airline_graph.nodes():
    if node in main_hubs:
        node_attributes[node] = {'color': '#F7C5CC', 'size': 50}  
    else:
        node_attributes[node] = {'color': '#538bc2', 'size': 30}  

nx.set_node_attributes(airline_graph, node_attributes)

plot = figure(
    tools="pan,wheel_zoom,save,reset,tap",
    active_scroll='wheel_zoom',
    x_range=Range1d(-12, 12),
    y_range=Range1d(-12, 12),
    title="Darknet Network",
    background_fill_color="#5e5b75"
)

network_graph = from_networkx(airline_graph, pos, scale=10, center=(0, 0))
network_graph.node_renderer.glyph = Circle(size="size", fill_color="color")
network_graph.node_renderer.selection_glyph = Circle(size="size", fill_color="color")
network_graph.node_renderer.hover_glyph = Circle(size="size", fill_color="color")

network_graph.edge_renderer.glyph = MultiLine(line_alpha=0.5, line_width=2)
network_graph.edge_renderer.selection_glyph = MultiLine(line_color=edge_highlight_color, line_width=2)
network_graph.edge_renderer.hover_glyph = MultiLine(line_color=edge_highlight_color, line_width=2)
network_graph.selection_policy = NodesAndLinkedEdges()
network_graph.inspection_policy = NodesAndLinkedEdges()

plot.renderers.append(network_graph)


hover_tool = HoverTool(tooltips=[("Node", "@index")])
plot.add_tools(hover_tool)
st.bokeh_chart(plot, use_container_width=True)
### ----------- PCA Analysis -----------
st.markdown("<br><br><br><br><br>", unsafe_allow_html=True) 
st.subheader("PCA Visualization")

X = df_filtered.select_dtypes(include=['number'])
X.replace([np.inf, -np.inf], np.nan, inplace=True)
X.dropna(inplace=True)
X = X.clip(lower=-1e6, upper=1e6)


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
pca = PCA(n_components=27)
components = pca.fit_transform(X_scaled)
total_var = np.cumsum(pca.explained_variance_ratio_)[-1] * 100  

df_pca = pd.DataFrame(components, columns=[f'PC{i+1}' for i in range(27)])
df_pca['Label'] = df_filtered.iloc[X.index]['Label'].values

label_colors = {
    'Non-Tor': '#eaac8b',
    'NonVPN': '#6d597a',
    'VPN': '#b56576',
    'Tor': '#e56b6f'
}
df_pca['Color'] = df_pca['Label'].map(label_colors)
component_options = [f'PC{i+1}-PC{i+2}-PC{i+3}' for i in range(0, 27, 3) if i+2 < 27]
selected_components = st.sidebar.selectbox("Select Principal Components:", component_options)
selected_indices = [int(pc[2:]) - 1 for pc in selected_components.split('-')]
selected_pc1, selected_pc2, selected_pc3 = selected_indices
selected_variance = np.sum(pca.explained_variance_ratio_[selected_indices]) * 100


x_min, x_max = st.sidebar.slider(f"PC{selected_pc1+1} Range", 
                                 float(df_pca.iloc[:, selected_pc1].min()), 
                                 float(df_pca.iloc[:, selected_pc1].max()), 
                                 (float(df_pca.iloc[:, selected_pc1].min()), 
                                  float(df_pca.iloc[:, selected_pc1].max())))

y_min, y_max = st.sidebar.slider(f"PC{selected_pc2+1} Range", 
                                 float(df_pca.iloc[:, selected_pc2].min()), 
                                 float(df_pca.iloc[:, selected_pc2].max()), 
                                 (float(df_pca.iloc[:, selected_pc2].min()), 
                                  float(df_pca.iloc[:, selected_pc2].max())))

scatter3d = go.Figure()
for label, color in label_colors.items():
    subset = df_pca[df_pca['Label'] == label]
    scatter3d.add_trace(go.Scatter3d(
        x=subset.iloc[:, selected_pc1], 
        y=subset.iloc[:, selected_pc2], 
        z=subset.iloc[:, selected_pc3], 
        mode='markers',
        marker=dict(size=5, opacity=0.8, color=color),
        name=label
    ))

scatter3d.update_layout(
    title=f'3D PCA Scatter ({selected_components}) - Selected: {selected_variance:.2f}% | Total: {total_var:.2f}%',
    height=400,
    width=1400
)
scatter2d = go.Figure()
for label, color in label_colors.items():
    subset = df_pca[df_pca['Label'] == label]
    scatter2d.add_trace(go.Scatter(
        x=subset.iloc[:, selected_pc1], 
        y=subset.iloc[:, selected_pc2], 
        mode='markers', 
        marker=dict(size=5, color=color),
        name=label
    ))

scatter2d.update_layout(
    title=f"{selected_components} - 2D PCA Plot",
    xaxis=dict(title=f"PC{selected_pc1+1}", range=[x_min, x_max]),
    yaxis=dict(title=f"PC{selected_pc2+1}", range=[y_min, y_max]),
    height=500,
    width=800
)

st.plotly_chart(scatter3d, use_container_width=True)
st.plotly_chart(scatter2d, use_container_width=True)
