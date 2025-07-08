import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# --- Page Configuration ---
st.set_page_config(
    layout="wide",
    page_title="Model Performance"
)

# # --- Sidebar ---
# # This adds the global cancer type selector to this page as well
# st.sidebar.header("Global Controls")
# cancer_type = st.sidebar.selectbox(
#     "Select a Cancer Type:",
#     ("Brain", "Breast", "Lung"),
#     key="cancer_type_performance"
# )

# --- Header ---
st.title("🔬 Model Performance")
st.markdown("This page details the performance of the fine-tuned DNABERT models used for predicting regulatory elements. High performance on these classification tasks is essential for accurate downstream variant effect prediction.")
st.divider()

# --- Data Loading and Processing ---
@st.cache_data
def load_and_process_accuracy_data():
    """Loads and prepares the TFBS accuracy statistics file."""
    try:
        # Load the data, which includes the 'Type' column
        df = pd.read_csv("data/300bp_TFBS_accuracy_Stat.csv", sep=",")
        
        # Rename columns for prettier labels in the plots
        rename_dict = {
            'eval_acc': 'Accuracy',
            'eval_precision': 'Precision',
            'eval_recall': 'Recall',
            'eval_f1': 'F1-score',
            'eval_mcc': 'MCC',
            'eval_auc': 'ROC-AUC'
        }
        df.rename(columns=rename_dict, inplace=True)
        
        return df
    except FileNotFoundError:
        st.error("Error: `data/300bp_TFBS_accuracy_Stat.tsv` not found. Please ensure the file is in the correct directory.")
        return None

df_accuracy = load_and_process_accuracy_data()

if df_accuracy is not None:
    # --- Section 1: Detailed Performance Metrics Boxplot (Recreating Figure 2a) ---
    st.header("Performance Metrics by Model Type")
    st.markdown("The boxplots below compare key evaluation metrics across Transcription Factor (TF), RNA-Binding Protein (RBP), and Histone Mark models, demonstrating consistently high performance.")
    st.dataframe(df_accuracy)
    # Melt the dataframe to plot all metrics in one chart
    metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1-score', 'MCC', 'ROC-AUC']
    df_melted = df_accuracy.melt(
        id_vars=['tags', 'Type'], 
        value_vars=metrics_to_plot,
        var_name='Metric', 
        value_name='Value'
    )
    
    # The 'Type' column now correctly contains 'TF', 'TF; RBP', and 'Histone'
    fig_box = px.box(
        df_melted, 
        x='Metric', 
        y='Value', 
        color='Type',  # Use the 'Type' column directly from the file
        category_orders={"Metric": metrics_to_plot}, # Ensure consistent x-axis order
        color_discrete_map={
            'TF': 'rgba(100, 149, 237, 0.8)',      # CornflowerBlue
            'TF; RBP': 'rgba(60, 179, 113, 0.8)',   # MediumSeaGreen
            'Histone': 'rgba(255, 99, 71, 0.8)'     # Tomato
        },
        title="Comparison of Evaluation Metrics Between Model Types",
        labels={"Value": "Metric Value", "Metric": "Evaluation Metric"}
    )
    fig_box.update_layout(
        legend_title_text='Model Type',
        font=dict(size=12)
    )
    st.plotly_chart(fig_box, use_container_width=True)
    
    st.divider()

    # # --- Section 2: ROC Curves (Recreating Figure 2c/d) ---
    # st.header("ROC Curves for Top Performing Models")
    # st.markdown("The Receiver Operating Characteristic (ROC) curves for the top-performing models demonstrate excellent discriminative ability, with Area Under the Curve (AUC) values approaching 1.0.")

    # # Select top 10 models based on AUC for visualization
    # top_models = df_accuracy.nlargest(10, 'eval_auc')

    # fig_roc = go.Figure()
    # fig_roc.add_shape(type='line', line=dict(dash='dash', color='grey'), x0=0, x1=1, y0=0, y1=1)

    # for index, row in top_models.iterrows():
    #     # We generate a representative curve shape, as we don't have the raw TPR/FPR points.
    #     # The AUC value, however, is the real data from your file.
    #     false_positive_rates = [0, 0.01, 0.05, 0.1, 0.2, 0.5, 1]
    #     # Create a curve that visually corresponds to the high AUC
    #     true_positive_rates = [0, 0.80, 0.95, 0.98, 0.99, 1.0, 1.0]
    #     fig_roc.add_trace(go.Scatter(
    #         x=false_positive_rates, y=true_positive_rates, 
    #         name=f"{row['tags']} (AUC={row['eval_auc']:.3f})", 
    #         mode='lines',
    #         hovertemplate='FPR: %{x:.3f}<br>TPR: %{y:.3f}'
    #     ))

    # fig_roc.update_layout(
    #     title='ROC Curves for Top 10 Performing TFBS Models',
    #     xaxis_title='False Positive Rate',
    #     yaxis_title='True Positive Rate',
    #     yaxis=dict(scaleanchor="x", scaleratio=1),
    #     xaxis=dict(constrain='domain'),
    #     legend_title="TFBS Model"
    # )
    # st.plotly_chart(fig_roc, use_container_width=True)

    # st.divider()

    # --- Section 3: Splice Site Model Performance (Recreating Figure 2g) ---
    st.header("Performance of Splice Site Models")
    st.markdown("The radar plot below compares the performance of the separate classifiers trained to recognize donor and acceptor splice sites, showing a robust balance across all metrics.")
    
    # Data taken from your paper for the radar plot
    splice_data = {
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'MCC'],
        'Donor': [94, 95, 93, 0.94, 0.89],
        'Acceptor': [96, 96, 95, 0.96, 0.92]
    }
    df_splice = pd.DataFrame(splice_data)

    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(
        r=df_splice['Donor'],
        theta=df_splice['Metric'],
        fill='toself',
        name='Donor Site Model'
    ))
    fig_radar.add_trace(go.Scatterpolar(
        r=df_splice['Acceptor'],
        theta=df_splice['Metric'],
        fill='toself',
        name='Acceptor Site Model'
    ))

    fig_radar.update_layout(
      polar=dict(
        radialaxis=dict(
          visible=True,
          range=[85, 100] # Adjust range to zoom in on the high performance
        )),
      showlegend=True,
      title="Performance Comparison: Donor vs. Acceptor Splice Site Models"
    )
    st.plotly_chart(fig_radar, use_container_width=True)

else:
    st.warning("Could not load model performance data. Please ensure the data files are correctly placed in the repository.")

