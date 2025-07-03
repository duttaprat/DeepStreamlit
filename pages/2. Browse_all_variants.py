import streamlit as st
import pandas as pd
import json
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, DataReturnMode
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test
import plotly.express as px

st.set_page_config(layout="wide", page_title="Browse Variants")

st.title("📊 Browse All Variants")
st.markdown("Use the sidebar to select the cancer type, regulatory element, and variant type to explore the full dataset.")

# --- Sidebar for selections ---
st.sidebar.header("Data Selection")
cancer_type = st.sidebar.selectbox("Select Cancer Type", ["Brain"], key="cancer_type_browser")
analysis_type = st.sidebar.selectbox("Genomic Regulatory Elements", ["Splice Sites", "TFBS Models"], key="analysis_type_browser")

analysis_options = {
    "Substitutions (SNVs)": "CaVEMan",
    "Insertions & Deletions (Indels)": "sanger_raw_pindel"
}
selected_analysis = st.sidebar.selectbox("Select Variant Type", options=list(analysis_options.keys()), key="variant_type_browser")
data_source = analysis_options[selected_analysis]

# --- Helper Functions (utility and plotting) ---
# (Keep your original helper functions like calculate_p_values, plot_km_curve, etc. here)
# NOTE: To keep this example concise, I am omitting the full function code. 
# You should copy your original functions here.
def calculate_p_values(df_kmf, df_transcript_info):
    # Your original function code here
    # ...
    return df_transcript_info

def plot_km_curve(group_A, group_B, title, logrank_p_value):
    # Your original function code here
    # ...
    kmf = KaplanMeierFitter()
    fig = go.Figure()
    # Add traces for group A
    kmf.fit(group_A['km_time'], event_observed=group_A['km_status'], label=f'Group A (n={len(group_A)})')
    fig.add_trace(go.Scatter(x=kmf.survival_function_.index, y=kmf.survival_function_.iloc[:, 0], mode='lines', name=f'Group A (n={len(group_A)})'))
    # Add traces for group B
    kmf.fit(group_B['km_time'], event_observed=group_B['km_status'], label=f'Group B (n={len(group_B)})')
    fig.add_trace(go.Scatter(x=kmf.survival_function_.index, y=kmf.survival_function_.iloc[:, 0], mode='lines', name=f'Group B (n={len(group_B)})'))
    
    fig.update_layout(
        title=f"{title}<br>Log-Rank p-value: {logrank_p_value:.4f}",
        xaxis_title="Time (Days)",
        yaxis_title="Survival Probability",
        legend_title="Cohorts"
    )
    return fig

# --- Data Loading ---
@st.cache_data
def load_data(cancer, analysis, source):
    base_path = f"data/{cancer}/"
    analysis_folder_map = {"Splice Sites": "Splice_Sites", "TFBS Models": "TFBS_Models"}
    analysis_folder = analysis_folder_map[analysis]
    
    tsv_path = f"{base_path}{analysis_folder}/"
    
    try:
        df_variants_frequency = pd.read_csv(f"{tsv_path}{source}_combined_{analysis_folder}_Variants_frequency.tsv", sep="\t")
        df_transcript_info = pd.read_csv(f"{tsv_path}{source}_combined_{analysis_folder}_Variants_frequency_with_gene_information.tsv", sep="\t")
        df_clinvar = pd.read_csv(f"{tsv_path}{source}_combined_{analysis_folder}_ClinVar_Information.tsv", sep="\t")
        df_clinical = pd.read_csv(f"{base_path}patient_clinical_updated.tsv", sep='\t')
        return df_variants_frequency, df_transcript_info, df_clinvar, df_clinical
    except FileNotFoundError as e:
        st.error(f"Data file not found. Please check your repository. Details: {e}")
        return None, None, None, None
    except Exception as e:
        st.error(f"An error occurred during data loading: {e}")
        return None, None, None, None

df_variants, df_info, df_clinvar, df_clinical = load_data(cancer_type, analysis_type, data_source)

# --- Main App Logic ---
if df_info is not None:
    st.header(f"Exploring {analysis_type} for {selected_analysis}")
    
    # Display the main data table
    gb = GridOptionsBuilder.from_dataframe(df_info)
    gb.configure_pagination(paginationAutoPageSize=False, paginationPageSize=20)
    gb.configure_selection('single', use_checkbox=True)
    gb.configure_side_bar()
    grid_options = gb.build()

    grid_response = AgGrid(
        df_info,
        gridOptions=grid_options,
        update_mode=GridUpdateMode.SELECTION_CHANGED,
        enable_enterprise_modules=False,
        height=600,
        width='100%',
        reload_data=True
    )
    
    selected_rows = grid_response['selected_rows']

    if selected_rows:
        st.divider()
        st.subheader("Survival Analysis for Selected Variant")
        
        # Get data for the selected variant
        selected_variant_info = pd.DataFrame(selected_rows).iloc[0]
        variant_id = selected_variant_info['variant_information']
        patient_ids = selected_variant_info['patient_ids'].split(',')
        
        # Create groups for KM plot
        group_b_ids = [pid.split('_')[0] for pid in patient_ids]
        df_clinical['group'] = df_clinical['manifest_patient_id'].apply(lambda x: 'Mutated (Group B)' if x in group_b_ids else 'Wild-Type (Group A)')
        
        group_A = df_clinical[df_clinical['group'] == 'Wild-Type (Group A)']
        group_B = df_clinical[df_clinical['group'] == 'Mutated (Group B)']
        
        # Perform log-rank test
        if not group_A.empty and not group_B.empty:
            results = logrank_test(group_A['km_time'], group_B['km_time'], event_observed_A=group_A['km_status'], event_observed_B=group_B['km_status'])
            p_value = results.p_value

            # Plot KM curve
            km_fig = plot_km_curve(group_A, group_B, f"Survival Plot for Variant: {variant_id}", p_value)
            st.plotly_chart(km_fig, use_container_width=True)
        else:
            st.warning("Not enough data to perform survival analysis for this variant (one group is empty).")

else:
    st.info("Waiting for data to load...")

