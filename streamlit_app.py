import streamlit as st
import pandas as pd
import plotly.express as px

# --- Page Configuration ---
st.set_page_config(
    layout="wide",
    page_title="DeepVRegulome",
    page_icon="🧬"
)

# --- Header and Introduction ---
st.title("DeepVRegulome: A DNABERT-based Framework")
st.subheader("Predicting the functional impact of short genomic variants on the human regulome")

st.markdown("""
Welcome to the interactive data portal for **DeepVRegulome**, a deep-learning method for predicting and interpreting functionally disruptive variants in the human regulome. This framework combines over 700 fine-tuned DNABERT models with comprehensive analysis tools to prioritize clinically relevant non-coding mutations.

This portal allows you to explore the data and key findings from our study on the TCGA glioblastoma (GBM) cohort.

**Navigate through the pages in the sidebar to:**
- **Model Performance:** Evaluate the accuracy and predictive power of our underlying models.
- **Browse All Variants:** Interactively filter and explore the complete dataset of predicted functional variants.
- **Key Discoveries in GBM:** View the specific, survival-associated variants highlighted in our paper.
- **Motif Validation:** See how our models learned biologically meaningful transcription factor binding motifs.
""")

st.divider()

# --- Summary Statistics (Recreating Table 1) ---
st.header("Summary of Predicted Functional Variants in GBM")
st.markdown("This table summarizes the total number of variants analyzed and the subset predicted as functionally disruptive by DeepVRegulome, focusing on those present in >10% of the GBM patient cohort.")

# Data for the summary table - directly from your paper's Table 1
summary_data = {
    "Fine-tuned Models": ["Splice Sites", "Splice Sites", "Splice Sites", "Splice Sites", "ChIP-seq Models", "ChIP-seq Models", "ChIP-seq Models", "ChIP-seq Models"],
    "Subtype": ["Acceptor", "Acceptor", "Donor", "Donor", "Histone Markers", "Histone Markers", "TFs", "TFs"],
    "Variant Type": ["SNVs", "Indels", "SNVs", "Indels", "SNVs", "Indels", "SNVs", "Indels"],
    "Total Variants": [19968, 34718, 23656, 20171, 127297, 254566, 932133, 2130274],
    "Total Regions": [14743, 19897, 15849, 13491, 57013, 87949, 358414, 533433],
    "Predicted Functional Variants (>10% Samples)": ["299 (1)", "1,822 (437)", "673 (4)", "504 (130)", "387 (2)", "536 (150)", "19,709 (1,087)", "35,867 (8,598)"],
    "Affected Regions (>10% Samples)": ["265 (3)", "1,375 (408)", "545 (4)", "423 (122)", "311 (5)", "373 (129)", "16,566 (322)", "30,411 (7,297)"]
}
summary_df = pd.DataFrame(summary_data)

# Display the dataframe with styling
st.dataframe(summary_df, use_container_width=True)

st.divider()

# --- Patient Cohort Sunburst Chart ---
st.header("GBM Patient Cohort Overview")

try:
    # Load clinical data
    clinical_file_path = "data/Brain/patient_clinical_updated.tsv"
    df_clinical = pd.read_csv(clinical_file_path, sep='\t')

    # Filter and prepare data for the chart
    df_clinical = df_clinical[(df_clinical["project_id"] == "CPTAC-3") | (df_clinical["project_id"] == "TCGA-GBM")].reset_index(drop=True)
    total_patients = df_clinical.shape[0]
    df_clinical['total_patients'] = f'Total GBM Patients: {total_patients}'
    df_clinical['primary_diagnosis'] = df_clinical['primary_diagnosis'].fillna('Unknown')
    
    # Create Sunburst chart
    sunburst_fig = px.sunburst(
        df_clinical,
        path=['total_patients', 'project_id', 'gender', 'primary_diagnosis'],
        title="Distribution of GBM Patients by Project, Gender, and Diagnosis",
        color='project_id',
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    sunburst_fig.update_layout(margin=dict(t=40, l=10, r=10, b=10))
    st.plotly_chart(sunburst_fig, use_container_width=True)

except FileNotFoundError:
    st.error("Could not find the clinical data file. Please ensure `data/Brain/patient_clinical_updated.tsv` is in your repository.")
except Exception as e:
    st.error(f"An error occurred while loading the patient cohort chart: {e}")

