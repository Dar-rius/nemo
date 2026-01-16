import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="NEMO - Neuro-Symbolic Decision Support", layout="wide")
st.title("NEMO")
st.subheader("IA Neuro-Symbolique pour l’aide à la décision stratégique")

try:
    df = pd.read_csv("results/embeddings/node2vec_clustered.csv")
    st.success(f"{len(df)} entités analysées")
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Distribution des types**")
        st.bar_chart(df['label'].value_counts())
    with col2:
        st.write("**Clusters stratégiques**")
        st.bar_chart(df['cluster'].value_counts())
    
    st.dataframe(df[['name', 'label', 'cluster']].sort_values('cluster'), use_container_width=True)

except Exception as e:
    st.error("Lancez d’abord le pipeline d’entraînement.")
