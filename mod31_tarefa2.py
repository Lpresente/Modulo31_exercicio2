#!/usr/bin/env python
# coding: utf-8

# In[4]:


import streamlit as st
import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from gower import gower_matrix
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster


# =========================
# CONFIGURAÇÕES
# =========================
SAMPLE_SIZE = 3000  # reduz processamento
DATA_URL = "https://raw.githubusercontent.com/Lpresente/Modulo31_exercicio2/main/online_shoppers_intention.csv"
LOGO_URL = "https://raw.githubusercontent.com/Lpresente/Modulo31_exercicio2/main/bac_logo.png"
ICON_URL = "https://raw.githubusercontent.com/Lpresente/Modulo31_exercicio2/main/icon.ico"


# =========================
# CACHE FUNCTIONS
# =========================
@st.cache_data(show_spinner=False)
def load_data():
    df = pd.read_csv(DATA_URL)
    if len(df) > SAMPLE_SIZE:
        df = df.sample(SAMPLE_SIZE, random_state=42)
    return df


@st.cache_data(show_spinner=False)
def calcular_gower(data_x, cat_features):
    return gower_matrix(data_x=data_x, cat_features=cat_features)


# =========================
# DENDROGRAMA
# =========================
def plot_dendrogram(color_threshold: float, num_groups: int, Z):
    fig, ax = plt.subplots(figsize=(24, 6))

    dendrogram(
        Z,
        p=6,
        truncate_mode='level',
        color_threshold=color_threshold,
        show_leaf_counts=True,
        leaf_font_size=8,
        leaf_rotation=45,
        show_contracted=True,
        ax=ax
    )

    ax.set_ylabel('Distância')
    ax.set_title(f'Dendrograma Hierárquico - {num_groups} Grupos')
    ax.set_yticks(np.linspace(0, .6, num=31))
    ax.set_xticks([])

    st.pyplot(fig)
    plt.close(fig)


# =========================
# APP PRINCIPAL
# =========================
def main():
    st.set_page_config(
        page_title="EBAC | Projeto Agrupamento Hierárquico",
        page_icon=ICON_URL,
        layout="wide"
    )

    # Sidebar
    st.sidebar.image(LOGO_URL, width=250)
    st.sidebar.markdown("# Projeto de Agrupamento Hierárquico")

    # Carregar dados
    df = load_data()

    st.title("Agrupamento Hierárquico")
    st.write("Análise de intenção de compra de visitantes online.")
    
# =========================
# VISUALIZAÇÃO INICIAL
# =========================
    st.subheader("Base de Dados")
    st.dataframe(df.head())

    st.subheader("Contagem Revenue")
    st.write(df['Revenue'].value_counts())

    fig, ax = plt.subplots()
    sns.countplot(x='Revenue', data=df, ax=ax)
    st.pyplot(fig)
    plt.close(fig)

# =========================
# DESCRITIVA
# =========================
    st.subheader("Informações Gerais")
    st.info(
        f"""
        Linhas: {df.shape[0]}
        Colunas: {df.shape[1]}
        Missing: {df.isna().sum().sum()}
        """
    )

    buffer = io.StringIO()
    df.info(buf=buffer)
    st.text(buffer.getvalue())

    st.subheader("Estatísticas Descritivas")
    st.dataframe(df.describe())

    st.subheader("Mapa de Correlação")
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(df.corr(numeric_only=True), cmap='viridis', ax=ax)
    st.pyplot(fig)
    plt.close(fig)

# =========================
# FEATURE SELECTION
# =========================
    session_navigation_pattern = [
        'Administrative',
        'Informational',
        'ProductRelated',
        'PageValues',
        'OperatingSystems',
        'Browser',
        'TrafficType',
        'VisitorType'
    ]

    temporal_indicators = ['SpecialDay', 'Month', 'Weekend']
    numerical = ['ProductRelated', 'PageValues', 'SpecialDay']

    df_ = df[session_navigation_pattern + temporal_indicators]
    df_dummies = pd.get_dummies(df_, drop_first=False)

    categorical_features = df_dummies.drop(columns=numerical).columns.values
    cat_features = [col in categorical_features for col in df_dummies.columns]

# =========================
# MATRIZ GOWER
# =========================
    st.subheader("Matriz de Distância Gower")
    with st.spinner("Calculando matriz Gower..."):
        dist_gower = calcular_gower(df_dummies, cat_features)

    st.success("Matriz calculada com sucesso")
    st.dataframe(pd.DataFrame(dist_gower).head())

# =========================
# LINKAGE
# =========================
    gdv = squareform(dist_gower, force='tovector')
    Z = linkage(gdv, method='complete')

    st.subheader("Dendrogramas")
    for qtd, threshold in [(3, .53), (4, .5)]:
        st.write(f"### {qtd} grupos")
        plot_dendrogram(threshold, qtd, Z)

    # =========================
    # CLUSTERIZAÇÃO
    # =========================
    df['grupo_3'] = fcluster(Z, t=3, criterion='maxclust')
    df['grupo_4'] = fcluster(Z, t=4, criterion='maxclust')

    st.subheader("Distribuição Grupo 3")
    st.dataframe(df['grupo_3'].value_counts())

    st.subheader("Distribuição Grupo 4")
    st.dataframe(df['grupo_4'].value_counts())

    # =========================
    # PAIRPLOT OTIMIZADO
    # =========================
    st.subheader("Pairplot (amostra reduzida)")
    sample_df = df.sample(min(1000, len(df)), random_state=42)

    pairplot_fig = sns.pairplot(
        sample_df[['BounceRates', 'Revenue', 'SpecialDay', 'grupo_3', 'grupo_4']],
        hue='Revenue'
    )

    st.pyplot(pairplot_fig.figure)
    plt.close('all')

    # =========================
    # CONCLUSÃO
    # =========================
    st.subheader("Conclusão")
    st.write(
        "Visitantes recorrentes demonstram maior propensão de compra, permitindo ações de marketing mais direcionadas."
    )


if __name__ == '__main__':
    main()


# In[ ]:




