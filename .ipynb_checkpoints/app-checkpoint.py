
import streamlit as st
import pandas as pd
import numpy as np



st.set_page_config(
    page_title="Dashboard d’aide à la décision environnementale",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded"
)


def load_data(file):
    ext = file.name.split(".")[-1].lower()

    if ext == "csv":
        return pd.read_csv(file)
    elif ext in ["xls", "xlsx"]:
        return pd.read_excel(file)
    else:
        st.error("Format de fichier non supporté.")
        return None

menu = st.sidebar.selectbox(
    "Navigation",
    ["Accueil", "Exploration de données", "Visualisations"]
)

if menu == "Accueil":
    st.title("Dashboard d’aide à la décision environnementale")
    st.subheader("Bienvenue dans le dashboard interactif")
    st.markdown("")


elif menu == "Exploration de données":
    st.title("Exploration de données")

    file_uploader = st.file_uploader(
        "Téléchargez votre fichier de données",
        type=["csv", "xls", "xlsx"]
    )

    if file_uploader:
        df = load_data(file_uploader)

        if df is not None:
            st.write("### Aperçu des données téléchargées")
            df.dropna(inplace=True)
            st.dataframe(df)


elif menu == "Visualisations":
    st.title("Visualisations")
    st.write("Les graphiques seront ajoutés ici.")


