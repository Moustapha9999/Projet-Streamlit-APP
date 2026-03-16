
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# CONFIGURATION DE LA PAGE 
st.set_page_config(
    page_title="Dashboard d’aide à la décision environnementale",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# FONCTIONS DE TRAITEMENT DES DONNÉES 

@st.cache_data
def load_data(file):
    """Charge le fichier de données (CSV ou Excel)."""
    ext = file.name.split(".")[-1].lower()
    try:
        if ext == "csv":
            return pd.read_csv(file)
        elif ext in ["xls", "xlsx"]:
            return pd.read_excel(file)
        else:
            st.error("Format de fichier non supporté.")
            return None
    except Exception as e:
        st.error(f"Erreur lors du chargement des données : {e}")
        return None

@st.cache_data
def process_data(df):
    """
    Nettoie, agrège (moyenne par ville), impute les NaN et prépare les données pour le clustering.
    Retourne le DataFrame agrégé (non normalisé) et le DataFrame normalisé pour K-Means.
    """
    # Colonnes de polluants pertinentes pour l'analyse
    polluant_cols = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'CO', 'SO2', 'O3', 'Benzene', 'Toluene', 'Xylene', 'NH3']
    
    # 1. Agrégation par ville (calcul de la moyenne, ignore les NaN )
    data_seg = df.groupby('City')[polluant_cols].mean().reset_index()

    # 2. Imputation des NaN au niveau de la ville (remplacement par la moyenne nationale)
    means_fill = data_seg[polluant_cols].mean()
    data_seg.fillna(means_fill, inplace=True)

    # 3. Préparation pour la normalisation
    X = data_seg.drop('City', axis=1)
    
    # 4. Normalisation (StandardScaler)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return data_seg, X_scaled, polluant_cols

@st.cache_data
def run_kmeans(X_scaled, data_seg, k):
    """Effectue la segmentation K-Means et l'Analyse en Composantes Principales (PCA)."""
    
    # K-Means
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
    data_seg['Cluster'] = kmeans.fit_predict(X_scaled)
    
    # PCA pour la visualisation 2D
    pca = PCA(n_components=2)
    components = pca.fit_transform(X_scaled)

    pca_df = pd.DataFrame(data=components, columns=['PC1', 'PC2'])
    pca_df['City'] = data_seg['City']
    pca_df['Cluster'] = data_seg['Cluster'].astype(str)
    
    # Profils d'interprétation
    profils_clusters = data_seg.drop('City', axis=1).groupby('Cluster').mean().reset_index()

    return data_seg, pca_df, profils_clusters

#  Dictionnaire de mappage des profils d'interprétation 
# Y'a 5 profils. VEUILLEZ AJUSTER SELON VOTRE ANALYSE EXACTE.
CLUSTER_MAPPING = {
    0: "Risque Modéré (Particules/NO)",             
    1: "Faible Risque (Air le plus Pur)",           
    2: "Haut Risque (Gaz Aromatiques)",              
    3: "Très Haut Risque (Particules/NH3)",         
    4: "Haut Risque Généralisé (Gaz/Ozone)"          
}

# (MENU) 

menu = st.sidebar.selectbox(
    "Navigation",
    ["Accueil", "Exploration de données", "Visualisations"]
)

#  Page d'Accueil 
if menu == "Accueil":
    st.title("Dashboard d’aide à la décision environnementale")
    st.subheader("Analyse de la qualité de l'air et segmentation des villes")
    st.markdown("""
        Bienvenue dans le dashboard interactif. Ce tableau de bord permet d'analyser les données de pollution (WAQI)
        et de regrouper les villes selon des profils de risque homogènes via l'algorithme K-Means. """)

# Page d'Exploration 
elif menu == "Exploration de données":
    st.title("Exploration de données")

    file_uploader = st.file_uploader(
        "Téléchargez votre fichier de données (e.g., city_day.csv)",
        type=["csv", "xls", "xlsx"]
    )

    if file_uploader:
        df = load_data(file_uploader)
        
        if df is not None:
           
            # Lancement du prétraitement
            data_seg, X_scaled, polluant_cols = process_data(df)
            st.session_state['data_seg'] = data_seg
            st.session_state['X_scaled'] = X_scaled
            st.session_state['polluant_cols'] = polluant_cols
            
            st.write("### Aperçu des données agrégées et nettoyées")
            st.dataframe(data_seg)
            st.success(f"Le prétraitement est terminé. {data_seg.shape[0]} villes sont prêtes pour la segmentation. Vous pouvez passer à la section 'Visualisations'.")


# Page de Visualisations
elif menu == "Visualisations":
    st.title("Visualisations et Lecture Décisionnelle")

    if 'data_seg' not in st.session_state:
        st.warning("Veuillez d'abord télécharger et traiter les données dans la section 'Exploration de données'.")
        st.stop()
    
    data_seg = st.session_state['data_seg']
    X_scaled = st.session_state['X_scaled']
    polluant_cols = st.session_state['polluant_cols']
    
    # VUE SYNTHÉTIQUE 
    st.header("1. Indicateurs de Performance Globaux")
    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Nombre de Villes Analysées", f"{data_seg.shape[0]}")
    
    # Calcul des moyennes des polluants clés
    pm25_avg = data_seg['PM2.5'].mean()
    pm10_avg = data_seg['PM10'].mean()
    no2_avg = data_seg['NO2'].mean()
    
    col2.metric("Moyenne PM2.5 ", f"{pm25_avg:.2f}")
    col3.metric("Moyenne PM10 ", f"{pm10_avg:.2f}")
    col4.metric("Moyenne NO2 ", f"{no2_avg:.2f}")

    st.markdown("---")

    # SÉLECTION ET COMPARAISON DES POLLUANTS 
    st.header("2. Analyse et Comparaison des Polluants")

    # Filtre interactif
    polluant_selectionne = st.selectbox(
        "Sélectionnez le Polluant à Analyser",
        polluant_cols,
        index=polluant_cols.index('PM2.5')
    )
    
    col_g1, col_g2 = st.columns(2)

    # Graphique de comparaison des villes
    with col_g1:
        st.subheader(f"Niveau moyen de {polluant_selectionne} par ville")
        df_sorted = data_seg.sort_values(by=polluant_selectionne, ascending=False).head(15)
        fig_comp = px.bar(
            df_sorted,
            x='City',
            y=polluant_selectionne,
            title=f"Top 15 Villes pour {polluant_selectionne}",
            color_discrete_sequence=px.colors.qualitative.Bold
        )
        st.plotly_chart(fig_comp, use_container_width=True)

    # Graphique de distribution (Boxplot) pour ce polluant
    with col_g2:
        st.subheader(f"Distribution de {polluant_selectionne}")
        fig_box = px.box(
            data_seg,
            y=polluant_selectionne,
            title=f"Distribution de {polluant_selectionne} (Échelle Nationale)",
            height=450
        )
        st.plotly_chart(fig_box, use_container_width=True)
        


    st.markdown("---")

    # SEGMENTATION ET LECTURE DÉCISIONNELLE 
    st.header("3. Segmentation K-Means des Villes")

    # Choix du nombre de clusters (k optimal = 10 selon notre cas)
    k_optimal = st.slider("Sélectionnez le nombre de Clusters (k)", 2, 12, 5)
    
    # Lancement du K-Means et PCA
    data_seg_km, pca_df, profils_clusters = run_kmeans(X_scaled, data_seg.copy(), k_optimal)
    
    # Mappage des profils pour l'affichage
    def map_profil(cluster_id):
        return CLUSTER_MAPPING.get(cluster_id, f"Cluster {cluster_id} (À Interpréter)")

    pca_df['Profil'] = pca_df['Cluster'].astype(int).apply(map_profil)
    profils_clusters['Profil'] = profils_clusters['Cluster'].astype(int).apply(map_profil)
    
    col_seg1, col_seg2 = st.columns([2, 1])

    # Visualisation de la Segmentation (PCA)
    with col_seg1:
        st.subheader("Regroupement des Villes par niveau de pollution (Visualisation PCA)")
        fig_seg = px.scatter(
            pca_df,
            x='PC1',
            y='PC2',
            color='Profil',
            hover_data={'City': True, 'Cluster': True, 'PC1': False, 'PC2': False},
            title="Segmentation des Villes par Profil de Pollution",
            labels={'PC1': 'Composante Principale 1', 'PC2': 'Composante Principale 2'},
            height=600
        )
        st.plotly_chart(fig_seg, use_container_width=True)
        


    # Lecture Décisionnelle (Villes à Risque Élevé)
    with col_seg2:
        st.subheader("Identification des Villes à Risque Élevé")
        
        profil_selection = st.selectbox(
            "Filtrer les Villes par Profil de Risque",
            pca_df['Profil'].unique()
        )

        villes_risque = pca_df[pca_df['Profil'] == profil_selection]['City'].tolist()
        st.info(f"Les **{len(villes_risque)}** villes dans le profil **{profil_selection}** sont :")
        st.markdown(f"**{', '.join(villes_risque)}**")
        
        st.markdown("**Aide à la Décision :**")
        if 'Haut Risque' in profil_selection:
             st.error("Action Immédiate : Ces villes nécessitent des mesures urgentes (interdiction de certaines industries, restriction de trafic, etc.).")
        elif 'Faible Risque' in profil_selection:
             st.success("Surveillance : Les autorités doivent maintenir une surveillance pour prévenir la dégradation.")
        else:
             st.warning("Mesures Préventives : Surveillance accrue et plans d'action en cas de pic de pollution.")


    # Caractérisation des Clusters (Profils Moyens)
    st.subheader("Caractérisation des Profils (Moyennes des Polluants)")
    st.markdown("Ce tableau montre les concentrations moyennes réelles (non normalisées) de chaque polluant pour justifier l'interprétation métier.")
    # On masque la colonne 'Cluster' de la DataFrame affichée
    st.dataframe(profils_clusters.drop('Cluster', axis=1), use_container_width=True)

  