import streamlit as st
import osmnx as ox
import geopandas as gpd
import folium
from streamlit_folium import st_folium
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import warnings
import streamlit.components.v1 as components
import time

# --- CONFIGURATION & CACHE (Mis à jour pour OSMnx v2.0+) ---
warnings.filterwarnings("ignore")
st.set_page_config(page_title="HAVEN RADAR", layout="wide")

# Nouvelle méthode officielle pour configurer le cache sans casser l'application
ox.settings.use_cache = True
ox.settings.cache_folder = './cache'
ox.settings.log_console = False

# Utilisation d'un serveur Overpass alternatif pour éviter les saturations (erreur 502)
ox.settings.overpass_url = "https://overpass.kumi.systems/api/interpreter"

# --- INITIALISATION ÉTATS ---
if 'favs' not in st.session_state:
    st.session_state.favs = {} 
if 'map_center' not in st.session_state:
    st.session_state.map_center = None
if 'last_res' not in st.session_state:
    st.session_state.last_res = None
if 'zoom_level' not in st.session_state:
    st.session_state.zoom_level = 13
if 'sync_idx' not in st.session_state:
    st.session_state.sync_idx = 0
if 'last_city' not in st.session_state:
    st.session_state.last_city = ""

# --- STYLE CSS ---
st.markdown("""
    <style>
    div.stButton > button:first-child { background-color: #28a745; color: white; border-radius: 5px; font-weight: bold; }
    .stProgress > div > div > div > div { background-color: #28a745; }
    </style>
""", unsafe_allow_html=True)

js_close_sidebar = "<script>parent.document.querySelector('button[kind=\"headerNoPadding\"]').click();</script>"

st.title("🏡 Haven Radar")

# --- BARRE LATÉRALE ---
with st.sidebar:
    st.header("Paramètres du Scan")
    commune_in = st.text_input("Secteur :", placeholder="Commune...", key="city_input")
    dist_route_val = st.slider("Distance Route (m) :", 30, 300, 70)
    rayon_iso_val = st.slider("Rayon Isolement (m) :", 50, 600, 180)
    taille_hameau_max = st.number_input("Voisins Max autorisés :", 1, 13, 2)
    lancer_scan = st.button("Lancer le Scan", use_container_width=True)
    
    st.markdown("---")
    st.subheader("⭐ Gestion Favoris")
    if st.session_state.last_res is not None:
        res_count = len(st.session_state.last_res)
        col_sel, col_add = st.columns([2, 1])
        with col_sel:
            st.session_state.sync_idx = st.selectbox(
                "Choisir #", 
                range(res_count), 
                index=min(st.session_state.sync_idx, res_count-1),
                format_func=lambda x: f"Haven #{x+1}", 
                label_visibility="collapsed"
            )
        with col_add:
            if st.button("Ajouter"):
                row = st.session_state.last_res.iloc[st.session_state.sync_idx]
                st.session_state.favs[st.session_state.sync_idx+1] = (row.geometry.centroid.y, row.geometry.centroid.x)
    
    if st.session_state.favs:
        st.write("📍 Accès rapide :")
        cols_fav = st.columns(4)
        for i, (num, coords) in enumerate(sorted(st.session_state.favs.items())):
            if cols_fav[i % 4].button(f"#{num}", key=f"fbtn_{num}"):
                st.session_state.map_center = [coords[0], coords[1]]
                st.session_state.zoom_level = 18
                st.session_state.sync_idx = num - 1
                st.rerun()
        if st.button("Vider les favoris", use_container_width=True):
            st.session_state.favs = {}
            st.rerun()

# --- LOGIQUE DE SCAN ---
if lancer_scan:
    if not commune_in:
        st.error("⚠️ Entrez une commune.")
    else:
        # PURGE SI NOUVELLE COMMUNE
        if commune_in.strip().lower() != st.session_state.last_city.strip().lower():
            st.session_state.favs = {}
            st.session_state.sync_idx = 0
            st.session_state.last_res = None
            
        components.html(js_close_sidebar, height=0, width=0)
        try:
            st.session_state.last_city = commune_in
            
            # Utilisation d'un conteneur de statut pour regrouper visuellement les étapes
            with st.status("Initialisation du Scan géomatique...", expanded=True) as status:
                p_bar = st.progress(0, text="Connexion aux serveurs cartographiques...")
                
                # Étape 1 : Géocodage de la commune
                base = ox.geocode_to_gdf(commune_in)
                
                # Progression fluide simulée jusqu'à 25%
                for percent in range(1, 26):
                    time.sleep(0.02)
                    p_bar.progress(percent, text=f"Localisation de la commune sur la carte : {percent}%")
                
                # Étape 2 : Communes voisines
                geom_c = base.geometry.iloc[0]
                voisines = ox.features_from_polygon(geom_c.buffer(0.015), tags={'admin_level': '8'})
                secteur = pd.concat([base, voisines[voisines.geometry.intersects(geom_c)]]).to_crs(epsg=2154)
                union_zone = secteur.geometry.union_all()
                
                # Progression fluide simulée jusqu'à 50%
                for percent in range(26, 51):
                    time.sleep(0.02)
                    p_bar.progress(percent, text=f"Calcul de la zone tampon et du secteur étendu : {percent}%")
                
                # Étape 3 : Requête OpenStreetMap
                bbox_m = secteur.geometry.union_all().buffer(1000)
                bbox = gpd.GeoSeries([bbox_m], crs=2154).to_crs(epsg=4326).iloc[0]
                
                p_bar.progress(51, text="Téléchargement des bâtiments et routes OpenStreetMap (Patientez)...")
                
                bat = ox.features_from_polygon(bbox, tags={'building': True})
                routes = ox.features_from_polygon(bbox, tags={'highway': ['primary', 'secondary', 'tertiary', 'residential', 'unclassified', 'trunk']})
                
                # Progression fluide simulée après le téléchargement bloquant
                for percent in range(52, 86):
                    time.sleep(0.01)
                    p_bar.progress(percent, text=f"Filtrage des polygones et calcul des axes routiers : {percent}%")
                
                # Étape 4 : Calculs géospatiaux mathématiques (KNN & distances)
                bat = bat[bat.geometry.type.isin(['Polygon', 'MultiPolygon'])].copy().to_crs(epsg=2154)
                routes = routes.to_crs(epsg=2154)
                
                routes_union = routes.geometry.union_all()
                bat['d_route'] = bat.geometry.centroid.apply(lambda x: routes_union.distance(x))
                
                candidates = bat[bat.geometry.centroid.within(union_zone) & (bat['d_route'] >= dist_route_val)].copy()
                
                if not candidates.empty:
                    coords_toutes = list(zip(bat.geometry.centroid.x, bat.geometry.centroid.y))
                    coords_candidates = list(zip(candidates.geometry.centroid.x, candidates.geometry.centroid.y))
                    nn = NearestNeighbors(radius=rayon_iso_val).fit(coords_toutes)
                    adj = nn.radius_neighbors_graph(coords_candidates).toarray()
                    candidates['taille_hameau'] = adj.sum(axis=1)
                    st.session_state.last_res = candidates[candidates['taille_hameau'] <= (taille_hameau_max + 1)].
