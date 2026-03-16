import streamlit as st
import osmnx as ox
import folium
from streamlit_folium import st_folium
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import warnings
import streamlit.components.v1 as components

# --- CONFIGURATION ---
warnings.filterwarnings("ignore")
st.set_page_config(page_title="HAVEN RADAR | Détecteur", layout="wide")

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

# --- STYLE CSS ---
st.markdown("""
    <style>
    div.stButton > button:first-child {
        background-color: #28a745; color: white; border-radius: 5px; font-weight: bold;
    }
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
            # L'index est synchronisé avec le dernier clic sur la carte
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
            if cols_fav[i % 4].button(f"#{num}", key=f"fbtn_{num}_{coords[0]}"):
                st.session_state.map_center = [coords[0], coords[1]]
                st.session_state.zoom_level = 18
                st.session_state.sync_idx = num - 1
                st.rerun()
        
        if st.button("Vider les favoris", use_container_width=True):
            st.session_state.favs = {}
            st.rerun()

# --- LOGIQUE DE SCAN ---
if lancer_scan or (commune_in and st.session_state.get('last_city') != commune_in):
    if not commune_in:
        st.error("⚠️ Entrez une commune.")
    else:
        components.html(js_close_sidebar, height=0, width=0)
        try:
            st.session_state['last_city'] = commune_in
            
            with st.container():
                p_bar = st.progress(0)
                
                # Étape 1
                p_bar.progress(15)
                base = ox.geocode_to_gdf(commune_in)
                
                # Étape 2
                p_bar.progress(40)
                geom_c = base.geometry.iloc[0]
                voisines = ox.features_from_polygon(geom_c.buffer(0.015), tags={'admin_level': '8'})
                secteur = pd.concat([base, voisines[voisines.geometry.intersects(geom_c)]]).to_crs(epsg=2154)
                union_zone = secteur.geometry.union_all()
                
                # Étape 3
                p_bar.progress(70)
                bbox = secteur.to_crs(epsg=4326).geometry.union_all().buffer(0.01) 
                bat = ox.features_from_polygon(bbox, tags={'building': True})
                routes = ox.features_from_polygon(bbox, tags={'highway': ['primary', 'secondary', 'tertiary', 'residential', 'unclassified', 'trunk']})
                
                # Étape 4 (Correction du blocage à 70%)
                bat = bat[bat.geometry.type.isin(['Polygon', 'MultiPolygon'])].copy().to_crs(epsg=2154)
                routes = routes.to_crs(epsg=2154)
                bat['d_route'] = bat.geometry.centroid.apply(lambda x: routes.distance(x).min())
                candidates = bat[bat.geometry.centroid.within(union_zone) & (bat['d_route'] >= dist_route_val)].copy()
                p_bar.progress(90)
                
                if not candidates.empty:
                    coords_toutes = list(zip(bat.geometry.centroid.x, bat.geometry.centroid.y))
                    coords_candidates = list(zip(candidates.geometry.centroid.x, candidates.geometry.centroid.y))
                    nn = NearestNeighbors(radius=rayon_iso_val).fit(coords_toutes)
                    adj = nn.radius_neighbors_graph(coords_candidates).toarray()
                    candidates['taille_hameau'] = adj.sum(axis=1)
                    st.session_state.last_res = candidates[candidates['taille_hameau'] <= (taille_hameau_max + 1)].copy().to_crs(epsg=4326)
                    st.session_state.map_center = [st.session_state.last_res.geometry.centroid.y.mean(), st.session_state.last_res.geometry.centroid.x.mean()]
                    st.session_state.zoom_level = 13
                
                p_bar.progress(100)
                p_bar.empty()
        except Exception as e:
            st.error(f"❌ Erreur : {str(e)}")

# --- AFFICHAGE RESULTATS ET CARTE ---
if st.session_state.last_res is not None:
    res = st.session_state.last_res
    if res.empty:
        st.warning("⚠️ Aucun bâtiment ne correspond aux critères.")
    else:
        st.success(f"✅ {len(res)} Havens détectés !")
        
        m = folium.Map(location=st.session_state.map_center, zoom_start=st.session_state.zoom_level)
        folium.TileLayer(tiles='https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}', attr='Google', name='Satellite', max_zoom=22).add_to(m)

        for i, (idx, row) in enumerate(res.iterrows()):
            lat, lon = row.geometry.centroid.y, row.geometry.centroid.x
            pop_html = f"""
            <div style='font-family:Arial; width:170px;'>
                <b>HAVEN #{i+1}</b><br>
                <small>Hameau de {int(row['taille_hameau'])} bât.</small><hr>
                <a href='https://www.google.com/maps/search/?api=1&query={lat},{lon}' target='_blank' style='color:#4285F4;text-decoration:none;'>🗺️ Google Maps</a><br>
                <a href='https://waze.com/ul?ll={lat},{lon}&navigate=yes' target='_blank' style='color:#33CCFF;text-decoration:none;'>🚙 Waze</a><br>
                <a href='https://www.google.com/maps/@?api=1&map_action=pano&viewpoint={lat},{lon}' target='_blank' style='color:#EA4335;text-decoration:none;'>🏙️ Street View</a>
            </div>
            """
            icon_html = f'<div style="background-color:red;border:1px solid white;border-radius:5
