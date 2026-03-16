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

# --- STYLE CSS ---
st.markdown("""
    <style>
    div.stButton > button:first-child {
        background-color: #28a745; color: white; border-radius: 5px; font-weight: bold;
    }
    .loading-text { font-weight: bold; font-size: 16px; margin-bottom: 5px; }
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
    taille_hameau_max = st.number_input("Taille hameau max :", 1, 13, 3)
    
    lancer_scan = st.button("🚀 Lancer le Scan", use_container_width=True)
    
    st.markdown("---")
    st.subheader("⭐ Favoris")
    
    # Affichage des favoris existants
    if st.session_state.favs:
        cols = st.columns(4)
        for i, (num, coords) in enumerate(sorted(st.session_state.favs.items())):
            if cols[i % 4].button(f"#{num}", key=f"fav_{num}_{coords[0]}"):
                st.session_state.map_center = [coords[0], coords[1]]
                st.session_state.zoom_level = 18
                st.rerun()
        
        if st.button("🗑️ Vider les favoris", use_container_width=True):
            st.session_state.favs = {}
            st.rerun()

# --- LOGIQUE DE SCAN AVEC BARRE DE PROGRESSION ---
if lancer_scan or (commune_in and st.session_state.get('last_city') != commune_in):
    if not commune_in:
        st.error("⚠️ Entrez une commune.")
    else:
        components.html(js_close_sidebar, height=0, width=0)
        try:
            st.session_state['last_city'] = commune_in
            
            # Réintégration de la barre de progression
            progress_container = st.container()
            with progress_container:
                st.markdown('<p class="loading-text">🛰️ Initialisation du scan...</p>', unsafe_allow_html=True)
                bar = st.progress(10)
                
                base = ox.geocode_to_gdf(commune_in)
                bar.progress(30)
                
                geom_c = base.geometry.iloc[0]
                voisines = ox.features_from_polygon(geom_c.buffer(0.015), tags={'admin_level': '8'})
                secteur = pd.concat([base, voisines[voisines.geometry.intersects(geom_c)]]).to_crs(epsg=2154)
                union_zone = secteur.geometry.union_all()
                bar.progress(50)
                
                bbox = secteur.to_crs(epsg=4326).geometry.union_all().buffer(0.01) 
                bat = ox.features_from_polygon(bbox, tags={'building': True})
                routes = ox.features_from_polygon(bbox, tags={'highway': ['primary', 'secondary', 'tertiary', 'residential', 'unclassified', 'trunk']})
                bar.progress(70)
                
                bat = bat[bat.geometry.type.isin(['Polygon', 'MultiPolygon'])].copy().to_crs(epsg=2154)
                routes = routes.to_crs(epsg=2154)
                bat['d_route'] = bat.geometry.centroid.apply(lambda x: routes.distance(x).min())
                candidates = bat[bat.geometry.centroid.within(union_zone) & (bat['d_route'] >= dist_route_val)].copy()
                bar.progress(90)
                
                if not candidates.empty:
                    coords_toutes = list(zip(bat.geometry.centroid.x, bat.geometry.centroid.y))
                    coords_candidates = list(zip(candidates.geometry.centroid.x, candidates.geometry.centroid.y))
                    nn = NearestNeighbors(radius=rayon_iso_val).fit(coords_toutes)
                    adj = nn.radius_neighbors_graph(coords_candidates).toarray()
                    candidates['taille_hameau'] = adj.sum(axis=1)
                    st.session_state.last_res = candidates[candidates['taille_hameau'] <= taille_hameau_max].copy().to_crs(epsg=4326)
                    st.session_state.map_center = [st.session_state.last_res.geometry.centroid.y.mean(), st.session_state.last_res.geometry.centroid.x.mean()]
                    st.session_state.zoom_level = 13
                
                bar.progress(100)
                progress_container.empty()
        except Exception as e:
            st.error(f"❌ Erreur : {str(e)}")

# --- AFFICHAGE CARTE ---
if st.session_state.last_res is not None:
    res = st.session_state.last_res
    
    # Capture du clic sur le bouton "Ajouter" via les données de la carte
    m = folium.Map(location=st.session_state.map_center, zoom_start=st.session_state.zoom_level)
    folium.TileLayer(tiles='https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}', attr='Google', name='Satellite', max_zoom=22).add_to(m)

    for i, (idx, row) in enumerate(res.iterrows()):
        lat, lon = row.geometry.centroid.y, row.geometry.centroid.x
        
        # Réintégration de TOUS les liens + Texte explicatif
        pop_html = f"""
        <div style='font-family:Arial; width:180px;'>
            <b>HAVEN #{i+1}</b><br>
            <small>Hameau de {int(row['taille_hameau'])} bât.</small><hr>
            <a href='https://www.google.com/maps/search/?api=1&query={lat},{lon}' target='_blank' style='color:#4285F4;text-decoration:none;'>🗺️ Google Maps</a><br>
            <a href='https://waze.com/ul?ll={lat},{lon}&navigate=yes' target='_blank' style='color:#33CCFF;text-decoration:none;'>🚙 Waze</a><br>
            <a href='https://www.google.com/maps/@?api=1&map_action=pano&viewpoint={lat},{lon}' target='_blank' style='color:#EA4335;text-decoration:none;'>🏙️ Street View</a><hr>
            <p style='font-size:10px;'>Pour ajouter aux favoris, utilisez le menu de sélection à gauche sous "Secteur".</p>
        </div>
        """
        icon_html = f'<div style="background-color:red;border:1px solid white;border-radius:50%;width:22px;height:22px;color:white;font-weight:bold;font-size:10px;display:flex;justify-content:center;align-items:center;">{i+1}</div>'
        folium.Marker([lat, lon], popup=folium.Popup(pop_html, max_width=200), icon=folium.DivIcon(html=icon_html)).add_to(m)
    
    # Rendu de la carte
    st_folium(m, width="100%", height=700, key=f"map_{st.session_state.map_center}", returned_objects=[])
    
    # Export CSV toujours présent
    csv = res[['taille_hameau', 'd_route']].assign(lat=res.geometry.centroid.y, lon=res.geometry.centroid.x).to_csv(index=False)
    st.download_button("📥 Export CSV", csv, "haven_radar.csv", "text/csv")
