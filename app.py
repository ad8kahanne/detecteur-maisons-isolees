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
        background-color: #28a745;
        color: white;
        height: 3em;
        width: 100%;
        border-radius: 5px;
        border: none;
        font-weight: bold;
    }
    .loading-text {
        font-weight: bold;
        font-size: 18px;
        margin-bottom: 10px;
    }
    </style>
""", unsafe_allow_html=True)

js_close_sidebar = "<script>parent.document.querySelector('button[kind=\"headerNoPadding\"]').click();</script>"

st.title("🏡 Haven Radar")

# --- BARRE LATÉRALE ---
with st.sidebar:
    st.header("Paramètres du Scan")
    commune_in = st.text_input("Secteur :", placeholder="Tapez votre commune ici...", key="city_input")
    dist_route_val = st.slider("Distance Route (m) :", 30, 300, 70)
    rayon_iso_val = st.slider("Rayon Isolement (m) :", 50, 600, 180)
    taille_hameau_max = st.number_input("Nombre de bâtiments du hameau :", 1, 13, 3)
    
    st.markdown("---")
    lancer_scan = st.button("Lancer le Scan")
    if st.button("🗑️ Vider les favoris"):
        st.session_state.favs = {}
        st.rerun()

# --- LOGIQUE DE SCAN ---
if lancer_scan or (commune_in and st.session_state.get('last_city') != commune_in):
    if not commune_in:
        st.error("⚠️ Veuillez entrer le nom d'une commune.")
    else:
        components.html(js_close_sidebar, height=0, width=0)
        try:
            st.session_state['last_city'] = commune_in
            loading_container = st.container()
            with loading_container:
                status_placeholder = st.empty()
                progress_bar = st.progress(0)
                
                status_placeholder.markdown('<p class="loading-text">🔍 Recherche en cours... 15%</p>', unsafe_allow_html=True)
                progress_bar.progress(15)
                base = ox.geocode_to_gdf(commune_in)
                geom_c = base.geometry.iloc[0]
                voisines = ox.features_from_polygon(geom_c.buffer(0.015), tags={'admin_level': '8'})
                secteur = pd.concat([base, voisines[voisines.geometry.intersects(geom_c)]]).to_crs(epsg=2154)
                secteur = secteur[secteur.geometry.area < 200_000_000] 
                union_zone = secteur.geometry.union_all()
                
                status_placeholder.markdown('<p class="loading-text">🔍 Recherche en cours... 40%</p>', unsafe_allow_html=True)
                progress_bar.progress(40)
                bbox = secteur.to_crs(epsg=4326).geometry.union_all().buffer(0.01) 
                bat = ox.features_from_polygon(bbox, tags={'building': True})
                routes = ox.features_from_polygon(bbox, tags={'highway': ['primary', 'secondary', 'tertiary', 'residential', 'unclassified', 'trunk']})
                
                try: autoroutes = ox.features_from_polygon(bbox, tags={'highway': 'motorway'})
                except: autoroutes = pd.DataFrame()
                
                status_placeholder.markdown('<p class="loading-text">🔍 Recherche en cours... 70%</p>', unsafe_allow_html=True)
                progress_bar.progress(70)
                bat = bat[bat.geometry.type.isin(['Polygon', 'MultiPolygon'])].copy().to_crs(epsg=2154)
                routes = routes.to_crs(epsg=2154)
                bat['d_route'] = bat.geometry.centroid.apply(lambda x: routes.distance(x).min())
                
                if not autoroutes.empty and 'geometry' in autoroutes.columns:
                    autoroutes = autoroutes.to_crs(epsg=2154)
                    bat['d_auto'] = bat.geometry.centroid.apply(lambda x: autoroutes.distance(x).min())
                    candidates = bat[bat.geometry.centroid.within(union_zone) & (bat['d_route'] >= dist_route_val) & (bat['d_auto'] > 350)].copy()
                else:
                    candidates = bat[bat.geometry.centroid.within(union_zone) & (bat['d_route'] >= dist_route_val)].copy()
                
                status_placeholder.markdown('<p class="loading-text">🔍 Recherche en cours... 90%</p>', unsafe_allow_html=True)
                progress_bar.progress(90)
                
                if not candidates.empty:
                    coords_toutes = list(zip(bat.geometry.centroid.x, bat.geometry.centroid.y))
                    coords_candidates = list(zip(candidates.geometry.centroid.x, candidates.geometry.centroid.y))
                    nn = NearestNeighbors(radius=rayon_iso_val).fit(coords_toutes)
                    adj = nn.radius_neighbors_graph(coords_candidates).toarray()
                    
                    candidates['taille_hameau'] = adj.sum(axis=1)
                    st.session_state.last_res = candidates[candidates['taille_hameau'] <= taille_hameau_max].copy().to_crs(epsg=4326)
                    st.session_state.map_center = [st.session_state.last_res.geometry.centroid.y.mean(), st.session_state.last_res.geometry.centroid.x.mean()]
                    st.session_state.zoom_level = 13
                
                status_placeholder.empty()
                progress_bar.empty()
                loading_container.empty()
        except Exception as e:
            st.error(f"❌ Erreur : {str(e)}")

# --- AFFICHAGE CARTE ET FAVORIS ---
if st.session_state.last_res is not None:
    res = st.session_state.last_res
    
    # Gestion des favoris
    col_f1, col_f2 = st.columns([3, 1])
    with col_f2:
        to_fav = st.selectbox("Ajouter aux favoris :", range(len(res)), format_func=lambda x: f"Haven #{x+1}")
        if st.button("⭐ Ajouter"):
            row = res.iloc[to_fav]
            lat, lon = row.geometry.centroid.y, row.geometry.centroid.x
            st.session_state.favs[to_fav+1] = (lat, lon)

    # Liste des favoris
    if st.session_state.favs:
        st.write("📍 **Accès rapide :**")
        fav_cols = st.columns(12)
        for i, (num, coords) in enumerate(sorted(st.session_state.favs.items())):
            # On ajoute coords[0] (lat) dans la clé pour forcer le refresh si on reclique
            if fav_cols[i % 12].button(f"#{num}", key=f"fav_{num}_{coords[0]}"):
                st.session_state.map_center = [coords[0], coords[1]]
                st.session_state.zoom_level = 18 # Zoom fort pour le favori
                st.rerun()

    # Affichage de la Carte
    m = folium.Map(
        location=st.session_state.map_center, 
        zoom_start=st.session_state.zoom_level,
        control_scale=True
    )
    folium.TileLayer(tiles='https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}', attr='Google', name='Satellite', max_zoom=22).add_to(m)

    for i, (idx, row) in enumerate(res.iterrows()):
        lat, lon = row.geometry.centroid.y, row.geometry.centroid.x
        pop_html = f"<div style='font-family:Arial; width:160px;'><b>HAVEN #{i+1}</b><br><small>Hameau de {int(row['taille_hameau'])} bât.</small><hr><a href='https://www.google.com/maps/search/?api=1&query={lat},{lon}' target='_blank' style='color:#4285F4;display:block;'>🗺️ Maps</a><a href='https://waze.com/ul?ll={lat},{lon}&navigate=yes' target='_blank' style='color:#33CCFF;display:block;'>🚙 Waze</a></div>"
        icon_html = f'<div style="background-color:red;border:2px solid white;border-radius:50%;width:25px;height:25px;color:white;font-weight:bold;font-size:12px;display:flex;justify-content:center;align-items:center;">{i+1}</div>'
        folium.Marker([lat, lon], popup=folium.Popup(pop_html, max_width=250), icon=folium.DivIcon(html=icon_html)).add_to(m)
    
    # On vide returned_objects pour plus de légèreté
    st_folium(m, width="100%", height=600, key=f"map_{st.session_state.map_center}", returned_objects=[])
    
    csv = res[['taille_hameau', 'd_route']].assign(lat=res.geometry.centroid.y, lon=res.geometry.centroid.x).to_csv(index=False)
    st.download_button("📥 Télécharger CSV", csv, "haven_radar.csv", "text/csv")
