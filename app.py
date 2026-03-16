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
st.set_page_config(page_title="HAVEN RADAR", layout="wide")

# --- INITIALISATION MÉMOIRE ---
if 'hidden_havens' not in st.session_state:
    st.session_state.hidden_havens = set()
if 'scan_results' not in st.session_state:
    st.session_state.scan_results = None

# --- STYLE CSS ---
st.markdown("""
    <style>
    div.stButton > button:first-child { background-color: #28a745; color: white; font-weight: bold; width: 100%; height: 3em; border-radius: 5px; }
    .loading-text { font-weight: bold; font-size: 18px; margin-bottom: 10px; }
    .btn-popup { display: block; text-align: center; padding: 5px; margin-top: 8px; border-radius: 4px; text-decoration: none; font-weight: bold; font-size: 11px; color: white !important; }
    .bg-gray { background-color: #6c757d; }
    .bg-red { background-color: #dc3545; }
    </style>
""", unsafe_allow_html=True)

st.title("🏡 Haven Radar")

# --- BARRE LATÉRALE ---
with st.sidebar:
    st.header("Paramètres du Scan")
    commune_in = st.text_input("Secteur :", key="city_input")
    dist_route_val = st.slider("Distance Route (m) :", 30, 300, 70)
    rayon_iso_val = st.slider("Rayon Isolement (m) :", 50, 600, 180)
    voisins_max_val = st.number_input("Voisins Max autorisés :", 0, 12, 2)
    st.markdown("---")
    lancer_scan = st.button("Lancer le Scan")

# --- LOGIQUE DE SCAN ---
if lancer_scan:
    if not commune_in:
        st.error("⚠️ Veuillez entrer une commune.")
    else:
        try:
            loading_container = st.container()
            with loading_container:
                status_placeholder = st.empty()
                progress_bar = st.progress(0)
                
                # Étape 1 : 30%
                status_placeholder.markdown('<p class="loading-text">🔍 Recherche en cours... 30%</p>', unsafe_allow_html=True)
                progress_bar.progress(30)
                base = ox.geocode_to_gdf(commune_in)
                geom_c = base.geometry.iloc[0]
                # Zone large pour ne rien rater
                bbox = base.to_crs(epsg=4326).geometry.union_all().buffer(0.02)
                
                # Étape 2 : 60%
                status_placeholder.markdown('<p class="loading-text">🔍 Recherche en cours... 60%</p>', unsafe_allow_html=True)
                progress_bar.progress(60)
                bat = ox.features_from_polygon(bbox, tags={'building': True})
                routes = ox.features_from_polygon(bbox, tags={'highway': ['primary', 'secondary', 'tertiary', 'residential', 'unclassified', 'trunk']})
                try: 
                    autoroutes = ox.features_from_polygon(bbox, tags={'highway': 'motorway'})
                except: 
                    autoroutes = pd.DataFrame()
                
                # Étape 3 : 90%
                status_placeholder.markdown('<p class="loading-text">🔍 Recherche en cours... 90%</p>', unsafe_allow_html=True)
                progress_bar.progress(90)
                bat = bat[bat.geometry.type.isin(['Polygon', 'MultiPolygon'])].copy().to_crs(epsg=2154)
                routes = routes.to_crs(epsg=2154)
                bat['d_route'] = bat.geometry.centroid.apply(lambda x: routes.distance(x).min())
                
                # Exclusion Autoroute 350m
                if not autoroutes.empty and 'geometry' in autoroutes.columns:
                    autoroutes = autoroutes.to_crs(epsg=2154)
                    bat['d_auto'] = bat.geometry.centroid.apply(lambda x: autoroutes.distance(x).min())
                    cand = bat[(bat['d_route'] >= dist_route_val) & (bat['d_auto'] > 350)].copy()
                else:
                    cand = bat[bat['d_route'] >= dist_route_val].copy()

                if not cand.empty:
                    nn = NearestNeighbors(radius=rayon_iso_val).fit(list(zip(bat.geometry.centroid.x, bat.geometry.centroid.y)))
                    adj = nn.radius_neighbors_graph(list(zip(cand.geometry.centroid.x, cand.geometry.centroid.y))).toarray()
                    cand['nb_voisins'] = adj.sum(axis=1) - 1
                    st.session_state.scan_results = cand[cand['nb_voisins'] <= voisins_max_val].copy().to_crs(epsg=4326)
                else:
                    st.session_state.scan_results = pd.DataFrame()
                
                loading_container.empty()
        except Exception as e:
            st.error(f"❌ Erreur : {e}")

# --- AFFICHAGE DE LA CARTE ---
if st.session_state.scan_results is not None:
    res = st.session_state.scan_results
    if res.empty:
        st.warning("⚠️ Aucun résultat trouvé.")
    else:
        st.success(f"✅ {len(res)} Havens détectés !")
        m = folium.Map(location=[res.geometry.centroid.y.mean(), res.geometry.centroid.x.mean()], zoom_start=13)
        folium.TileLayer(tiles='https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}', attr='Google', name='Satellite', max_zoom=22).add_to(m)

        for i, (idx, row) in enumerate(res.iterrows()):
            lat, lon = row.geometry.centroid.y, row.geometry.centroid.x
            id_str = str(idx)
            is_hidden = id_str in st.session_state.hidden_havens
            color = "#808080" if is_hidden else "red"
            
            # Popup avec boutons Griser / Rétablir
            btn_label = "🔄 Rétablir" if is_hidden else "🔘 Griser"
            btn_class = "bg-red" if is_hidden else "bg-gray"
            
            pop_html = f"""
            <div style='font-family:Arial; width:150px;'>
                <b>HAVEN #{i+1}</b><hr>
                <a href='http://maps.google.com/?q={lat},{lon}' target='_blank' style='color:#4285F4;display:block;margin-bottom:5px;'>🗺️ Google Maps</a>
                <a href='https://waze.com/ul?ll={lat},{lon}&navigate=yes' target='_blank' style='color:#33CCFF;display:block;'>🚙 Waze</a>
                <p style='display:none;'>ID:{id_str}</p>
                <a href='#' class='btn-popup {btn_class}'>{btn_label}</a>
            </div>
            """
            
            icon_html = f'<div style="background-color:{color}; border:2px solid white; border-radius:50%; width:24px; height:24px; color:white; font-weight:bold; font-size:11px; display:flex; justify-content:center; align-items:center; box-shadow:0 2px 4px rgba(0,0,0,0.3);">{i+1}</div>'
            folium.Marker([lat, lon], popup=folium.Popup(pop_html, max_width=200), icon=folium.DivIcon(html=icon_html)).add_to(m)

        # Affichage carte plein écran
        out = st_folium(m, width="100%", height=700, key="haven_map")

        # Logique de clic sur bouton dans popup
        if out and out.get("last_object_clicked_popup"):
            clicked_text = out["last_object_clicked_popup"]
            try:
                target_id = clicked_text.split("ID:")[1].split("</a>")[0].split("</p>")[0]
                if "Griser" in clicked_text:
                    st.session_state.hidden_havens.add(target_id)
                    st.rerun()
                elif "Rétablir" in clicked_text:
                    st.session_state.hidden_havens.remove(target_id)
                    st.rerun()
            except: pass
