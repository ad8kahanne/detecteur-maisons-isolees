import streamlit as st
import osmnx as ox
import folium
from streamlit_folium import st_folium
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import warnings

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
    div.stButton > button:first-child { background-color: #28a745; color: white; font-weight: bold; width: 100%; }
    /* Style des boutons dans la bulle */
    .btn-popup {
        display: inline-block;
        padding: 5px 10px;
        margin: 4px 0;
        border-radius: 4px;
        text-decoration: none;
        font-size: 11px;
        font-weight: bold;
        text-align: center;
        width: 100%;
    }
    .btn-gray { background-color: #6c757d; color: white !important; }
    .btn-red { background-color: #dc3545; color: white !important; }
    </style>
""", unsafe_allow_html=True)

st.title("🏡 Haven Radar")

# --- SIDEBAR ---
with st.sidebar:
    st.header("Paramètres")
    commune_in = st.text_input("Secteur :", key="city_input")
    dist_route_val = st.slider("Distance Route (m) :", 30, 300, 70)
    rayon_iso_val = st.slider("Rayon Isolement (m) :", 50, 600, 180)
    voisins_max_val = st.number_input("Voisins Max :", 0, 12, 2)
    
    if st.button("Lancer le Scan"):
        try:
            # Barre de progression simple (pourcentage seulement)
            p_bar = st.progress(0)
            st.write("Chargement... 20%")
            p_bar.progress(20)
            
            base = ox.geocode_to_gdf(commune_in)
            bbox = base.to_crs(epsg=4326).geometry.union_all().buffer(0.01)
            
            st.write("Chargement... 50%")
            p_bar.progress(50)
            bat = ox.features_from_polygon(bbox, tags={'building': True})
            routes = ox.features_from_polygon(bbox, tags={'highway': True})
            
            # Correction erreur "No matching features"
            try:
                auto = ox.features_from_polygon(bbox, tags={'highway': 'motorway'})
            except:
                auto = pd.DataFrame()

            st.write("Chargement... 90%")
            p_bar.progress(90)
            bat = bat[bat.geometry.type.isin(['Polygon', 'MultiPolygon'])].copy().to_crs(epsg=2154)
            routes = routes.to_crs(epsg=2154)
            bat['d_route'] = bat.geometry.centroid.apply(lambda x: routes.distance(x).min())
            
            if not auto.empty:
                auto = auto.to_crs(epsg=2154)
                bat['d_auto'] = bat.geometry.centroid.apply(lambda x: auto.distance(x).min())
                cand = bat[(bat['d_route'] >= dist_route_val) & (bat['d_auto'] > 350)].copy()
            else:
                cand = bat[bat['d_route'] >= dist_route_val].copy()

            nn = NearestNeighbors(radius=rayon_iso_val).fit(list(zip(bat.geometry.centroid.x, bat.geometry.centroid.y)))
            adj = nn.radius_neighbors_graph(list(zip(cand.geometry.centroid.x, cand.geometry.centroid.y))).toarray()
            cand['nb_voisins'] = adj.sum(axis=1) - 1
            st.session_state.scan_results = cand[cand['nb_voisins'] <= voisins_max_val].copy().to_crs(epsg=4326)
            p_bar.empty()
        except Exception as e:
            st.error(f"Erreur : {e}")

# --- GESTION DU TRI (HORS CARTE POUR STABILITÉ) ---
col_tri1, col_tri2 = st.columns([2, 1])
if st.session_state.scan_results is not None:
    with col_tri2:
        st.subheader("Gestion du Tri")
        # On utilise des boutons Streamlit natifs pour griser/dégriser
        # car les boutons dans Folium provoquent le bug de disparition.
        target_id = st.selectbox("Sélectionner Haven #", 
                               options=range(len(st.session_state.scan_results)),
                               format_func=lambda x: f"Haven #{x+1}")
        
        id_real = str(st.session_state.scan_results.index[target_id])
        
        if id_real in st.session_state.hidden_havens:
            if st.button("🔄 Rétablir en rouge"):
                st.session_state.hidden_havens.remove(id_real)
                st.rerun()
        else:
            if st.button("🔘 Griser la pastille"):
                st.session_state.hidden_havens.add(id_real)
                st.rerun()

    with col_tri1:
        df = st.session_state.scan_results
        m = folium.Map(location=[df.geometry.centroid.y.mean(), df.geometry.centroid.x.mean()], zoom_start=13)
        folium.TileLayer(tiles='https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}', attr='Google', name='Satellite').add_to(m)

        for i, (idx, row) in enumerate(df.iterrows()):
            lat, lon = row.geometry.centroid.y, row.geometry.centroid.x
            is_hidden = str(idx) in st.session_state.hidden_havens
            color = "#808080" if is_hidden else "#FF0000"
            
            pop_html = f"""
            <div style='font-family:Arial; width:130px;'>
                <b>HAVEN #{i+1}</b><hr>
                <a href='http://maps.google.com/?q={lat},{lon}' target='_blank'>🗺️ Google Maps</a><br>
                <a href='https://waze.com/ul?ll={lat},{lon}&navigate=yes' target='_blank'>🚙 Waze</a><br>
                <p style='font-size:9px; color:gray; margin-top:5px;'>Utilisez le menu à droite pour griser.</p>
            </div>
            """
            
            icon_html = f'''<div style="background-color:{color}; border:2px solid white; border-radius:50%; width:22px; height:22px; color:white; font-weight:bold; font-size:10px; display:flex; justify-content:center; align-items:center;">{i+1}</div>'''
            folium.Marker([lat, lon], popup=folium.Popup(pop_html, max_width=200), icon=folium.DivIcon(html=icon_html)).add_to(m)

        st_folium(m, width=700, height=500, key="map_stable")
