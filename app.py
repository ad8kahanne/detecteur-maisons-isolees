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

# --- INITIALISATION MÉMOIRE ---
if 'hidden_havens' not in st.session_state:
    st.session_state.hidden_havens = set()
if 'scan_results' not in st.session_state:
    st.session_state.scan_results = None

# --- STYLE CSS ---
st.markdown("""
    <style>
    div.stButton > button:first-child {
        background-color: #28a745; color: white; height: 3em; width: 100%;
        border-radius: 5px; border: none; font-weight: bold;
    }
    .loading-text { font-weight: bold; font-size: 18px; margin-bottom: 10px; }
    .btn-mask { 
        display: inline-block; padding: 5px 10px; background-color: #f0f2f6; 
        color: #31333F; text-decoration: none; border-radius: 5px; 
        font-size: 11px; font-weight: bold; border: 1px solid #d1d5db;
        margin-top: 10px; cursor: pointer;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🏡 Haven Radar")

# --- BARRE LATÉRALE ---
with st.sidebar:
    st.header("Paramètres")
    commune_in = st.text_input("Secteur :", key="city_input")
    dist_route_val = st.slider("Distance Route (m) :", 30, 300, 70)
    rayon_iso_val = st.slider("Rayon Isolement (m) :", 50, 600, 180)
    voisins_max_val = st.number_input("Voisins Max :", 0, 12, 2)
    
    col1, col2 = st.columns(2)
    with col1:
        lancer_scan = st.button("Lancer le Scan")
    with col2:
        if st.button("🔄 Reset Tri"):
            st.session_state.hidden_havens = set()
            st.rerun()

# --- LOGIQUE DE SCAN ---
if lancer_scan:
    if not commune_in:
        st.error("⚠️ Entrez une commune.")
    else:
        try:
            loading = st.container()
            with loading:
                p_bar = st.progress(0)
                st.markdown('<p class="loading-text">🔍 Scan en cours... 40%</p>', unsafe_allow_html=True)
                p_bar.progress(40)
                
                base = ox.geocode_to_gdf(commune_in)
                geom_c = base.geometry.iloc[0]
                bbox = base.to_crs(epsg=4326).geometry.union_all().buffer(0.01)
                
                bat = ox.features_from_polygon(bbox, tags={'building': True})
                routes = ox.features_from_polygon(bbox, tags={'highway': ['primary', 'secondary', 'tertiary', 'residential', 'unclassified', 'trunk']})
                try: auto = ox.features_from_polygon(bbox, tags={'highway': 'motorway'})
                except: auto = pd.DataFrame()

                st.markdown('<p class="loading-text">🔍 Scan en cours... 80%</p>', unsafe_allow_html=True)
                p_bar.progress(80)
                bat = bat[bat.geometry.type.isin(['Polygon', 'MultiPolygon'])].copy().to_crs(epsg=2154)
                routes = routes.to_crs(epsg=2154)
                bat['d_route'] = bat.geometry.centroid.apply(lambda x: routes.distance(x).min())
                
                if not auto.empty and 'geometry' in auto.columns:
                    auto = auto.to_crs(epsg=2154)
                    bat['d_auto'] = bat.geometry.centroid.apply(lambda x: auto.distance(x).min())
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
                loading.empty()
        except Exception as e:
            st.error(f"❌ Erreur : {e}")

# --- AFFICHAGE ---
if st.session_state.scan_results is not None:
    df = st.session_state.scan_results
    if df.empty:
        st.warning("⚠️ Aucun résultat.")
    else:
        st.success(f"✅ {len(df)} Havens détectés")
        m = folium.Map(location=[df.geometry.centroid.y.mean(), df.geometry.centroid.x.mean()], zoom_start=13)
        folium.TileLayer(tiles='https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}', attr='Google', name='Satellite', max_zoom=22).add_to(m)

        for i, (idx, row) in enumerate(df.iterrows()):
            lat, lon = row.geometry.centroid.y, row.geometry.centroid.x
            is_hidden = str(idx) in st.session_state.hidden_havens
            color = "#A0A0A0" if is_hidden else "#FF0000"
            
            pop_content = f"""
            <div style='font-family:Arial; width:150px;'>
                <b>HAVEN #{i+1}</b><hr>
                <a href='http://maps.google.com/?q={lat},{lon}' target='_blank' style='color:#4285F4;display:block;margin:5px 0;'>🗺️ Google Maps</a>
                <a href='https://waze.com/ul?ll={lat},{lon}&navigate=yes' target='_blank' style='color:#33CCFF;display:block;margin:5px 0;'>🚙 Waze</a>
                <div style='text-align:center;'>
                    <p style='display:none;'>ID:{idx}</p>
                    <span style='color:gray; font-size:9px;'>Cliquer sur la pastille pour griser</span>
                </div>
            </div>
            """
            
            icon_html = f'''<div style="background-color:{color}; border:2px solid white; border-radius:50%; width:24px; height:24px; color:white; font-weight:bold; font-size:11px; display:flex; justify-content:center; align-items:center; box-shadow:0 2px 4px rgba(0,0,0,0.3);">{i+1}</div>'''
            
            folium.Marker([lat, lon], popup=folium.Popup(pop_content, max_width=200), icon=folium.DivIcon(html=icon_html)).add_to(m)
        
        # Interaction
        map_data = st_folium(m, width="100%", height=600, key="map_haven")
        
        # Détection du clic pour griser
        if map_data and map_data.get("last_object_clicked_popup"):
            try:
                raw_id = map_data["last_object_clicked_popup"].split("ID:")[1].split("</p>")[0]
                if raw_id not in st.session_state.hidden_havens:
                    st.session_state.hidden_havens.add(raw_id)
                    st.rerun()
            except: pass
