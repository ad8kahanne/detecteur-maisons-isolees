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

# --- MÉMOIRE ---
if 'hidden_havens' not in st.session_state:
    st.session_state.hidden_havens = set()
if 'scan_results' not in st.session_state:
    st.session_state.scan_results = None

# --- STYLE ---
st.markdown("""
    <style>
    div.stButton > button:first-child { background-color: #28a745; color: white; font-weight: bold; width: 100%; }
    .btn-action { 
        display: block; padding: 6px; margin-top: 5px; text-align: center;
        border-radius: 4px; text-decoration: none; font-size: 11px; font-weight: bold;
    }
    .btn-griser { background-color: #e0e0e0; color: #333; border: 1px solid #ccc; }
    .btn-retablir { background-color: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
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
            with st.spinner("Analyse en cours..."):
                base = ox.geocode_to_gdf(commune_in)
                bbox = base.to_crs(epsg=4326).geometry.union_all().buffer(0.01)
                bat = ox.features_from_polygon(bbox, tags={'building': True})
                routes = ox.features_from_polygon(bbox, tags={'highway': ['primary', 'secondary', 'tertiary', 'residential', 'unclassified', 'trunk']})
                try: auto = ox.features_from_polygon(bbox, tags={'highway': 'motorway'})
                except: auto = pd.DataFrame()

                bat = bat[bat.geometry.type.isin(['Polygon', 'MultiPolygon'])].copy().to_crs(epsg=2154)
                routes = routes.to_crs(epsg=2154)
                bat['d_route'] = bat.geometry.centroid.apply(lambda x: routes.distance(x).min())
                
                if not auto.empty and 'geometry' in auto.columns:
                    auto = auto.to_crs(epsg=2154)
                    bat['d_auto'] = bat.geometry.centroid.apply(lambda x: auto.distance(x).min())
                    cand = bat[(bat['d_route'] >= dist_route_val) & (bat['d_auto'] > 350)].copy()
                else:
                    cand = bat[bat['d_route'] >= dist_route_val].copy()

                nn = NearestNeighbors(radius=rayon_iso_val).fit(list(zip(bat.geometry.centroid.x, bat.geometry.centroid.y)))
                adj = nn.radius_neighbors_graph(list(zip(cand.geometry.centroid.x, cand.geometry.centroid.y))).toarray()
                cand['nb_voisins'] = adj.sum(axis=1) - 1
                st.session_state.scan_results = cand[cand['nb_voisins'] <= voisins_max_val].copy().to_crs(epsg=4326)
        except Exception as e:
            st.error(f"Erreur : {e}")

# --- AFFICHAGE CARTE ---
if st.session_state.scan_results is not None:
    df = st.session_state.scan_results
    m = folium.Map(location=[df.geometry.centroid.y.mean(), df.geometry.centroid.x.mean()], zoom_start=13)
    folium.TileLayer(tiles='https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}', attr='Google', name='Satellite', max_zoom=22).add_to(m)

    for idx, row in df.iterrows():
        lat, lon = row.geometry.centroid.y, row.geometry.centroid.x
        id_str = str(idx)
        is_hidden = id_str in st.session_state.hidden_havens
        color = "#808080" if is_hidden else "#FF0000"
        
        # Popup avec les nouveaux boutons
        pop_html = f"""
        <div style='font-family:Arial; width:140px; line-height:1.5;'>
            <b>HAVEN</b><hr style='margin:5px 0;'>
            <a href='https://www.google.com/maps?q={lat},{lon}' target='_blank' style='color:#4285F4; text-decoration:none;'>🗺️ Google Maps</a><br>
            <a href='https://waze.com/ul?ll={lat},{lon}&navigate=yes' target='_blank' style='color:#33CCFF; text-decoration:none;'>🚙 Waze</a>
            <div style='margin-top:10px; border-top: 1px solid #eee; padding-top:5px;'>
                <p style='display:none;'>ID:{id_str}</p>
                <a href='#' class='btn-action btn-griser'>🔘 Griser</a>
                <a href='#' class='btn-action btn-retablir'>🔄 Rétablir</a>
            </div>
        </div>
        """
        
        icon_html = f'''<div style="background-color:{color}; border:2px solid white; border-radius:50%; width:22px; height:22px; color:white; font-weight:bold; font-size:10px; display:flex; justify-content:center; align-items:center; box-shadow:0 2px 4px rgba(0,0,0,0.3);">{idx % 100}</div>'''
        folium.Marker([lat, lon], popup=folium.Popup(pop_html, max_width=200), icon=folium.DivIcon(html=icon_html)).add_to(m)

    # Affichage stable
    map_data = st_folium(m, width="100%", height=600, key="stable_map")

    # Logique de clic sur bouton (via détection de popup)
    if map_data and map_data.get("last_object_clicked_popup"):
        clicked_text = map_data["last_object_clicked_popup"]
        try:
            target_id = clicked_text.split("ID:")[1].split("</p>")[0]
            if "btn-griser" in clicked_text and target_id not in st.session_state.hidden_havens:
                st.session_state.hidden_havens.add(target_id)
                st.rerun()
            elif "btn-retablir" in clicked_text and target_id in st.session_state.hidden_havens:
                st.session_state.hidden_havens.remove(target_id)
                st.rerun()
        except: pass
