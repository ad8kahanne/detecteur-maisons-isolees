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

# --- INITIALISATION DU TRI ---
if 'hidden_havens' not in st.session_state:
    st.session_state.hidden_havens = set()

# --- STYLE CSS ---
st.markdown("""
    <style>
    div.stButton > button:first-child {
        background-color: #28a745; color: white; height: 3em; width: 100%;
        border-radius: 5px; border: none; font-weight: bold;
    }
    .loading-text { font-weight: bold; font-size: 18px; margin-bottom: 10px; }
    </style>
""", unsafe_allow_html=True)

js_close_sidebar = "<script>parent.document.querySelector('button[kind=\"headerNoPadding\"]').click();</script>"

st.title("🏡 Haven Radar")

# --- BARRE LATÉRALE ---
with st.sidebar:
    st.header("Paramètres du Scan")
    commune_in = st.text_input("Secteur :", value="", placeholder="Tapez votre commune ici...", key="city_input")
    dist_route_val = st.slider("Distance Route (m) :", 30, 300, 70)
    rayon_iso_val = st.slider("Rayon Isolement (m) :", 50, 600, 180)
    voisins_max_val = st.number_input("Voisins Max autorisés :", 0, 12, 2)
    
    if st.button("Réinitialiser le tri"):
        st.session_state.hidden_havens = set()
        st.rerun()
        
    st.markdown("---")
    lancer_scan = st.button("Lancer le Scan")

# --- LOGIQUE DE SCAN ---
if lancer_scan or (st.session_state.city_input and st.session_state.get('last_run') != st.session_state.city_input):
    if not commune_in:
        st.error("⚠️ Veuillez entrer le nom d'une commune.")
    else:
        components.html(js_close_sidebar, height=0, width=0)
        try:
            if st.session_state.get('last_run') != commune_in:
                st.session_state.hidden_havens = set()
                st.session_state['last_run'] = commune_in

            loading_container = st.container()
            with loading_container:
                status_placeholder = st.empty()
                progress_bar = st.progress(0)
                
                # Etape 1
                status_placeholder.markdown('<p class="loading-text">🔍 Recherche en cours... 30%</p>', unsafe_allow_html=True)
                progress_bar.progress(30)
                base = ox.geocode_to_gdf(commune_in)
                geom_c = base.geometry.iloc[0]
                voisines = ox.features_from_polygon(geom_c.buffer(0.015), tags={'admin_level': '8'})
                secteur = pd.concat([base, voisines[voisines.geometry.intersects(geom_c)]]).to_crs(epsg=2154)
                union_zone = secteur.geometry.union_all()
                
                # Etape 2
                status_placeholder.markdown('<p class="loading-text">🔍 Recherche en cours... 60%</p>', unsafe_allow_html=True)
                progress_bar.progress(60)
                bbox = secteur.to_crs(epsg=4326).geometry.union_all().buffer(0.01) 
                bat = ox.features_from_polygon(bbox, tags={'building': True})
                routes = ox.features_from_polygon(bbox, tags={'highway': ['primary', 'secondary', 'tertiary', 'residential', 'unclassified', 'trunk']})
                try: autoroutes = ox.features_from_polygon(bbox, tags={'highway': 'motorway'})
                except: autoroutes = pd.DataFrame()

                # Etape 3
                status_placeholder.markdown('<p class="loading-text">🔍 Recherche en cours... 90%</p>', unsafe_allow_html=True)
                progress_bar.progress(90)
                bat = bat[bat.geometry.type.isin(['Polygon', 'MultiPolygon'])].copy().to_crs(epsg=2154)
                routes = routes.to_crs(epsg=2154)
                bat['d_route'] = bat.geometry.centroid.apply(lambda x: routes.distance(x).min())
                
                # Filtre autoroute & distance
                dist_auto = 350
                if not autoroutes.empty and 'geometry' in autoroutes.columns:
                    autoroutes = autoroutes.to_crs(epsg=2154)
                    bat['d_autoroute'] = bat.geometry.centroid.apply(lambda x: autoroutes.distance(x).min())
                    candidates = bat[bat.geometry.centroid.within(union_zone) & (bat['d_route'] >= dist_route_val) & (bat['d_autoroute'] > dist_auto)].copy()
                else:
                    candidates = bat[bat.geometry.centroid.within(union_zone) & (bat['d_route'] >= dist_route_val)].copy()

                if not candidates.empty:
                    coords_toutes = list(zip(bat.geometry.centroid.x, bat.geometry.centroid.y))
                    nn = NearestNeighbors(radius=rayon_iso_val).fit(coords_toutes)
                    adj = nn.radius_neighbors_graph(list(zip(candidates.geometry.centroid.x, candidates.geometry.centroid.y))).toarray()
                    candidates['nb_voisins'] = adj.sum(axis=1) - 1
                    res = candidates[candidates['nb_voisins'] <= voisins_max_val].copy().to_crs(epsg=4326)
                    
                    loading_container.empty()

                    if not res.empty:
                        st.success(f"✅ {len(res)} Havens détectés !")
                        m = folium.Map(location=[res.geometry.centroid.y.mean(), res.geometry.centroid.x.mean()], zoom_start=13)
                        folium.TileLayer(tiles='https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}', attr='Google', name='Satellite', max_zoom=22).add_to(m)

                        for i, (idx, row) in enumerate(res.iterrows()):
                            lat, lon = row.geometry.centroid.y, row.geometry.centroid.x
                            is_hidden = idx in st.session_state.hidden_havens
                            color = "#808080" if is_hidden else "red"
                            
                            pop_html = f"""
                            <div style='font-family:Arial; width:160px;'>
                                <b>HAVEN #{i+1}</b> {'(Masqué)' if is_hidden else ''}<hr>
                                <a href='https://www.google.com/maps/search/?api=1&query={lat},{lon}' target='_blank' style='color:#4285F4;text-decoration:none;display:block;margin-bottom:5px;'>🗺️ Maps</a>
                                <a href='https://waze.com/ul?ll={lat},{lon}&navigate=yes' target='_blank' style='color:#33CCFF;text-decoration:none;display:block;margin-bottom:5px;'>🚙 Waze</a>
                                <p style='font-size:10px; color:gray; margin-top:10px;'>ID: {idx}</p>
                            </div>
                            """
                            
                            icon_html = f'''<div style="background-color:{color}; border:2px solid white; border-radius:50%; width:25px; height:25px; color:white; font-weight:bold; font-size:12px; display:flex; justify-content:center; align-items:center; box-shadow:0px 0px 5px rgba(0,0,0,0.5);">{i+1}</div>'''
                            
                            folium.Marker(
                                [lat, lon], 
                                popup=folium.Popup(pop_html, max_width=250), 
                                icon=folium.DivIcon(html=icon_html)
                            ).add_to(m)
                        
                        # Affichage de la carte
                        output = st_folium(m, width="100%", height=600, key="main_map")
                        
                        # Logique pour masquer via clic (On utilise les données du dernier objet cliqué)
                        if output and output.get("last_object_clicked_popup"):
                            clicked_id = output["last_object_clicked_popup"].split("ID: ")[1].split("</p>")[0]
                            if clicked_id not in st.session_state.hidden_havens:
                                st.session_state.hidden_havens.add(clicked_id)
                                st.rerun()

                        st.info("💡 Pour masquer un Haven, cliquez sur sa pastille puis fermez la bulle.")
                else:
                    loading_container.empty()
                    st.warning("⚠️ Aucun bâtiment trouvé.")
        except Exception as e:
            st.error(f"❌ Erreur : {str(e)}")
