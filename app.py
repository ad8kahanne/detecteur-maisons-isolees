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

# --- STYLE CSS & JAVASCRIPT ---
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
    [data-testid="stSidebar"] {
        box-shadow: 2px 0px 10px rgba(0,0,0,0.2);
    }
    .loading-text {
        font-weight: bold;
        font-size: 18px;
        margin-bottom: 10px;
    }
    </style>
""", unsafe_allow_html=True)

js_close_sidebar = """
<script>
    parent.document.querySelector('button[kind="headerNoPadding"]').click();
</script>
"""

st.title("🏡 Haven Radar")

# --- BARRE LATÉRALE ---
with st.sidebar:
    st.header("Paramètres du Scan")
    commune_in = st.text_input("Secteur :", value="", placeholder="Tapez votre commune ici...", key="city_input")
    dist_route_val = st.slider("Distance Route (m) :", 30, 300, 70)
    rayon_iso_val = st.slider("Rayon Isolement (m) :", 50, 600, 180)
    
    # MODIFICATION : On demande la taille totale du hameau (minimum 1)
    taille_hameau_max = st.number_input("Nombre de bâtiments du hameau :", 1, 13, 3)
    
    st.markdown("---")
    lancer_scan = st.button("Lancer le Scan")

# --- LOGIQUE DE DÉCLENCHEMENT ---
if lancer_scan or (st.session_state.city_input and st.session_state.get('last_run') != st.session_state.city_input):
    if not commune_in:
        st.error("⚠️ Veuillez entrer le nom d'une commune.")
    else:
        components.html(js_close_sidebar, height=0, width=0)
        
        try:
            st.session_state['last_run'] = commune_in
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
                tags_routes = ['primary', 'secondary', 'tertiary', 'residential', 'unclassified', 'trunk']
                routes = ox.features_from_polygon(bbox, tags={'highway': tags_routes})
                
                try:
                    autoroutes = ox.features_from_polygon(bbox, tags={'highway': 'motorway'})
                except:
                    autoroutes = pd.DataFrame()
                
                status_placeholder.markdown('<p class="loading-text">🔍 Recherche en cours... 70%</p>', unsafe_allow_html=True)
                progress_bar.progress(70)
                bat = bat[bat.geometry.type.isin(['Polygon', 'MultiPolygon'])].copy().to_crs(epsg=2154)
                routes = routes.to_crs(epsg=2154)
                bat['d_route'] = bat.geometry.centroid.apply(lambda x: routes.distance(x).min())
                
                if not autoroutes.empty and 'geometry' in autoroutes.columns:
                    autoroutes = autoroutes.to_crs(epsg=2154)
                    bat['d_autoroute'] = bat.geometry.centroid.apply(lambda x: autoroutes.distance(x).min())
                    candidates = bat[bat.geometry.centroid.within(union_zone) & (bat['d_route'] >= dist_route_val) & (bat['d_autoroute'] > 350)].copy()
                else:
                    candidates = bat[bat.geometry.centroid.within(union_zone) & (bat['d_route'] >= dist_route_val)].copy()
                
                status_placeholder.markdown('<p class="loading-text">🔍 Recherche en cours... 90%</p>', unsafe_allow_html=True)
                progress_bar.progress(90)
                
                if not candidates.empty:
                    coords_toutes = list(zip(bat.geometry.centroid.x, bat.geometry.centroid.y))
                    coords_candidates = list(zip(candidates.geometry.centroid.x, candidates.geometry.centroid.y))
                    nn = NearestNeighbors(radius=rayon_iso_val).fit(coords_toutes)
                    adj = nn.radius_neighbors_graph(coords_candidates).toarray()
                    
                    # LOGIQUE CORRIGÉE : nb_voisins + 1 = Taille totale du hameau
                    candidates['taille_hameau'] = adj.sum(axis=1)
                    res = candidates[candidates['taille_hameau'] <= taille_hameau_max].copy().to_crs(epsg=4326)
                    
                    status_placeholder.empty()
                    progress_bar.empty()
                    loading_container.empty()

                    if res.empty:
                        st.warning("⚠️ Aucun résultat trouvé.")
                    else:
                        st.success(f"✅ {len(res)} Havens détectés !")
                        m = folium.Map(location=[res.geometry.centroid.y.mean(), res.geometry.centroid.x.mean()], zoom_start=13)
                        folium.TileLayer(tiles='https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}', attr='Google', name='Satellite', max_zoom=22).add_to(m)

                        for i, (idx, row) in enumerate(res.iterrows()):
                            lat, lon = row.geometry.centroid.y, row.geometry.centroid.x
                            u_google = f"https://www.google.com/maps/search/?api=1&query={lat},{lon}"
                            u_waze = f"https://waze.com/ul?ll={lat},{lon}&navigate=yes"
                            u_sv = f"https://www.google.com/maps/@?api=1&map_action=pano&viewpoint={lat},{lon}"
                            pop_html = f"<div style='font-family:Arial; width:160px;'><b>HAVEN #{i+1}</b><br><small>Hameau de {int(row['taille_hameau'])} bât.</small><hr><a href='{u_google}' target='_blank' style='color:#4285F4;display:block;'>🗺️ Maps</a><a href='{u_waze}' target='_blank' style='color:#33CCFF;display:block;'>🚙 Waze</a><a href='{u_sv}' target='_blank' style='color:#EA4335;display:block;'>🏙️ Street View</a></div>"
                            icon_html = f'<div style="background-color:red;border:2px solid white;border-radius:50%;width:25px;height:25px;color:white;font-weight:bold;font-size:12px;display:flex;justify-content:center;align-items:center;">{i+1}</div>'
                            folium.Marker([lat, lon], popup=folium.Popup(pop_html, max_width=250), icon=folium.DivIcon(html=icon_html)).add_to(m)
                        
                        st_folium(m, width="100%", height=600, returned_objects=[])
                        csv = res[['taille_hameau', 'd_route']].assign(lat=res.geometry.centroid.y, lon=res.geometry.centroid.x).to_csv(index=False)
                        st.download_button("📥 Télécharger CSV", csv, "haven_radar.csv", "text/csv")
                else:
                    loading_container.empty()
                    st.warning("⚠️ Aucun bâtiment ne correspond aux critères.")
        except Exception as e:
            st.error(f"❌ Erreur : {str(e)}")
