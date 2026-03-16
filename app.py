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
st.set_page_config(page_title="HAVEN RADAR", layout="wide", initial_sidebar_state="expanded")

# --- DESIGN & CSS ---
st.markdown("""
    <style>
    /* Bouton Lancer le Scan */
    div.stButton > button:first-child {
        background-color: #28a745;
        color: white;
        height: 3.5em;
        width: 100%;
        border-radius: 8px;
        border: none;
        font-weight: bold;
        font-size: 18px;
        transition: 0.3s;
    }
    div.stButton > button:hover {
        background-color: #218838;
        box-shadow: 0 4px 15px rgba(40, 167, 69, 0.3);
    }
    /* Look Sidebar */
    [data-testid="stSidebar"] {
        background-color: #f8f9fa;
        box-shadow: 2px 0px 10px rgba(0,0,0,0.1);
    }
    /* Boutons dans les Popups Folium */
    .btn-gps {
        display: block;
        text-align: center;
        padding: 8px;
        margin: 5px 0;
        border-radius: 5px;
        color: white !important;
        text-decoration: none;
        font-weight: bold;
        font-size: 13px;
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
    st.header("⚙️ Paramètres")
    commune_in = st.text_input("Secteur de recherche :", value="", placeholder="Ex: Turenne, Corrèze...", key="city_input")
    
    st.markdown("---")
    dist_route_val = st.slider("Distance Route standard (m) :", 30, 300, 70)
    rayon_iso_val = st.slider("Rayon Isolement (m) :", 50, 600, 180)
    voisins_max_val = st.number_input("Voisins Max autorisés :", 0, 12, 2)
    
    st.markdown("---")
    lancer_scan = st.button("🚀 LANCER LE RADAR")

# --- LOGIQUE DE DÉCLENCHEMENT ---
if lancer_scan:
    if not commune_in:
        st.error("⚠️ Veuillez entrer le nom d'une commune.")
    else:
        components.html(js_close_sidebar, height=0, width=0)
        
        try:
            # Zone d'information dynamique
            status = st.status("🔍 Initialisation du scan...", expanded=True)
            
            # 1. Localisation
            status.write("📍 Localisation de la zone...")
            base = ox.geocode_to_gdf(commune_in)
            geom_c = base.geometry.iloc[0]
            voisines = ox.features_from_polygon(geom_c.buffer(0.015), tags={'admin_level': '8'})
            secteur = pd.concat([base, voisines[voisines.geometry.intersects(geom_c)]]).to_crs(epsg=2154)
            secteur = secteur[secteur.geometry.area < 200_000_000] 
            union_zone = secteur.geometry.union_all()
            
            # 2. OSM Data
            status.write("🛰️ Récupération des bâtiments et routes (Standard + Autoroutes)...")
            bbox = secteur.to_crs(epsg=4326).geometry.union_all().buffer(0.01) 
            bat = ox.features_from_polygon(bbox, tags={'building': True})
            tags_routes = ['primary', 'secondary', 'tertiary', 'residential', 'unclassified', 'trunk']
            routes = ox.features_from_polygon(bbox, tags={'highway': tags_routes})
            autoroutes = ox.features_from_polygon(bbox, tags={'highway': 'motorway'})
            
            # 3. Analyse Spatiale
            status.write("📐 Calcul des distances (Filtrage zone calme)...")
            bat = bat[bat.geometry.type.isin(['Polygon', 'MultiPolygon'])].copy().to_crs(epsg=2154)
            routes = routes.to_crs(epsg=2154)
            bat['d_route'] = bat.geometry.centroid.apply(lambda x: routes.distance(x).min())
            
            if not autoroutes.empty:
                autoroutes = autoroutes.to_crs(epsg=2154)
                bat['d_autoroute'] = bat.geometry.centroid.apply(lambda x: autoroutes.distance(x).min())
                candidates = bat[
                    bat.geometry.centroid.within(union_zone) & 
                    (bat['d_route'] >= dist_route_val) & 
                    (bat['d_autoroute'] > 350)
                ].copy()
            else:
                candidates = bat[bat.geometry.centroid.within(union_zone) & (bat['d_route'] >= dist_route_val)].copy()
            
            # 4. Voisinage
            if not candidates.empty:
                status.write("🧪 Analyse finale de la densité de voisinage...")
                coords_toutes = list(zip(bat.geometry.centroid.x, bat.geometry.centroid.y))
                coords_candidates = list(zip(candidates.geometry.centroid.x, candidates.geometry.centroid.y))
                nn = NearestNeighbors(radius=rayon_iso_val).fit(coords_toutes)
                adj = nn.radius_neighbors_graph(coords_candidates).toarray()
                candidates['nb_voisins'] = adj.sum(axis=1) - 1
                res = candidates[candidates['nb_voisins'] <= voisins_max_val].copy().to_crs(epsg=4326)
                
                status.update(label="✅ Scan terminé !", state="complete", expanded=False)

                if res.empty:
                    st.warning("⚠️ Aucun résultat ne respecte les critères.")
                else:
                    # AFFICHAGE DES RÉSULTATS
                    m1, m2 = st.columns([1, 1])
                    m1.metric("Havens détectés", len(res))
                    
                    csv = res[['nb_voisins', 'd_route']].assign(lat=res.geometry.centroid.y, lon=res.geometry.centroid.x).to_csv(index=False)
                    m2.download_button("📥 Télécharger CSV", csv, "havens.csv", "text/csv", use_container_width=True)

                    # --- CARTE ---
                    m = folium.Map(location=[res.geometry.centroid.y.mean(), res.geometry.centroid.x.mean()], zoom_start=13)
                    folium.TileLayer(tiles='https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}', attr='Google', name='Satellite', max_zoom=22).add_to(m)

                    for i, (idx, row) in enumerate(res.iterrows()):
                        lat, lon = row.geometry.centroid.y, row.geometry.centroid.x
                        u_google = f"https://www.google.com/maps/dir/?api=1&destination={lat},{lon}"
                        u_waze = f"https://waze.com/ul?ll={lat},{lon}&navigate=yes"
                        u_sv = f"https://www.google.com/maps/@?api=1&map_action=pano&viewpoint={lat},{lon}"
                        
                        pop_html = f"""
                        <div style='font-family:Arial; width:170px;'>
                            <b style='color:#28a745;'>HAVEN #{i+1}</b><br>
                            <small>Voisins: {int(row['nb_voisins'])} | Route: {int(row['d_route'])}m</small>
                            <hr style='margin:8px 0;'>
                            <a href='{u_google}' target='_blank' class='btn-gps' style='background-color:#4285F4;'>🗺️ Google Maps</a>
                            <a href='{u_waze}' target='_blank' class='btn-gps' style='background-color:#33CCFF;'>🚙 Waze</a>
                            <a href='{u_sv}' target='_blank' class='btn-gps' style='background-color:#EA4335;'>🏙️ Street View</a>
                        </div>"""
                        
                        icon_html = f"""<div style="background-color:#E31A1C; border:2px solid white; border-radius:50%; width:28px; height:28px; color:white; font-weight:bold; display:flex; justify-content:center; align-items:center; box-shadow: 0 0 10px rgba(0,0,0,0.3);">{i+1}</div>"""
                        folium.Marker([lat, lon], popup=folium.Popup(pop_html, max_width=250), icon=folium.DivIcon(html=icon_html)).add_to(m)
                    
                    st_folium(m, width="100%", height=600, returned_objects=[])
            else:
                status.update(label="❌ Échec", state="error")
                st.warning("⚠️ Trop dense ou trop proche d'axes majeurs.")
        except Exception as e:
            st.error(f"❌ Erreur : {str(e)}")
