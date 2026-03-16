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
    div.stButton > button:first-child { background-color: #28a745; color: white; border-radius: 5px; font-weight: bold; }
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
            if cols_fav[i % 4].button(f"#{num}", key=f"fbtn_{num}"):
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
            p_bar = st.progress(0)
            
            base = ox.geocode_to_gdf(commune_in)
            p_bar.progress(25)
            
            geom_c = base.geometry.iloc[0]
            voisines = ox.features_from_polygon(geom_c.buffer(0.015), tags={'admin_level': '8'})
            secteur = pd.concat([base, voisines[voisines.geometry.intersects(geom_c)]]).to_crs(epsg=2154)
            union_zone = secteur.geometry.union_all()
            p_bar.progress(50)
            
            bbox = secteur.to_crs(epsg=4326).geometry.union_all().buffer(0.01) 
            bat = ox.features_from_polygon(bbox, tags={'building': True})
            routes = ox.features_from_polygon(bbox, tags={'highway': ['primary', 'secondary', 'tertiary', 'residential', 'unclassified', 'trunk']})
            p_bar.progress(75)
            
            bat = bat[bat.geometry.type.isin(['Polygon', 'MultiPolygon'])].copy().to_crs(epsg=2154)
            routes = routes.to_crs(epsg=2154)
            bat['d_route'] = bat.geometry.centroid.apply(lambda x: routes.distance(x).min())
            candidates = bat[bat.geometry.centroid.within(union_zone) & (bat['d_route'] >= dist_route_val)].copy()
            
            if not candidates.empty:
                coords_toutes = list(zip(bat.geometry.centroid.x, bat.geometry.centroid.y))
                coords_candidates = list(zip(candidates.geometry.centroid.x, candidates.geometry.centroid.y))
                nn = NearestNeighbors(radius=rayon_iso_val).fit(coords_toutes)
                adj = nn.radius_neighbors_graph(coords_candidates).toarray()
                candidates['taille_hameau'] = adj.sum(axis=1)
                st.session_state.last_res = candidates[candidates['taille_hameau'] <= (taille_hameau_max + 1)].copy().to_crs(epsg=4326)
                st.session_state.map_center = [st.session_state.last_res.geometry.centroid.y.mean(), st.session_state.last_res.geometry.centroid.x.mean()]
            
            p_bar.progress(100)
            p_bar.empty()
        except Exception as e:
            st.error(f"❌ Erreur : {str(e)}")

# --- AFFICHAGE RESULTATS ET CARTE ---
if st.session_state.last_res is not None:
    res = st.session_state.last_res
    if not res.empty:
        st.success(f"✅ {len(res)} Havens détectés !")
        m = folium.Map(location=st.session_state.map_center, zoom_start=st.session_state.zoom_level)
        folium.TileLayer(tiles='https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}', attr='Google', name='Satellite', max_zoom=22).add_to(m)

        for i, (idx, row) in enumerate(res.iterrows()):
            lat, lon = row.geometry.centroid.y, row.geometry.centroid.x
            pop_html = f"""<div style='font-family:Arial; width:170px;'><b>HAVEN #{i+1}</b><br><small>{int(row['taille_hameau'])} bât.</small><hr>
            <a href='http://maps.google.com/maps?q={lat},{lon}' target='_blank'>🗺️ Google Maps</a><br>
            <a href='https://waze.com/ul?ll={lat},{lon}&navigate=yes' target='_blank'>🚙 Waze</a></div>"""
            
            # Correction de la syntaxe de l'icône
            icon_c = f'<div style="background-color:red; border:1px solid white; border-radius:50%; width:22px; height:22px; color:white; font-weight:bold; font-size:10px; display:flex; justify-content:center; align-items:center;">{i+1}</div>'
            folium.Marker([lat, lon], popup=folium.Popup(pop_html, max_width=200), icon=folium.DivIcon(html=icon_c)).add_to(m)
        
        # Gestion du clic pour synchronisation
        map_data = st_folium(m, width="100%", height=700, key="main_map", returned_objects=["last_object_clicked"])
        
        if map_data and map_data.get("last_object_clicked"):
            c_lat, c_lon = map_data["last_object_clicked"]["lat"], map_data["last_object_clicked"]["lng"]
            dists = res.geometry.centroid.apply(lambda p: ((p.y - c_lat)**2 + (p.x - c_lon)**2)**0.5)
            new_idx = res.index.get_loc(dists.idxmin())
            if st.session_state.sync_idx != new_idx:
                st.session_state.sync_idx = new_idx
                st.rerun()

        csv = res[['taille_hameau', 'd_route']].assign(lat=res.geometry.centroid.y, lon=res.geometry.centroid.x).to_csv(index=False)
        st.download_button("📥 Télécharger CSV", csv, "haven_radar.csv", "text/csv")
