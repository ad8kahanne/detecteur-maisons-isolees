import streamlit as st
import osmnx as ox
import folium
from streamlit_folium import st_folium
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import warnings

# --- CONFIGURATION ---
warnings.filterwarnings("ignore")
st.set_page_config(page_title="HAVEN RADAR | Détecteur", layout="wide")

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
    </style>
""", unsafe_allow_html=True)

st.title("🏡 Haven Radar")

# --- BARRE LATÉRALE ---
with st.sidebar:
    st.header("Paramètres du Scan")
    # On utilise une clé (key) pour stabiliser le champ texte
    commune_in = st.text_input("Commune :", value="", placeholder="Tapez votre commune ici...", key="city_input")
    dist_route_val = st.slider("Distance Route (m) :", 30, 300, 70)
    rayon_iso_val = st.slider("Rayon Isolement (m) :", 50, 600, 180)
    voisins_max_val = st.number_input("Voisins Max autorisés :", 0, 12, 2)
    st.markdown("---")
    lancer_scan = st.button("Lancer le Scan")

# --- LOGIQUE DE DÉCLENCHEMENT (BOUTON OU ENTREE) ---
# On déclenche si le bouton est cliqué OU si l'utilisateur valide le texte
if lancer_scan or (st.session_state.city_input and st.session_state.get('last_run') != st.session_state.city_input):
    if not commune_in:
        st.error("⚠️ Veuillez entrer le nom d'une commune.")
    else:
        try:
            st.session_state['last_run'] = commune_in
            status = st.empty()
            progress = st.progress(0)

            # Étape 1
            status.info("📍 Localisation du secteur...")
            progress.progress(15)
            base = ox.geocode_to_gdf(commune_in)
            geom_c = base.geometry.iloc[0]
            voisines = ox.features_from_polygon(geom_c.buffer(0.015), tags={'admin_level': '8'})
            secteur = pd.concat([base, voisines[voisines.geometry.intersects(geom_c)]]).to_crs(epsg=2154)
            secteur = secteur[secteur.geometry.area < 200_000_000] 
            union_zone = secteur.geometry.union_all()
            
            # Étape 2
            status.info("🛰️ Récupération des bâtiments et routes...")
            progress.progress(40)
            bbox = secteur.to_crs(epsg=4326).geometry.union_all().buffer(0.008) 
            bat = ox.features_from_polygon(bbox, tags={'building': True})
            tags_routes = ['primary', 'secondary', 'tertiary', 'residential', 'unclassified', 'trunk']
            routes = ox.features_from_polygon(bbox, tags={'highway': tags_routes})
            
            # Étape 3
            status.info("📐 Analyse de l'isolement routier...")
            progress.progress(70)
            bat = bat[bat.geometry.type.isin(['Polygon', 'MultiPolygon'])].copy().to_crs(epsg=2154)
            routes = routes.to_crs(epsg=2154)
            bat['d_route'] = bat.geometry.centroid.apply(lambda x: routes.distance(x).min())
            candidates = bat[bat.geometry.centroid.within(union_zone) & (bat['d_route'] >= dist_route_val)].copy()
            
            # Étape 4
            status.info("🧪 Filtrage du voisinage...")
            progress.progress(90)
            
            if not candidates.empty:
                coords_toutes = list(zip(bat.geometry.centroid.x, bat.geometry.centroid.y))
                coords_candidates = list(zip(candidates.geometry.centroid.x, candidates.geometry.centroid.y))
                nn = NearestNeighbors(radius=rayon_iso_val).fit(coords_toutes)
                adj = nn.radius_neighbors_graph(coords_candidates).toarray()
                candidates['nb_voisins'] = adj.sum(axis=1) - 1
                res = candidates[candidates['nb_voisins'] <= voisins_max_val].copy().to_crs(epsg=4326)
                
                status.empty()
                progress.empty()

                if res.empty:
                    st.warning("⚠️ Aucun refuge trouvé avec ces réglages.")
                else:
                    st.success(f"✅ {len(res)} Havens détectés !")
                    
                    m = folium.Map(location=[res.geometry.centroid.y.mean(), res.geometry.centroid.x.mean()], zoom_start=13)
                    folium.TileLayer(tiles='https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}', attr='Google', name='Satellite', max_zoom=22).add_to(m)

                    for i, (idx, row) in enumerate(res.iterrows()):
                        lat, lon = row.geometry.centroid.y, row.geometry.centroid.x
                        u_google = f"https://www.google.com/maps/search/?api=1&query={lat},{lon}"
                        u_waze = f"https://waze.com/ul?ll={lat},{lon}&navigate=yes"
                        u_sv = f"https://www.google.com/maps/@?api=1&map_action=pano&viewpoint={lat},{lon}"
                        
                        pop_html = f"""<div style='font-family:Arial; width:160px;'>
                        <b>HAVEN #{i+1}</b><br><small>Voisins : {int(row['nb_voisins'])}</small><hr>
                        <a href='{u_google}' target='_blank' style='color:#4285F4;display:block;'>🗺️ Google Maps</a>
                        <a href='{u_waze}' target='_blank' style='color:#33CCFF;display:block;'>🚙 Waze</a>
                        <a href='{u_sv}' target='_blank' style='color:#EA4335;display:block;'>🏙️ Street View</a>
                        </div>"""
                        
                        # Utilisation de marqueurs rouges simples et fiables
                        folium.Marker(
                            [lat, lon], 
                            popup=folium.Popup(pop_html, max_width=250),
                            icon=folium.Icon(color='red', icon='home', prefix='fa')
                        ).add_to(m)
                    
                    st_folium(m, width="100%", height=600, returned_objects=[])
                    
                    csv = res[['nb_voisins', 'd_route']].assign(lat=res.geometry.centroid.y, lon=res.geometry.centroid.x).to_csv(index=False)
                    st.download_button("📥 Télécharger CSV", csv, "haven_radar.csv", "text/csv")
            else:
                status.empty()
                progress.empty()
                st.warning("⚠️ Zone trop dense.")
        except Exception as e:
            st.error(f"❌ Erreur : {str(e)}")
