import streamlit as st
import osmnx as ox
import folium
from streamlit_folium import st_folium
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import warnings
import time

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
    commune_in = st.text_input("Commune :", value="", placeholder="Tapez votre commune ici...")
    dist_route_val = st.slider("Distance Route (m) :", 30, 300, 70)
    rayon_iso_val = st.slider("Rayon Isolement (m) :", 50, 600, 180)
    voisins_max_val = st.number_input("Voisins Max autorisés :", 0, 12, 2)
    st.markdown("---")
    lancer_scan = st.button("Lancer le Scan")

# --- LOGIQUE DE SCAN AVEC BARRE DE PROGRESSION ---
if lancer_scan:
    if not commune_in:
        st.error("⚠️ Veuillez entrer le nom d'une commune.")
    else:
        try:
            # Création des éléments de statut
            status_text = st.empty()
            progress_bar = st.progress(0)

            # ÉTAPE 1
            status_text.markdown("📍 **Étape 1/4 :** Localisation du secteur et définition des limites...")
            progress_bar.progress(10)
            base = ox.geocode_to_gdf(commune_in)
            geom_c = base.geometry.iloc[0]
            voisines = ox.features_from_polygon(geom_c.buffer(0.015), tags={'admin_level': '8'})
            secteur = pd.concat([base, voisines[voisines.geometry.intersects(geom_c)]]).to_crs(epsg=2154)
            secteur = secteur[secteur.geometry.area < 200_000_000] 
            union_zone = secteur.geometry.union_all()
            
            # ÉTAPE 2
            status_text.markdown("🛰️ **Étape 2/4 :** Récupération des données satellites (bâtiments et routes)...")
            progress_bar.progress(35)
            bbox = secteur.to_crs(epsg=4326).geometry.union_all().buffer(0.008) 
            bat = ox.features_from_polygon(bbox, tags={'building': True})
            tags_routes = ['primary', 'secondary', 'tertiary', 'residential', 'unclassified', 'trunk']
            routes = ox.features_from_polygon(bbox, tags={'highway': tags_routes})
            
            # ÉTAPE 3
            status_text.markdown("📐 **Étape 3/4 :** Analyse géométrique et calcul des distances aux routes...")
            progress_bar.progress(60)
            bat = bat[bat.geometry.type.isin(['Polygon', 'MultiPolygon'])].copy().to_crs(epsg=2154)
            routes = routes.to_crs(epsg=2154)
            bat['d_route'] = bat.geometry.centroid.apply(lambda x: routes.distance(x).min())
            candidates = bat[bat.geometry.centroid.within(union_zone) & (bat['d_route'] >= dist_route_val)].copy()
            
            # ÉTAPE 4
            status_text.markdown("🧪 **Étape 4/4 :** Test d'isolement du voisinage (filtrage final)...")
            progress_bar.progress(85)
            
            if not candidates.empty:
                coords_toutes = list(zip(bat.geometry.centroid.x, bat.geometry.centroid.y))
                coords_candidates = list(zip(candidates.geometry.centroid.x, candidates.geometry.centroid.y))
                nn = NearestNeighbors(radius=rayon_iso_val).fit(coords_toutes)
                adj = nn.radius_neighbors_graph(coords_candidates).toarray()
                candidates['nb_voisins'] = adj.sum(axis=1) - 1
                res = candidates[candidates['nb_voisins'] <= voisins_max_val].copy().to_crs(epsg=4326)
                
                # TERMINÉ
                progress_bar.progress(100)
                status_text.empty() # On efface le texte de statut
                progress_bar.empty() # On efface la barre

                if res.empty:
                    st.warning("⚠️ Aucun résultat pour ces critères.")
                else:
                    st.success(f"✅ {len(res)} pépites détectées !")
                    
                    m = folium.Map(location=[res.geometry.centroid.y.mean(), res.geometry.centroid.x.mean()], zoom_start=13)
                    folium.TileLayer(tiles='https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}', attr='Google', name='Satellite', max_zoom=22).add_to(m)

                    for i, (idx, row) in enumerate(res.iterrows()):
                        lat, lon = row.geometry.centroid.y, row.geometry.centroid.x
                        url_google = f"https://www.google.com/maps/search/?api=1&query={lat},{lon}"
                        url_waze = f"https://waze.com/ul?ll={lat},{lon}&navigate=yes"
                        url_sv = f"https://www.google.com/maps/@?api=1&map_action=pano&viewpoint={lat},{lon}"
                        
                        html_popup = f"""
                        <div style="font-family: Arial; min-width: 180px;">
                            <b>HAVEN #{i+1}</b><br>
                            <small>Voisins : {int(row['nb_voisins'])} | Route : {int(row['d_route'])}m</small><br>
                            <hr>
                            <a href="{url_google}" target="_blank" style="display:block;background:#4285F4;color:white;text-align:center;padding:5px;border-radius:3px;text-decoration:none;margin-bottom:5px;">🚗 Google Maps</a>
                            <a href="{url_waze}" target="_blank" style="display:block;background:#33CCFF;color:white;text-align:center;padding:5px;border-radius:3px;text-decoration:none;margin-bottom:5px;">🚙 Waze</a>
                            <a href="{url_sv}" target="_blank" style="display:block;background:#f44336;color:white;text-align:center;padding:5px;border-radius:3px;text-decoration:none;">🏙️ Street View</a>
                        </div>"""
                        
                        folium.Marker([lat, lon], popup=folium.Popup(html_popup, max_width=250)).add_to(m)
                    
                    st_folium(m, width="100%", height=600, returned_objects=[])
                    
                    # Export
                    csv = res[['nb_voisins', 'd_route']].assign(lat=res.geometry.centroid.y, lon=res.geometry.centroid.x).to_csv(index=False)
                    st.download_button("📥 Télécharger la liste (CSV)", csv, "haven_export.csv", "text/csv")
            else:
                progress_bar.empty()
                status_text.empty()
                st.warning("⚠️ Aucun bâtiment isolé trouvé dans cette zone.")
        except Exception as e:
            st.error(f"❌ Erreur lors du scan : {str(e)}")import streamlit as st
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

# --- TITRE AVEC NOUVEL EMOJI ---
st.title("🏡 Haven Radar")

# --- BARRE LATÉRALE ---
with st.sidebar:
    st.header("Paramètres du Scan")
    
    # Utilisation du placeholder pour le champ vide
    commune_in = st.text_input(
        "Commune :", 
        value="", 
        placeholder="Tapez votre commune ici..."
    )
    
    dist_route_val = st.slider("Distance Route (m) :", 30, 300, 70)
    rayon_iso_val = st.slider("Rayon Isolement (m) :", 50, 600, 180)
    voisins_max_val = st.number_input("Voisins Max autorisés :", 0, 12, 2)
    
    st.markdown("---")
    lancer_scan = st.button("Lancer le Scan")

# --- LOGIQUE DE DÉCLENCHEMENT ---
# Le scan se lance si on clique sur le bouton OU si l'utilisateur presse Entrée dans le champ texte
if lancer_scan or (commune_in and st.session_state.get('last_commune') != commune_in):
    if not commune_in:
        st.error("⚠️ Veuillez entrer le nom d'une commune.")
    else:
        try:
            with st.spinner("📡 Analyse en cours..."):
                # Sauvegarde de la commune pour éviter les doubles scans inutiles
                st.session_state['last_commune'] = commune_in
                
                # 1. Analyse Géographique
                base = ox.geocode_to_gdf(commune_in)
                geom_c = base.geometry.iloc[0]
                voisines = ox.features_from_polygon(geom_c.buffer(0.015), tags={'admin_level': '8'})
                secteur = pd.concat([base, voisines[voisines.geometry.intersects(geom_c)]]).to_crs(epsg=2154)
                secteur = secteur[secteur.geometry.area < 200_000_000] 
                union_zone = secteur.geometry.union_all()
                
                # 2. Données
                bbox = secteur.to_crs(epsg=4326).geometry.union_all().buffer(0.008) 
                bat = ox.features_from_polygon(bbox, tags={'building': True})
                tags_routes = ['primary', 'secondary', 'tertiary', 'residential', 'unclassified', 'trunk']
                routes = ox.features_from_polygon(bbox, tags={'highway': tags_routes})
                
                # 3. Calculs
                bat = bat[bat.geometry.type.isin(['Polygon', 'MultiPolygon'])].copy().to_crs(epsg=2154)
                routes = routes.to_crs(epsg=2154)
                bat['d_route'] = bat.geometry.centroid.apply(lambda x: routes.distance(x).min())
                candidates = bat[bat.geometry.centroid.within(union_zone) & (bat['d_route'] >= dist_route_val)].copy()
                
                if not candidates.empty:
                    coords_toutes = list(zip(bat.geometry.centroid.x, bat.geometry.centroid.y))
                    coords_candidates = list(zip(candidates.geometry.centroid.x, candidates.geometry.centroid.y))
                    nn = NearestNeighbors(radius=rayon_iso_val).fit(coords_toutes)
                    adj = nn.radius_neighbors_graph(coords_candidates).toarray()
                    candidates['nb_voisins'] = adj.sum(axis=1) - 1
                    res = candidates[candidates['nb_voisins'] <= voisins_max_val].copy().to_crs(epsg=4326)
                    
                    if res.empty:
                        st.warning("⚠️ Aucun résultat pour ces critères.")
                    else:
                        st.success(f"✅ {len(res)} pépites détectées !")
                        m = folium.Map(location=[res.geometry.centroid.y.mean(), res.geometry.centroid.x.mean()], zoom_start=13)
                        folium.TileLayer(tiles='https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}', attr='Google', name='Satellite', max_zoom=22).add_to(m)

                        for i, (idx, row) in enumerate(res.iterrows()):
                            lat, lon = row.geometry.centroid.y, row.geometry.centroid.x
                            url_google = f"https://www.google.com/maps/search/?api=1&query={lat},{lon}"
                            url_waze = f"https://waze.com/ul?ll={lat},{lon}&navigate=yes"
                            url_sv = f"https://www.google.com/maps/@?api=1&map_action=pano&viewpoint={lat},{lon}"
                            
                            html_popup = f"""
                            <div style="font-family: Arial; min-width: 180px;">
                                <b>HAVEN #{i+1}</b><br>
                                <small>Voisins : {int(row['nb_voisins'])} | Route : {int(row['d_route'])}m</small><br>
                                <hr>
                                <a href="{url_google}" target="_blank" style="display:block;background:#4285F4;color:white;text-align:center;padding:5px;border-radius:3px;text-decoration:none;margin-bottom:5px;">🚗 Google Maps</a>
                                <a href="{url_waze}" target="_blank" style="display:block;background:#33CCFF;color:white;text-align:center;padding:5px;border-radius:3px;text-decoration:none;margin-bottom:5px;">🚙 Waze</a>
                                <a href="{url_sv}" target="_blank" style="display:block;background:#f44336;color:white;text-align:center;padding:5px;border-radius:3px;text-decoration:none;">🏙️ Street View</a>
                            </div>"""
                            
                            folium.Marker([lat, lon], popup=folium.Popup(html_popup, max_width=250)).add_to(m)
                        
                        st_folium(m, width="100%", height=600, returned_objects=[])
                        
                        # Export
                        csv = res[['nb_voisins', 'd_route']].assign(lat=res.geometry.centroid.y, lon=res.geometry.centroid.x).to_csv(index=False)
                        st.download_button("📥 Télécharger (CSV)", csv, "haven_export.csv", "text/csv")
                else:
                    st.warning("⚠️ Aucun bâtiment isolé trouvé.")
        except Exception as e:
            st.error(f"❌ Erreur : {str(e)}")
