import streamlit as st
import osmnx as ox
import folium
from streamlit_folium import st_folium
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import warnings

# --- CONFIGURATION ---
warnings.filterwarnings("ignore")
st.set_page_config(page_title="HAVEN RADAR | Détecteur de Refuges", layout="wide")

# --- STYLE CSS POUR LE BOUTON ---
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

# --- TITRE PRINCIPAL ---
st.title("🏚️ Haven Radar")

# --- BARRE LATÉRALE (INTERFACE) ---
with st.sidebar:
    st.header("Paramètres du Scan")
    
    # MODIFICATION : Champ vide avec texte d'aide (placeholder)
    commune_in = st.text_input(
        "Commune :", 
        value="", 
        placeholder="Tapez votre commune ici..."
    )
    
    dist_route_val = st.slider("Distance Route (m) :", 30, 300, 70)
    rayon_iso_val = st.slider("Rayon Isolement (m) :", 50, 600, 180)
    voisins_max_val = st.number_input("Voisins Max autorisés :", 0, 12, 2)
    
    st.markdown("---")
    # MODIFICATION : Nom du bouton simplifié
    lancer_scan = st.button("Lancer le Scan")

# --- LOGIQUE DE SCAN ---
if lancer_scan:
    # SÉCURITÉ : Vérifier si la commune est renseignée
    if not commune_in:
        st.error("⚠️ Veuillez entrer le nom d'une commune avant de lancer le scan.")
    else:
        try:
            with st.spinner("📡 Analyse géographique en cours..."):
                # 1. Définition du secteur
                base = ox.geocode_to_gdf(commune_in)
                geom_c = base.geometry.iloc[0]
                
                # Limites et communes voisines
                voisines = ox.features_from_polygon(geom_c.buffer(0.015), tags={'admin_level': '8'})
                secteur = pd.concat([base, voisines[voisines.geometry.intersects(geom_c)]]).to_crs(epsg=2154)
                secteur = secteur[secteur.geometry.area < 200_000_000] 
                union_zone = secteur.geometry.union_all()
                
                # 2. Téléchargement des données
                bbox = secteur.to_crs(epsg=4326).geometry.union_all().buffer(0.008) 
                bat = ox.features_from_polygon(bbox, tags={'building': True})
                
                tags_routes = ['primary', 'secondary', 'tertiary', 'residential', 'unclassified', 'trunk']
                routes = ox.features_from_polygon(bbox, tags={'highway': tags_routes})
                
                # 3. Traitement Géométrique
                bat = bat[bat.geometry.type.isin(['Polygon', 'MultiPolygon'])].copy().to_crs(epsg=2154)
                routes = routes.to_crs(epsg=2154)
                
                # 4. Calcul des distances aux routes
                bat['d_route'] = bat.geometry.centroid.apply(lambda x: routes.distance(x).min())
                candidates = bat[bat.geometry.centroid.within(union_zone) & (bat['d_route'] >= dist_route_val)].copy()
                
                if not candidates.empty:
                    # 5. Vérification Isolement
                    coords_toutes = list(zip(bat.geometry.centroid.x, bat.geometry.centroid.y))
                    coords_candidates = list(zip(candidates.geometry.centroid.x, candidates.geometry.centroid.y))
                    
                    nn = NearestNeighbors(radius=rayon_iso_val).fit(coords_toutes)
                    adj = nn.radius_neighbors_graph(coords_candidates).toarray()
                    
                    candidates['nb_voisins'] = adj.sum(axis=1) - 1
                    res = candidates[candidates['nb_voisins'] <= voisins_max_val].copy().to_crs(epsg=4326)
                    
                    if res.empty:
                        st.warning("⚠️ Aucun bâtiment ne correspond à ces seuils.")
                    else:
                        st.success(f"✅ {len(res)} pépites détectées !")
                        
                        # --- AFFICHAGE DE LA CARTE ---
                        m = folium.Map(location=[res.geometry.centroid.y.mean(), res.geometry.centroid.x.mean()], zoom_start=13)
                        
                        # Couche Satellite
                        folium.TileLayer(
                            tiles='https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}', 
                            attr='Google', 
                            name='Satellite', 
                            max_zoom=22
                        ).add_to(m)
                        
                        for i, (idx, row) in enumerate(res.iterrows()):
                            lat, lon = row.geometry.centroid.y, row.geometry.centroid.x
                            
                            url_google = f"https://www.google.com/maps/search/?api=1&query={lat},{lon}"
                            # AJOUT : Lien Waze pour compléter
                            url_waze = f"https://waze.com/ul?ll={lat},{lon}&navigate=yes"
                            url_sv = f"https://www.google.com/maps/@?api=1&map_action=pano&viewpoint={lat},{lon}"
                            
                            html_popup = f"""
                            <div style="font-family: Arial; min-width: 180px;">
                                <b>Pépite #{i+1}</b><br>
                                <small>Voisins : {int(row['nb_voisins'])} | Route : {int(row['d_route'])}m</small><br>
                                <hr>
                                <a href="{url_google}" target="_blank" style="display:block;background:#4285F4;color:white;text-align:center;padding:5px;border-radius:3px;text-decoration:none;margin-bottom:5px;">🚗 Google Maps</a>
                                <a href="{url_waze}" target="_blank" style="display:block;background:#33CCFF;color:white;text-align:center;padding:5px;border-radius:3px;text-decoration:none;margin-bottom:5px;">🚙 Waze</a>
                                <a href="{url_sv}" target="_blank" style="display:block;background:#f44336;color:white;text-align:center;padding:5px;border-radius:3px;text-decoration:none;">🏙️ Street View</a>
                            </div>
                            """
                            
                            folium.Marker(
                                [lat, lon],
                                icon=folium.DivIcon(html=f'<div style="color:white;background:#E31A1C;border-radius:50%;width:24px;height:24px;display:flex;justify-content:center;align-items:center;border:1px solid white;font-weight:bold;">{i+1}</div>'),
                                popup=folium.Popup(html_popup, max_width=250)
                            ).add_to(m)
                        
                        st_folium(m, width="100%", height=600, returned_objects=[])
                        
                        # AJOUT : Bouton de téléchargement pour l'aspect business
                        csv = res[['nb_voisins', 'd_route']].to_csv(index=False)
                        st.download_button("📥 Télécharger les résultats (CSV)", csv, "haven_results.csv", "text/csv")
                else:
                    st.warning("⚠️ Rien trouvé avec ces paramètres.")

        except Exception as e:
            st.error(f"❌ Erreur : {str(e)}")
