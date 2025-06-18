import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import scipy.stats as stats
from scipy.spatial import cKDTree

st.set_page_config(page_title="Anàlisi Von Mises", layout="wide")

st.title("🔍 Visualització i Anàlisi de Tensions Von Mises")
st.write("Carrega un fitxer Excel amb resultats de Von Mises per començar.")

# ✅ Càrrega del fitxer
uploaded_file = st.file_uploader("Selecciona un arxiu Excel", type=["csv"])

# ⛔ Si no hi ha fitxer, atura aquí
if uploaded_file is None:
    st.warning("⚠️ Si us plau, carrega un arxiu Excel per continuar.")
    st.stop()

# ✅ Llegim les dades
try:
    df = pd.read_csv(uploaded_file)
except Exception as e:
    st.error(f"Error al carregar l'arxiu: {e}")
    st.stop()

# ✅ Verificació de columnes necessàries
required_cols = ['posx', 'posy', 'posz', 'FunctionTop:StressesVon MisesCentroid', 'Pid']
missing = [col for col in required_cols if col not in df.columns]
if missing:
    st.error(f"Falten les columnes necessàries: {', '.join(missing)}")
    st.stop()

# ✅ Mostrem dades
st.success("Arxiu carregat correctament 🎉")
st.dataframe(df.head())

# --- Selecció de PID 1 i PID 2 ---
df_pid_1 = df[df['Pid'] == 1]
df_pid_2 = df[df['Pid'] == 2]

# Estadístiques bàsiques per PID 1
st.subheader("📊 Estadístiques bàsiques PID 1")
data_1 = df_pid_1['FunctionTop:StressesVon MisesCentroid']
st.write(f"**PID 1 seleccionat: 1**")
st.write(f"Màxim: {data_1.max():.4f} MPa")
st.write(f"Mínim: {data_1.min():.4f} MPa")
st.write(f"Mitja: {data_1.mean():.4f} MPa")
st.write(f"Mitjana: {data_1.median():.4f} MPa")
st.write(f"Desviació estàndard: {data_1.std():.4f}")
st.write("Quartils:")
st.write(data_1.quantile([0.25, 0.5, 0.75, 0.95]))
st.write(f"Asimetria (skewness): {stats.skew(data_1):.4f}")
st.write(f"Kurtosis: {stats.kurtosis(data_1):.4f}")

# Selector de escala de color per a la gràfica
color_scales = ['Jet', 'Viridis', 'Cividis', 'Plasma', 'Inferno', 'Magma', 'Turbo', 'Hot', 'Cool']
color_scale_sel = st.selectbox("Selecciona escala de color per a la tensió Von Mises:", color_scales, index=0)

# Selecciona rang per a l'escala de color
min_val = float(df_pid_1['FunctionTop:StressesVon MisesCentroid'].min())
max_val = float(df_pid_1['FunctionTop:StressesVon MisesCentroid'].max())

st.write("### Ajusta el rang de valors per a l'escala de color (Von Mises)")
color_range_min, color_range_max = st.slider(
    "Selecciona rang de color per a l'escala de Von Mises",
    min_val, max_val,
    (min_val, max_val),
    step=0.01
)

# Selector de percentatge de mostra (usar per a PID 1 per mostrar en 3D)
porcentaje = st.slider("Selecciona percentatge de mostra (PID 1)", 0.01, 1.0, 1.0)
df_sample_1 = df_pid_1.sample(frac=porcentaje, random_state=42)

# Gràfica 3D per a PID 1
st.subheader("🧱 Gràfica 3D - Tensions Von Mises PID 1")
fig1 = px.scatter_3d(
    df_sample_1,
    x='posx', y='posy', z='posz',
    color='FunctionTop:StressesVon MisesCentroid',
    color_continuous_scale=color_scale_sel,
    range_color=[color_range_min, color_range_max],
    title=f'Distribució de Tensions Von Mises PID 1'
)
st.plotly_chart(fig1, use_container_width=True)

# --- Zones de contacte entre PID 1 i PID 2 ---
st.subheader("🔎 Zones de contacte entre PID 1 i PID 2")

# Umbral de distància per considerar contacte (en les mateixes unitats que posx,y,z, ej mm)
dist_umbral = st.slider("Distància màxima per considerar contacte (mm)", 0.1, 10.0, 1.0, step=0.1)

# Construïm arbres KD per a cerca ràpida
coords_1 = df_pid_1[['posx', 'posy', 'posz']].values
coords_2 = df_pid_2[['posx', 'posy', 'posz']].values

tree_2 = cKDTree(coords_2)
# Trobar índexs de punts en pid_1 que tenen veïns en pid_2 dins del umbral
contact_idx_1 = tree_2.query_ball_point(coords_1, r=dist_umbral)

# Filtrar nodes que tenen veïns en PID 2
contact_nodes_1 = [i for i, neighbors in enumerate(contact_idx_1) if neighbors]

if contact_nodes_1:
    df_contact_1 = df_pid_1.iloc[contact_nodes_1]
    st.write(f"S'han trobat **{len(df_contact_1)}** nodes de PID 1 amb contacte dins {dist_umbral} mm amb PID 2.")
    st.dataframe(df_contact_1)

    # Mostrar 3D amb els contactes destacats
    fig_contact = go.Figure()

    # Tots PID 1 en gris clar
    fig_contact.add_trace(go.Scatter3d(
        x=df_pid_1['posx'], y=df_pid_1['posy'], z=df_pid_1['posz'],
        mode='markers',
        marker=dict(size=2, color='lightgray', opacity=0.3),
        name=f'PID 1 (1)'
    ))

    # Tots PID 2 en blau clar
    fig_contact.add_trace(go.Scatter3d(
        x=df_pid_2['posx'], y=df_pid_2['posy'], z=df_pid_2['posz'],
        mode='markers',
        marker=dict(size=2, color='lightblue', opacity=0.3),
        name=f'PID 2 (2)'
    ))

    # Nodes de contacte PID 1 en vermell
    fig_contact.add_trace(go.Scatter3d(
        x=df_contact_1['posx'], y=df_contact_1['posy'], z=df_contact_1['posz'],
        mode='markers',
        marker=dict(size=5, color='red', symbol='circle'),
        name='Nodes de contacte PID 1'
    ))

    fig_contact.update_layout(
        title=f'Zones de contacte entre PID 1 i PID 2 (distància ≤ {dist_umbral} mm)',
        scene=dict(
            xaxis_title='X [mm]',
            yaxis_title='Y [mm]',
            zaxis_title='Z [mm]'
        ),
        legend=dict(x=0, y=1)
    )
    st.plotly_chart(fig_contact, use_container_width=True)
else:
    st.write(f"No s'han trobat nodes de PID 1 en contacte amb PID 2 dins la distància de {dist_umbral} mm.")

# --- Destacar nodes dins d'un rang de tensions Von Mises ---
st.subheader("⚫ Destacar nodes dins d'un rang de tensions Von Mises")

# Definir rang de tensions Von Mises per destacar
min_von_mises = st.number_input("Valor mínim de Von Mises (MPa)", min_value=0.0, value=0.5, step=0.01)
max_von_mises = st.number_input("Valor màxim de Von Mises (MPa)", min_value=0.0, value=1.0, step=0.01)

# Filtrar nodes dins del rang
df_range = df_pid_1[(df_pid_1['FunctionTop:StressesVon MisesCentroid'] >= min_von_mises) &
                    (df_pid_1['FunctionTop:StressesVon MisesCentroid'] <= max_von_mises)]

if not df_range.empty:
    st.write(f"S'han trobat **{len(df_range)}** nodes de PID 1 amb tensions Von Mises entre {min_von_mises} i {max_von_mises} MPa.")
    st.dataframe(df_range)

    # Mostrar 3D amb els nodes dins del rang en negre
    fig_range = go.Figure()

    # Tots PID 1 en gris clar
    fig_range.add_trace(go.Scatter3d(
        x=df_pid_1['posx'], y=df_pid_1['posy'], z=df_pid_1['posz'],
        mode='markers',
        marker=dict(size=2, color='lightgray', opacity=0.3),
        name=f'PID 1 (1)'
    ))

    # Nodes dins del rang en negre
    fig_range.add_trace(go.Scatter3d(
        x=df_range['posx'], y=df_range['posy'], z=df_range['posz'],
        mode='markers',
        marker=dict(size=5, color='black', symbol='circle'),
        name='Nodes dins del rang'
    ))

    fig_range.update_layout(
        title=f'Nodes de PID 1 amb tensions Von Mises entre {min_von_mises} i {max_von_mises} MPa',
        scene=dict(
            xaxis_title='X [mm]',
            yaxis_title='Y [mm]',
            zaxis_title='Z [mm]'
        ),
        legend=dict(x=0,
::contentReference[oaicite:0]{index=0}
