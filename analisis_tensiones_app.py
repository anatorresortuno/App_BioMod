import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import scipy.stats as stats
from scipy.spatial import cKDTree  # Para buscar proximidades eficientes

st.set_page_config(page_title="Análisis Von Mises", layout="wide")

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
required_cols = ['posx', 'posy', 'posz', 'FunctionTop:StressesVon MisesCentroid']
missing = [col for col in required_cols if col not in df.columns]
if missing:
    st.error(f"Falten les columnes necessàries: {', '.join(missing)}")
    st.stop()

# ✅ Mostrem dades
st.success("Arxiu carregat correctament 🎉")
st.dataframe(df.head())

# === Selector PID múltiple (dos PIDs para comparar) ===
if 'Pid' in df.columns:
    pids = df['Pid'].unique().tolist()
    pids.sort()
    pids.insert(0, "Tots (Ambos)")

    pid_1 = st.selectbox("Selecciona primer PID:", pids, index=0)
    pid_2 = st.selectbox("Selecciona segon PID:", pids, index=0)
    
    if pid_1 == "Tots (Ambos)":
        df_pid_1 = df.copy()
    else:
        df_pid_1 = df[df['Pid'] == pid_1]

    if pid_2 == "Tots (Ambos)":
        df_pid_2 = df.copy()
    else:
        df_pid_2 = df[df['Pid'] == pid_2]
else:
    st.warning("No s'ha trobat la columna 'Pid'. S'analitzaran tots els nodes junts.")
    df_pid_1 = df.copy()
    df_pid_2 = df.copy()
    pid_1 = pid_2 = "Tots (Ambos)"

# Estadísticas básicas para PID 1 (puedes replicar para PID 2 si quieres)
st.subheader("📊 Estadístiques bàsiques PID 1")
data_1 = df_pid_1['FunctionTop:StressesVon MisesCentroid']
st.write(f"**PID 1 seleccionat: {pid_1}**")
st.write(f"Màxim: {data_1.max():.4f} MPa")
st.write(f"Mínim: {data_1.min():.4f} MPa")
st.write(f"Mitja: {data_1.mean():.4f} MPa")
st.write(f"Mitjana: {data_1.median():.4f} MPa")
st.write(f"Desviació estàndard: {data_1.std():.4f}")
st.write("Quartils:")
st.write(data_1.quantile([0.25, 0.5, 0.75, 0.95]))
st.write(f"Asimetria (skewness): {stats.skew(data_1):.4f}")
st.write(f"Kurtosis: {stats.kurtosis(data_1):.4f}")

# Selector de escala de color para la gráfica
color_scales = ['Jet', 'Viridis', 'Cividis', 'Plasma', 'Inferno', 'Magma', 'Turbo', 'Hot', 'Cool']
color_scale_sel = st.selectbox("Selecciona escala de color per la tensió Von Mises:", color_scales, index=0)

# Selector de percentatge de mostra (usar para PID 1 para mostrar en 3D)
porcentaje = st.slider("Selecciona percentatge de mostra (PID 1)", 0.01, 1.0, 1.0)
df_sample_1 = df_pid_1.sample(frac=porcentaje, random_state=42)

# Gráfica 3D para PID 1
st.subheader("🧱 Gràfica 3D - Tensions Von Mises PID 1")
fig1 = px.scatter_3d(
    df_sample_1,
    x='posx', y='posy', z='posz',
    color='FunctionTop:StressesVon MisesCentroid',
    color_continuous_scale=color_scale_sel,
    title=f'Distribució de Tensions Von Mises PID {pid_1}'
)
st.plotly_chart(fig1, use_container_width=True)

# --- Zonas de contacto entre PID 1 y PID 2 ---

st.subheader("🔎 Zones de contacte entre PID 1 i PID 2")

# Umbral de distancia para considerar contacto (en las mismas unidades que posx,y,z, ej mm)
dist_umbral = st.slider("Distància màxima per considerar contacte (mm)", 0.1, 10.0, 1.0, step=0.1)

# Construimos árboles KD para búsqueda rápida
coords_1 = df_pid_1[['posx', 'posy', 'posz']].values
coords_2 = df_pid_2[['posx', 'posy', 'posz']].values

tree_2 = cKDTree(coords_2)
# Encontrar índices de puntos en pid_1 que tienen vecinos en pid_2 dentro del umbral
contact_idx_1 = tree_2.query_ball_point(coords_1, r=dist_umbral)

# Filtrar nodos que tienen vecinos en PID 2
contact_nodes_1 = [i for i, neighbors in enumerate(contact_idx_1) if neighbors]

if contact_nodes_1:
    df_contact_1 = df_pid_1.iloc[contact_nodes_1]
    st.write(f"S'han trobat **{len(df_contact_1)}** nodes de PID 1 amb contacte dins {dist_umbral} mm amb PID 2.")
    st.dataframe(df_contact_1)

    # Mostrar 3D con los contactos destacados
    fig_contact = go.Figure()

    # Todos PID 1 en gris claro
    fig_contact.add_trace(go.Scatter3d(
        x=df_pid_1['posx'], y=df_pid_1['posy'], z=df_pid_1['posz'],
        mode='markers',
        marker=dict(size=2, color='lightgray', opacity=0.3),
        name=f'PID 1 ({pid_1})'
    ))

    # Todos PID 2 en azul claro
    fig_contact.add_trace(go.Scatter3d(
        x=df_pid_2['posx'], y=df_pid_2['posy'], z=df_pid_2['posz'],
        mode='markers',
        marker=dict(size=2, color='lightblue', opacity=0.3),
        name=f'PID 2 ({pid_2})'
    ))

    # Nodos de contacto PID 1 en rojo
    fig_contact.add_trace(go.Scatter3d(
        x=df_contact_1['posx'], y=df_contact_1['posy'], z=df_contact_1['posz'],
        mode='markers',
        marker=dict(size=5, color='red', symbol='circle'),
        name='Nodes de contacte PID 1'
    ))

    fig_contact.update_layout(
        title=f'Zones de contacte entre PID {pid_1} i PID {pid_2} (distància ≤ {dist_umbral} mm)',
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

# --- Fin zones de contacto ---

