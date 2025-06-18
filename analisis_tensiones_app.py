import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import scipy.stats as stats
from scipy.spatial import cKDTree  # Añadido para zonas de contacto

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

# === Selector Pid ===
if 'Pid' in df.columns:
    pids = df['Pid'].unique().tolist()
    pids.sort()
    pids.insert(0, "Tots (Ambos)")
    
    pid_seleccionat = st.selectbox("Selecciona Pid per analitzar:", pids)
    
    if pid_seleccionat == "Tots (Ambos)":
        df_filtrat = df.copy()
    else:
        df_filtrat = df[df['Pid'] == pid_seleccionat]
else:
    st.warning("No s'ha trobat la columna 'Pid'. S'analitzaran tots els nodes junts.")
    df_filtrat = df.copy()
    pid_seleccionat = "Tots (Ambos)"
# === Fin Selector Pid ===

# === Nueva sección: Análisis zona de contacto entre dos PIDs ===
if 'Pid' in df.columns and len(pids) > 2:
    st.subheader("🔗 Anàlisi de zones de contacte entre PIDs")

    pid1 = st.selectbox("Selecciona PID 1 per contacte:", pids[1:], index=0)
    pid2 = st.selectbox("Selecciona PID 2 per contacte:", pids[1:], index=1 if len(pids) > 2 else 0)

    if pid1 == pid2:
        st.warning("Selecciona dos PIDs diferents per analitzar la zona de contacte.")
    else:
        umbral_distancia = st.slider("Distància màxima per considerar contacte (mm)", 0.001, 1.0, 0.01, 0.001)

        df_pid1 = df[df['Pid'] == pid1]
        df_pid2 = df[df['Pid'] == pid2]

        coords_pid1 = df_pid1[['posx', 'posy', 'posz']].values
        coords_pid2 = df_pid2[['posx', 'posy', 'posz']].values

        tree_pid2 = cKDTree(coords_pid2)
        vecinos = tree_pid2.query_ball_point(coords_pid1, r=umbral_distancia)
        contacto_mask_pid1 = np.array([len(v) > 0 for v in vecinos])
        contacto_pid1 = df_pid1[contacto_mask_pid1]

        st.write(f"Nodos de PID `{pid1}` en contacte amb PID `{pid2}` (distància < {umbral_distancia} mm): **{len(contacto_pid1)}**")

        scatter_pid1 = go.Scatter3d(
            x=df_pid1['posx'], y=df_pid1['posy'], z=df_pid1['posz'],
            mode='markers',
            marker=dict(size=3, color='blue', opacity=0.5),
            name=f'PID {pid1}'
        )

        scatter_pid2 = go.Scatter3d(
            x=df_pid2['posx'], y=df_pid2['posy'], z=df_pid2['posz'],
            mode='markers',
            marker=dict(size=3, color='red', opacity=0.5),
            name=f'PID {pid2}'
        )

        scatter_contacto = go.Scatter3d(
            x=contacto_pid1['posx'], y=contacto_pid1['posy'], z=contacto_pid1['posz'],
            mode='markers',
            marker=dict(size=6, color='yellow'),
            name='Nodos en contacto'
        )

        fig_contacte = go.Figure(data=[scatter_pid1, scatter_pid2, scatter_contacto])
        fig_contacte.update_layout(
            title=f"Contacte entre PID {pid1} i PID {pid2}",
            scene=dict(
                xaxis_title='X [mm]',
                yaxis_title='Y [mm]',
                zaxis_title='Z [mm]'
            )
        )
        st.plotly_chart(fig_contacte, use_container_width=True)
# === Fin nueva sección ===


# 🔢 Estadístiques bàsiques
st.subheader("📊 Estadístiques bàsiques")
data = df_filtrat['FunctionTop:StressesVon MisesCentroid']
st.write(f"**PID seleccionat: {pid_seleccionat}**")
st.write(f"Màxim: {data.max():.4f} MPa")
st.write(f"Mínim: {data.min():.4f} MPa")
st.write(f"Mitja: {data.mean():.4f} MPa")
st.write(f"Mitjana: {data.median():.4f} MPa")
st.write(f"Desviació estàndard: {data.std():.4f}")
st.write("Quartils:")
st.write(data.quantile([0.25, 0.5, 0.75, 0.95]))
st.write(f"Asimetria (skewness): {stats.skew(data):.4f}")
st.write(f"Kurtosis: {stats.kurtosis(data):.4f}")

# 🔘 Selector de percentatge de mostra
porcentaje = st.slider("Selecciona percentatge de mostra", 0.01, 1.0, 1.0)
df_sample = df_filtrat.sample(frac=porcentaje, random_state=42)

# 📊 Gràfica 3D
st.subheader("🧱 Gràfica 3D - Tensions Von Mises")
fig = px.scatter_3d(
    df_sample,
    x='posx', y='posy', z='posz',
    color='FunctionTop:StressesVon MisesCentroid',
    color_continuous_scale='Jet',
    range_color=[0, 1],
    title='Distribució de Tensions Von Mises'
)
st.plotly_chart(fig, use_container_width=True)


# Filtrar nodes
st.subheader("⚫ Filtrar nodes per rang de tensió")

max_tension = float(df_filtrat['FunctionTop:StressesVon MisesCentroid'].max())
min_tension = float(df_filtrat['FunctionTop:StressesVon MisesCentroid'].min())

# Slider rango inferior
lower = st.slider("Tensió mínima (majors que)", min_value=min_tension, max_value=max_tension, value=min_tension, step=0.01)

# Slider rango superior, debe ser mayor o igual que lower
upper = st.slider("Tensió màxima (menors que)", min_value=lower, max_value=max_tension, value=max_tension, step=0.01)

# Filtrar nodos dentro del rango
nodos_filtrados = df_filtrat[(df_filtrat['FunctionTop:StressesVon MisesCentroid'] > lower) & (df_filtrat['FunctionTop:StressesVon MisesCentroid'] < upper)]
num_filtrados = len(nodos_filtrados)

st.write(f"Nombres de nodes amb tensió entre {lower:.3f} i {upper:.3f}: **{num_filtrados}**")

# Scatter de todos los puntos en gris claro y con transparencia
scatter_all = go.Scatter3d(
    x=df_filtrat['posx'], y=df_filtrat['posy'], z=df_filtrat['posz'],
    mode='markers',
    marker=dict(size=2, color='lightgray', opacity=0.4),
    name='Tots els punts'
)

# Scatter de nodos filtrados en negro y tamaño mayor
scatter_filtrados = go.Scatter3d(
    x=nodos_filtrados['posx'], y=nodos_filtrados['posy'], z=nodos_filtrados['posz'],
    mode='markers',
    marker=dict(size=5, color='black', symbol='circle'),
    name=f'Nodes amb tensió entre {lower:.3f} i {upper:.3f}'
)

fig = go.Figure(data=[=[scatter_all, scatter_filtrados])

fig.update_layout(
    title=f"Nodes amb tensió entre {lower:.3f} i {upper:.3f} (Total: {num_filtrados})",
    scene=dict(
        xaxis_title='X [mm]',
        yaxis_title='Y [mm]',
        zaxis_title='Z [mm]'
    ),
    legend=dict(x=0, y=1)
)

st.plotly_chart(fig, use_container_width=True)
