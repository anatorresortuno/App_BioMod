import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import scipy.stats as stats

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

# === NUEVO: Selector Pid ===
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
# === FIN NUEVO ===


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

fig = go.Figure(data=[scatter_all, scatter_filtrados])

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
