import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import scipy.stats as stats
from scipy.spatial import cKDTree
from datetime import datetime
from io import BytesIO
import hashlib

# Configuració de la pàgina
st.set_page_config(page_title="Anàlisi Von Mises", layout="wide")
st.title("Visualització i Anàlisi de Tensions Von Mises")
st.write("Carrega un fitxer Excel amb resultats de Von Mises per començar.")

# === Càrrega de fitxer ===
uploaded_file = st.file_uploader("Selecciona un arxiu Excel (CSV)", type=["csv"])
if uploaded_file is None:
    st.warning("Si us plau, carrega un arxiu Excel per continuar.")
    st.stop()

try:
    df = pd.read_csv(uploaded_file, error_bad_lines=False, warn_bad_lines=True)
except Exception as e:
    st.error(f"Error al carregar l'arxiu: {e}")
    st.stop()

# Comprovar columnes necessàries
required_cols = ['posx', 'posy', 'posz', 'FunctionTop:StressesVon MisesCentroid', 'Pid']
missing = [col for col in required_cols if col not in df.columns]
if missing:
    st.error(f"Falten les columnes necessàries: {', '.join(missing)}")
    st.stop()

st.success("Arxiu carregat correctament")
st.dataframe(df.head())

# === Funció per calcular estadístiques ===
def calcular_estadistiques(df_sub, label):
    data_vm = df_sub['FunctionTop:StressesVon MisesCentroid']
    return {
        'Element': label,
        'Nodes': len(df_sub),
        'Max': data_vm.max(),
        'Min': data_vm.min(),
        'Mean': data_vm.mean(),
        'Median': data_vm.median(),
        'StdDev': data_vm.std(),
        'Q25': data_vm.quantile(0.25),
        'Q50': data_vm.quantile(0.5),
        'Q75': data_vm.quantile(0.75),
        'Q95': data_vm.quantile(0.95),
        'Skewness': stats.skew(data_vm),
        'Kurtosis': stats.kurtosis(data_vm)
    }

# === Estadístiques per PID ===
df_pid_1 = df[df['Pid'] == 1]
df_pid_2 = df[df['Pid'] == 2]
stats_pid_1 = calcular_estadistiques(df_pid_1, 'PID 1')
stats_pid_2 = calcular_estadistiques(df_pid_2, 'PID 2')
stats_ambos = calcular_estadistiques(df[df['Pid'].isin([1, 2])], 'PID 1+2')

df_stats_mostrar = pd.DataFrame([stats_pid_1, stats_pid_2, stats_ambos])
st.subheader("Estadístiques acumulades")
st.dataframe(df_stats_mostrar)

# === Acumulador en sessió ===
if 'df_acumulat' not in st.session_state:
    st.session_state.df_acumulat = pd.DataFrame()
if 'fitxers_carregats' not in st.session_state:
    st.session_state.fitxers_carregats = set()

nom_fitxer = uploaded_file.name
data_registre = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# Funció per renombrar els elements (PID -> noms més entenedors)
def renombrar_element(element):
    return {
        'PID 1': 'VE',
        'PID 2': 'Maxilar',
        'PID 1+2': 'Total'
    }.get(element, element)

# Extraure el nom base d'un fitxer (ex: 'case1' de 'case1_xyz.csv')
def base_nom_fitxer(nom):
    return nom.split('_')[0].split('.')[0]  # Quita extensió també

# Generar color únic i consistent per nom base
def generar_color_per_nom(nom_base):
    hash_object = hashlib.md5(nom_base.encode())
    return '#' + hash_object.hexdigest()[:6]

# Crear nou DataFrame amb les estadístiques i dades addicionals
df_noves = pd.DataFrame([stats_pid_1, stats_pid_2, stats_ambos])
df_noves['Fitxer'] = nom_fitxer
df_noves['Data'] = data_registre

# Aplicar renombraments i generar noms base i colors
df_noves['Element'] = df_noves['Element'].apply(renombrar_element)
df_noves['BaseNom'] = df_noves['Fitxer'].apply(base_nom_fitxer)
df_noves['Color'] = df_noves['BaseNom'].apply(generar_color_per_nom)

# Comprovar si el fitxer ja s'ha carregat per evitar duplicats
if nom_fitxer in st.session_state.fitxers_carregats:
    st.warning(f"L'arxiu **{nom_fitxer}** ja ha estat carregat anteriorment. No s'han afegit duplicats.")
else:
    st.session_state.df_acumulat = pd.concat([st.session_state.df_acumulat, df_noves], ignore_index=True)
    st.session_state.fitxers_carregats.add(nom_fitxer)
    st.success(f"Les estadístiques de **{nom_fitxer}** s'han afegit a l'acumulat.")

# === Funció per aplicar color de text segons columna Color ===
def aplicar_color_text(df):
    # Aplica el color a TODAS las celdas de cada fila según el valor en 'Color'
    def color_row(row):
        return [f'color: {row.Color}'] * len(row)
    return df.style.apply(color_row, axis=1)

# Mostrar la taula acumulada amb colors en el text
st.subheader("Taula acumulada amb colors")
if not st.session_state.df_acumulat.empty:
    styled_df = aplicar_color_text(st.session_state.df_acumulat)
    st.dataframe(styled_df)
else:
    st.write("No hi ha dades acumulades encara.")

# === Exportar Excel acumulat ===
excel_buffer = BytesIO()
with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
    st.session_state.df_acumulat.to_excel(writer, index=False)
excel_buffer.seek(0)

st.download_button(
    label="Descarrega l'Excel acumulat d'estadístiques",
    data=excel_buffer,
    file_name="estadistiques_acumulades.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

# === Visualització ===
pid_selection = st.radio("Selecciona PID a visualitzar:", options=['1', '2', 'Ambos'], index=2)

if pid_selection == '1':
    df_sel = df[df['Pid'] == 1]
elif pid_selection == '2':
    df_sel = df[df['Pid'] == 2]
else:
    df_sel = df[df['Pid'].isin([1, 2])]

# Estadístiques bàsiques PID seleccionat
st.subheader(f"Estadístiques bàsiques PID {pid_selection}")
data_vm = df_sel['FunctionTop:StressesVon MisesCentroid']
st.write(f"Màxim: {data_vm.max():.4f} MPa")
st.write(f"Mínim: {data_vm.min():.4f} MPa")
st.write(f"Mitja: {data_vm.mean():.4f} MPa")
st.write(f"Mitjana: {data_vm.median():.4f} MPa")
st.write(f"Desviació estàndard: {data_vm.std():.4f}")
st.write("Quartils:")
st.write(data_vm.quantile([0.25, 0.5, 0.75, 0.95]))
st.write(f"Asimetria (skewness): {stats.skew(data_vm):.4f}")
st.write(f"Kurtosis: {stats.kurtosis(data_vm):.4f}")

# Selector escala de color
color_scales = ['Jet', 'Viridis', 'Cividis', 'Plasma', 'Inferno', 'Magma', 'Turbo', 'Hot', 'Cool']
color_scale_sel = st.selectbox("Selecciona escala de color per la tensió Von Mises:", color_scales, index=0)

# Rangs color
min_val = float(data_vm.min())
max_val = float(data_vm.max())
st.write("### Ajusta el rang de valors per a l'escala de color (Von Mises)")
color_range_min, color_range_max = st.slider(
    "Selecciona rang de color per a l'escala de Von Mises",
    min_val, max_val,
    (min_val, max_val),
    step=0.01
)

# Percentatge mostra
porcentaje = st.slider(f"Selecciona percentatge de mostra (PID {pid_selection})", 0.01, 1.0, 1.0)
df_sample = df_sel.sample(frac=porcentaje, random_state=42)

# Gràfica 3D per PID seleccionat
st.subheader(f"Gràfica 3D - Tensions Von Mises PID {pid_selection}")
fig1 = px.scatter_3d(
    df_sample,
    x='posx', y='posy', z='posz',
    color='FunctionTop:StressesVon MisesCentroid',
    color_continuous_scale=color_scale_sel,
    range_color=[color_range_min, color_range_max],
    title=f'Distribució de Tensions Von Mises PID {pid_selection}'
)
st.plotly_chart(fig1, use_container_width=True)

# Anàlisi de contacte si PID 'Ambos'
if pid_selection == 'Ambos':
    st.subheader("Zones de contacte entre PID 1 i PID 2")

    dist_umbral = st.slider("Distància màxima per considerar contacte (mm)", 0.1, 10.0, 1.0, step=0.1)

    coords_1 = df_pid_1[['posx', 'posy', 'posz']].values
    coords_2 = df_pid_2[['posx', 'posy', 'posz']].values

    tree_2 = cKDTree(coords_2)
    contact_idx_1 = tree_2.query_ball_point(coords_1, r=dist_umbral)
    contact_nodes_1 = [i for i, neighbors in enumerate(contact_idx_1) if neighbors]
    
    if contact_nodes_1:
        df_contact_1 = df_pid_1.iloc[contact_nodes_1]
        st.write(f"S'han trobat **{len(df_contact_1)}** nodes de PID 1 amb contacte dins {dist_umbral} mm amb PID 2.")
        st.dataframe(df_contact_1)

        fig_contact = go.Figure()
        fig_contact.add_trace(go.Scatter3d(
            x=df_pid_1['posx'], y=df_pid_1['posy'], z=df_pid_1['posz'],
            mode='markers',
            marker=dict(size=2, color='lightgray', opacity=0.3),
            name='PID 1'
        ))
        fig_contact.add_trace(go.Scatter3d(
            x=df_pid_2['posx'], y=df_pid_2['posy'], z=df_pid_2['posz'],
            mode='markers',
            marker=dict(size=2, color='lightblue', opacity=0.3),
            name='PID 2'
        ))
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

# Destacar nodos dentro del rango Von Mises según PID seleccionado
st.subheader("Destacar nodes dins d'un rang de tensions Von Mises")

min_von_mises = st.number_input("Valor mínim de Von Mises (MPa)", min_value=0.0, value=0.5, step=0.01)
max_von_mises = st.number_input("Valor màxim de Von Mises (MPa)", min_value=0.0, value=1.0, step=0.01)

df_range = df_sel[(df_sel['FunctionTop:StressesVon MisesCentroid'] >= min_von_mises) &
                  (df_sel['FunctionTop:StressesVon MisesCentroid'] <= max_von_mises)]

if not df_range.empty:
    st.write(f"S'han trobat **{len(df_range)}** nodes de PID {pid_selection} amb tensions Von Mises entre {min_von_mises} i {max_von_mises} MPa.")
    st.dataframe(df_range)

    fig_range = go.Figure()
    fig_range.add_trace(go.Scatter3d(
        x=df_sel['posx'], y=df_sel['posy'], z=df_sel['posz'],
        mode='markers',
        marker=dict(size=2, color='lightgray', opacity=0.3),
        name=f'PID {pid_selection}'
    ))

    fig_range.add_trace(go.Scatter3d(
        x=df_range['posx'], y=df_range['posy'], z=df_range['posz'],
        mode='markers',
        marker=dict(size=5, color='black', symbol='circle'),
        name=f'Nodes dins del rang ({min_von_mises} - {max_von_mises} MPa)'
    ))

    fig_range.update_layout(
        title=f'Nodes de PID {pid_selection} amb tensions Von Mises entre {min_von_mises} i {max_von_mises} MPa',
        scene=dict(
            xaxis_title='X [mm]',
            yaxis_title='Y [mm]',
            zaxis_title='Z [mm]'
        ),
        legend=dict(
            x=0,
            y=1,
            traceorder="normal",
            title_font_family="Arial",
            font=dict(
                family="Courier New, monospace",
                size=12,
                color="black"
            ),
            bgcolor="LightSteelBlue",
            bordercolor="Black",
            borderwidth=2
        )
    )
    st.plotly_chart(fig_range, use_container_width=True)
else:
    st.write(f"No s'han trobat nodes de PID {pid_selection} amb tensions dins el rang indicat.")
