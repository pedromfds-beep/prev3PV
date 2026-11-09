import requests
import pandas as pd
import numpy as np
import pvlib
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# ==============================================================================
# CONFIGURAÇÃO DA PÁGINA STREAMLIT
# ==============================================================================
st.set_page_config(
    page_title="Previsão de Produção Fotovoltaica",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# BARRA LATERAL - CONFIGURAÇÕES DO SISTEMA
# ==============================================================================
st.sidebar.title("⚙️ Configurações da Central")

st.sidebar.subheader("📍 Localização")
col_lat, col_lon = st.sidebar.columns(2)
lat = col_lat.number_input("Latitude (°)", value=38.529, format="%.4f")
lon = col_lon.number_input("Longitude (°)", value=-8.010, format="%.4f")
altitude = st.sidebar.number_input("Altitude (m)", value=250, step=10)

st.sidebar.subheader("⚡ Potência e Perdas")
capacity_unit = st.sidebar.radio("Unidade de Potência:", ["MWp", "kWp"], horizontal=True)
if capacity_unit == "MWp":
    capacity_val = st.sidebar.number_input("Potência Instalada (MWp)", value=1.0, min_value=0.01, step=0.5, format="%.2f")
    total_peak_kw = capacity_val * 1000.0
else:
    capacity_val = st.sidebar.number_input("Potência Instalada (kWp)", value=1000.0, min_value=1.0, step=10.0, format="%.1f")
    total_peak_kw = capacity_val

losses_pct = st.sidebar.slider("Perdas Globais do Sistema (%)", min_value=5, max_value=30, value=14, step=1)
gamma_pdc = st.sidebar.number_input("Coef. Temperatura Potência (%/°C)", value=-0.40, step=0.02, format="%.2f") / 100.0

st.sidebar.subheader("🔄 Geometria dos Módulos")
system_type = st.sidebar.selectbox(
    "Tipo de Estrutura:",
    ["Seguidor Solar de 1 Eixo (HSAT)", "Estrutura Fixa"],
    index=0
)

if system_type == "Seguidor Solar de 1 Eixo (HSAT)":
    max_angle = st.sidebar.slider("Ângulo Máximo de Rotação (±°)", 30, 75, 60, step=5)
    gcr = st.sidebar.slider("Rácio Ocupação Solo (GCR)", 0.20, 0.60, 0.35, step=0.05,
                           help="Largura do painel dividida pela distância entre eixos.")
    backtrack = st.sidebar.checkbox("Ativar Backtracking (anti-sombreamento)", value=True)
    axis_tilt = 0.0
    axis_azimuth = 180.0
    fixed_tilt = None
    fixed_azimuth = None
else:
    fixed_tilt = st.sidebar.slider("Inclinação dos Painéis Tilt (°)", 0.0, 60.0, 37.0, step=1.0)
    fixed_azimuth = st.sidebar.slider("Azimute dos Painéis (°: 180° = Sul)", 90.0, 270.0, 180.0, step=5.0)
    max_angle, gcr, backtrack = None, None, None

albedo = 0.20


# ==============================================================================
# FUNÇÃO DE OBTENÇÃO DE DADOS COM CACHE (1 HORA)
# ==============================================================================
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_weather_data(latitude, longitude):
    """Descarrega a previsão meteorológica horária dos próximos 3 dias via Open-Meteo."""
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": [
            "temperature_2m",
            "wind_speed_10m",
            "shortwave_radiation",       # GHI (W/m²)
            "direct_normal_irradiance",   # DNI (W/m²)
            "diffuse_radiation",          # DHI (W/m²)
        ],
        "forecast_days": 3,
        "timezone": "UTC"
    }
    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    
    df = pd.DataFrame(data['hourly'])
    df['ds'] = pd.to_datetime(df['time'], utc=True)
    df.set_index('ds', inplace=True)
    
    df['temp_air'] = df['temperature_2m']
    df['wind_speed'] = df['wind_speed_10m'] / 3.6  # km/h -> m/s
    df['ghi'] = df['shortwave_radiation'].fillna(0).clip(lower=0)
    df['dni'] = df['direct_normal_irradiance'].fillna(0).clip(lower=0)
    df['dhi'] = df['diffuse_radiation'].fillna(0).clip(lower=0)
    return df


# ==============================================================================
# CÁLCULO FÍSICO COM PVLIB
# ==============================================================================
def run_pvlib_simulation(weather_df):
    times = weather_df.index
    
    # 1. Posição solar
    solpos = pvlib.solarposition.get_solarposition(
        time=times, latitude=lat, longitude=lon, altitude=altitude
    )
    solar_zenith = solpos['apparent_zenith']
    solar_azimuth = solpos['azimuth']
    solar_elevation = solpos['apparent_elevation']
    
    # 2. Geometria (Seguidor vs Fixo)
    if system_type == "Seguidor Solar de 1 Eixo (HSAT)":
        trk = pvlib.tracking.singleaxis(
            apparent_zenith=solar_zenith,
            solar_azimuth=solar_azimuth,
            axis_tilt=axis_tilt,
            axis_azimuth=axis_azimuth,
            max_angle=max_angle,
            backtrack=backtrack,
            gcr=gcr
        )
        surface_tilt = trk['surface_tilt'].fillna(0.0)
        surface_azimuth = trk['surface_azimuth'].fillna(axis_azimuth)
        tracker_theta = trk['tracker_theta'].fillna(0.0)
        aoi = trk['aoi']
    else:
        surface_tilt = pd.Series(fixed_tilt, index=times)
        surface_azimuth = pd.Series(fixed_azimuth, index=times)
        tracker_theta = pd.Series(0.0, index=times)
        aoi = pvlib.irradiance.aoi(
            surface_tilt=fixed_tilt,
            surface_azimuth=fixed_azimuth,
            solar_zenith=solar_zenith,
            solar_azimuth=solar_azimuth
        )
        
    # 3. Transposição POA (Hay-Davies)
    dni_extra = pvlib.irradiance.get_extra_radiation(times.dayofyear).values
    poa_comp = pvlib.irradiance.get_total_irradiance(
        surface_tilt=surface_tilt,
        surface_azimuth=surface_azimuth,
        solar_zenith=solar_zenith,
        solar_azimuth=solar_azimuth,
        dni=weather_df['dni'],
        ghi=weather_df['ghi'],
        dhi=weather_df['dhi'],
        dni_extra=dni_extra,
        albedo=albedo,
        model='haydavies'
    )
    poa_global = poa_comp['poa_global'].fillna(0).clip(lower=0)
    poa_direct = poa_comp['poa_direct'].fillna(0).clip(lower=0)
    
    # 4. Modelo Térmico Faiman
    t_cell = pvlib.temperature.faiman(
        poa_global=poa_global,
        temp_air=weather_df['temp_air'],
        wind_speed=weather_df['wind_speed'],
        u0=25.0, u1=6.84
    )
    
    # 5. Potência DC e AC
    # pdc0 para 1 kWp = 1000 W
    p_dc_1kw = pvlib.pvsystem.pvwatts_dc(
        effective_irradiance=poa_global,
        temp_cell=t_cell,
        pdc0=1000.0,
        gamma_pdc=gamma_pdc,
        temp_ref=25.0
    ).clip(lower=0)
    
    p_ac_1kw = (p_dc_1kw * (1.0 - losses_pct / 100.0)).clip(lower=0)
    
    # Corte noturno
    night = (solar_elevation <= 0) | (poa_global <= 0)
    p_ac_1kw[night] = 0.0
    p_dc_1kw[night] = 0.0
    
    # Escala para a potência total da central:
    # p_ac_1kw e p_dc_1kw estão em Watts para 1 kWp de gerador base.
    # Para uma central de total_peak_kw (kWp):
    p_dc_total_kw = (p_dc_1kw / 1000.0) * total_peak_kw
    p_dc_total_mw = p_dc_total_kw / 1000.0
    
    p_ac_total_kw = (p_ac_1kw / 1000.0) * total_peak_kw
    p_ac_total_mw = p_ac_total_kw / 1000.0
    
    res = pd.DataFrame({
        'Data/Hora (UTC)': times.tz_localize(None),
        'Elevação Solar (°)': solar_elevation.values,
        'Rotação Theta (°)': tracker_theta.values,
        'AOI (°)': aoi.values,
        'POA Total (W/m²)': poa_global.values,
        'GHI Horizontal (W/m²)': weather_df['ghi'].values,
        'Temp Ar (°C)': weather_df['temp_air'].values,
        'Temp Célula (°C)': t_cell.values,
        'Vento (m/s)': weather_df['wind_speed'].values,
        'Potência DC (kW)': p_dc_total_kw.values,
        'Potência DC (MW)': p_dc_total_mw.values,
        'Potência AC (kW)': p_ac_total_kw.values,
        'Potência AC (MW)': p_ac_total_mw.values
    }, index=times)
    
    return res


# ==============================================================================
# CORPO DA APLICAÇÃO
# ==============================================================================

st.title("☀️ Previsão Operacional de Produção Fotovoltaica")
st.markdown(
    f"Previsão física determinística contínua calculada via **pvlib** com previsões meteorológicas em tempo real da **Open-Meteo**. "
    f"Configuração atual: **{capacity_val:.2f} {capacity_unit}** em **{system_type}** ({lat:.3f}° N, {lon:.3f}° W)."
)

# Obter dados e simular
with st.spinner("A obter previsões meteorológicas e a calcular modelo físico pvlib..."):
    try:
        weather_df = fetch_weather_data(lat, lon)
        df_res = run_pvlib_simulation(weather_df)
    except Exception as e:
        st.error(f"Erro ao obter dados ou executar a simulação: {e}")
        st.stop()

# --- CÁLCULO DOS PICOS MÁXIMOS DIÁRIOS E ENERGIA ---
df_res['Data'] = df_res['Data/Hora (UTC)'].dt.date
daily_mwh = df_res.groupby('Data')['Potência AC (MW)'].sum()
daily_dates = list(daily_mwh.index)

p_col = 'Potência AC (MW)' if capacity_unit == "MWp" else 'Potência AC (kW)'
daily_pmax = df_res.groupby('Data')[p_col].max()
unit_pow = "MW" if capacity_unit == "MWp" else "kW"

# --- CARTÕES INDICADORES: APENAS PICO MÁXIMO DE POTÊNCIA POR DIA ---
cols_peak = st.columns(len(daily_dates))
for i, d in enumerate(daily_dates):
    peak_val = daily_pmax[d]
    label_day = "Hoje" if i == 0 else ("Amanhã" if i == 1 else "Depois de Amanhã")
    val_str = f"{peak_val:.3f} {unit_pow}" if capacity_unit == "MWp" else f"{peak_val:.0f} {unit_pow}"
    cols_peak[i].metric(
        label=f"⚡ Pico Máximo — {label_day} ({d.strftime('%d/%m')})",
        value=val_str
    )

st.write("")

# --- SECÇÃO VISUAL: MAPA DE PORTUGAL (CONTORNOS) E BARRAS DE PRODUÇÃO ---
col_map, col_bar = st.columns([1, 1])

with col_map:
    fig_map = go.Figure()
    fig_map.add_trace(go.Scattergeo(
        lat=[lat],
        lon=[lon],
        mode='markers+text',
        #text=[f"  📍 Central ({capacity_val:.1f} {capacity_unit})"],
        #textposition="top right",
        #textfont=dict(size=12, color='#2c3e50', family='Arial Black'),
        marker=dict(
            size=16,
            color='#ff7f0e',
            symbol='star',
            line=dict(color='#d35400', width=2)
        ),
        hoverinfo='text',
        hovertext=(
            f"<b>Central Fotovoltaica</b><br>"
            f"Coordenadas: {lat:.3f}° N, {lon:.3f}° W<br>"
            f"Potência: {capacity_val:.1f} {capacity_unit}<br>"
            f"Estrutura: {system_type}<br>"
            f"Produção Total (3 dias): {daily_mwh.sum():.0f} MWh"
        ),
        name="Central Solar"
    ))

    fig_map.update_geos(
        scope='europe',
        center=dict(lat=39.5, lon=-8.0),
        lataxis_range=[36.7, 42.4],
        lonaxis_range=[-10.1, -5.9],
        showcountries=True,
        countrycolor="#2c3e50",
        countrywidth=2,
        showsubunits=True,
        subunitcolor="#bdc3c7",
        showcoastlines=True,
        coastlinecolor="#2980b9",
        coastlinewidth=1.8,
        showland=True,
        landcolor="#f8f9fa",
        showocean=True,
        oceancolor="#eaf2f8",
        resolution=50
    )

    fig_map.update_layout(
        title="🗺️ Localização da Central (Portugal)",
        margin=dict(l=10, r=10, t=40, b=10),
        height=400,
        showlegend=False
    )
    st.plotly_chart(fig_map, use_container_width=True)

with col_bar:
    energy_vals = [daily_mwh[d] for d in daily_dates]
    bar_labels = [
        f"Hoje ({daily_dates[0].strftime('%d/%m')})",
        f"Amanhã ({daily_dates[1].strftime('%d/%m')})",
        f"Depois ({daily_dates[2].strftime('%d/%m')})"
    ]

    max_e = max(energy_vals) if len(energy_vals) > 0 and max(energy_vals) > 0 else 1.0
    bar_colors = [
        '#ff7f0e' if e >= max_e * 0.7 else ('#f39c12' if e >= max_e * 0.35 else '#3498db')
        for e in energy_vals
    ]

    fig_bar = go.Figure(go.Bar(
        x=bar_labels,
        y=energy_vals,
        text=[f"<b>{e:.0f} MWh</b>" for e in energy_vals],
        textposition='outside',
        textfont=dict(size=24, color='#1a252f', family='Arial Black'),
        cliponaxis=False,
        marker=dict(
            color=bar_colors,
            line=dict(color='#2c3e50', width=1.5)
        ),
        hovertemplate="<b>%{x}</b><br>Produção Prevista: %{y:.0f} MWh<extra></extra>"
    ))

    fig_bar.update_layout(
        title="📊 Produção de Energia PV Prevista (MWh)",
        yaxis_title="Produção (MWh)",
        margin=dict(l=10, r=10, t=50, b=10),
        height=400,
        yaxis=dict(
            tickformat='.0f',
            range=[0, max(max_e * 1.45, 1.0)]
        )
    )
    st.plotly_chart(fig_bar, use_container_width=True)

st.write("")

# --- ABAS DE VISUALIZAÇÃO INTERATIVA ---
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Produção e Potência",
    "☀️ Irradiação Solar",
    "🌡️ Meteorologia (Temp. & Vento)",
    "📐 Geometria e Dados Tabulares"
])

# TAB 1: POTÊNCIA
with tab1:
    fig_pow = go.Figure()
    y_col = 'Potência AC (MW)' if capacity_unit == "MWp" else 'Potência AC (kW)'
    fig_pow.add_trace(go.Scatter(
        x=df_res['Data/Hora (UTC)'],
        y=df_res[y_col],
        mode='lines',
        name=f'Potência AC ({capacity_unit})',
        line=dict(color='#ff7f0e', width=2.5),
        fill='tozeroy',
        fillcolor='rgba(255, 127, 14, 0.15)'
    ))
    fig_pow.update_layout(
        title=f"Curva de Carga Prevista - Próximos 3 Dias ({capacity_unit})",
        xaxis_title="Data / Hora (UTC)",
        yaxis_title=f"Potência Prevista ({capacity_unit})",
        hovermode="x unified",
        margin=dict(l=40, r=40, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_pow, use_container_width=True)

# TAB 2: IRRADIAÇÃO
with tab2:
    fig_irr = go.Figure()
    fig_irr.add_trace(go.Scatter(
        x=df_res['Data/Hora (UTC)'], y=df_res['POA Total (W/m²)'],
        mode='lines', name='POA no Plano dos Módulos (W/m²)',
        line=dict(color='red', width=2)
    ))
    fig_irr.add_trace(go.Scatter(
        x=df_res['Data/Hora (UTC)'], y=df_res['GHI Horizontal (W/m²)'],
        mode='lines', name='GHI Horizontal (W/m²)',
        line=dict(color='#1f77b4', dash='dash')
    ))
    fig_irr.update_layout(
        title="Irradiação Solar: Plano dos Painéis (POA) vs. Plano Horizontal (GHI)",
        xaxis_title="Data / Hora (UTC)", yaxis_title="Irradiação (W/m²)",
        hovermode="x unified",
        margin=dict(l=40, r=40, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_irr, use_container_width=True)

# TAB 3: METEOROLOGIA
with tab3:
    fig_met = make_subplots(specs=[[{"secondary_y": True}]])
    fig_met.add_trace(
        go.Scatter(x=df_res['Data/Hora (UTC)'], y=df_res['Temp Célula (°C)'],
                   name='Temp. Célula Fotovoltaica (°C)', line=dict(color='crimson', width=2)),
        secondary_y=False
    )
    fig_met.add_trace(
        go.Scatter(x=df_res['Data/Hora (UTC)'], y=df_res['Temp Ar (°C)'],
                   name='Temp. Ar Ambiente (°C)', line=dict(color='coral', dash='dot')),
        secondary_y=False
    )
    fig_met.add_trace(
        go.Scatter(x=df_res['Data/Hora (UTC)'], y=df_res['Vento (m/s)'],
                   name='Vento a 10m (m/s)', line=dict(color='teal', dash='dash')),
        secondary_y=True
    )
    fig_met.update_yaxes(title_text="Temperatura (°C)", secondary_y=False)
    fig_met.update_yaxes(title_text="Velocidade do Vento (m/s)", secondary_y=True)
    fig_met.update_layout(
        title="Variáveis Meteorológicas e Térmicas",
        xaxis_title="Data / Hora (UTC)",
        hovermode="x unified",
        margin=dict(l=40, r=40, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_met, use_container_width=True)

# TAB 4: GEOMETRIA E TABELA
with tab4:
    if system_type == "Seguidor Solar de 1 Eixo (HSAT)":
        fig_geom = go.Figure()
        fig_geom.add_trace(go.Scatter(
            x=df_res['Data/Hora (UTC)'], y=df_res['Rotação Theta (°)'],
            mode='lines+markers', name='Ângulo de Rotação Theta (°: Este<0, Oeste>0)',
            line=dict(color='purple', width=2)
        ))
        fig_geom.update_layout(
            title="Dinâmica de Rotação do Seguidor de 1 Eixo com Backtracking",
            xaxis_title="Data / Hora (UTC)", yaxis_title="Ângulo de Rotação (°)",
            yaxis_range=[-65, 65],
            hovermode="x unified",
            margin=dict(l=40, r=40, t=50, b=40)
        )
        st.plotly_chart(fig_geom, use_container_width=True)
    
    st.subheader("📋 Tabela de Previsão Horária")
    st.dataframe(df_res.style.format(precision=2), height=350, use_container_width=True)
    
    csv_bytes = df_res.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Descarregar Tabela de Previsão em CSV",
        data=csv_bytes,
        file_name="previsao_fotovoltaica_pvlib.csv",
        mime="text/csv"
    )

st.caption("Powered by pvlib-python & Open-Meteo Weather API.")
