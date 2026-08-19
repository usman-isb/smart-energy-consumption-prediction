import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pickle, os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ── Paths ─────────────────────────────────────────────────────────
BASE   = os.path.dirname(os.path.abspath(__file__))
PROC   = os.path.normpath(os.path.join(BASE, '..', 'data', 'processed'))
MODELS = os.path.normpath(os.path.join(BASE, '..', 'models'))
FIGS   = os.path.normpath(os.path.join(BASE, '..', 'notebooks', 'figures'))

# ── Page config ───────────────────────────────────────────────────
st.set_page_config(
    page_title="Smart Energy Forecasting",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
[data-testid="stMetricValue"] { font-size: 1.8rem; }
.block-container { padding-top: 1.5rem; }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────
MODEL_NAMES   = ['Random Forest', 'XGBoost', 'LSTM', 'GRU', 'CNN-LSTM']
PJM_KEYS      = ['RF_PJM',     'XGB_PJM',     'LSTM_PJM',     'GRU_PJM',     'CNNLSTM_PJM']
UCI_KEYS      = ['RF_UCI',     'XGB_UCI',     'LSTM_UCI',     'GRU_UCI',     'CNNLSTM_UCI']
AEP_KEYS      = ['RF_AEP',     'XGB_AEP',     'LSTM_AEP',     'GRU_AEP',     'CNNLSTM_AEP']
UCIMV_KEYS    = ['RF_UCI_MV',  'XGB_UCI_MV',  'LSTM_UCI_MV',  'GRU_UCI_MV',  'CNNLSTM_UCI_MV']
PALETTE       = ['#4472C4', '#ED7D31', '#70AD47', '#FFC000', '#7030A0']
METRICS_CFG   = [('RMSE', 'lower'), ('MAE', 'lower'), ('MAPE (%)', 'lower'), ('R²', 'higher')]

# ── Cached loaders ────────────────────────────────────────────────
@st.cache_data
def load_pjm():
    return pd.read_csv(os.path.join(PROC, 'pjm_processed.csv'),
                       index_col='Datetime', parse_dates=True)

@st.cache_data
def load_uci():
    return pd.read_csv(os.path.join(PROC, 'uci_processed.csv'),
                       index_col='Datetime', parse_dates=True)

@st.cache_data
def load_aep():
    p = os.path.join(PROC, 'aep_processed.csv')
    if not os.path.exists(p): return None
    return pd.read_csv(p, index_col='Datetime', parse_dates=True)

@st.cache_data
def load_predictions():
    with open(os.path.join(MODELS, 'all_predictions.pkl'), 'rb') as f:
        return pickle.load(f)

@st.cache_data
def load_aep_predictions():
    p = os.path.join(MODELS, 'aep_predictions.pkl')
    if not os.path.exists(p): return None
    with open(p, 'rb') as f: return pickle.load(f)

@st.cache_data
def load_ucimv_predictions():
    p = os.path.join(MODELS, 'uci_mv_predictions.pkl')
    if not os.path.exists(p): return None
    with open(p, 'rb') as f: return pickle.load(f)

@st.cache_data
def load_ablation():
    p = os.path.join(MODELS, 'ablation_results.pkl')
    if not os.path.exists(p): return None
    with open(p, 'rb') as f: return pickle.load(f)

# ── Helpers ───────────────────────────────────────────────────────
def compute_metrics(y_true, y_pred):
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae  = float(mean_absolute_error(y_true, y_pred))
    mask = y_true > 0
    mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)
    r2   = float(r2_score(y_true, y_pred))
    return rmse, mae, mape, r2

def build_metrics_df(keys, res_dict):
    rows = []
    for name, key in zip(MODEL_NAMES, keys):
        rmse, mae, mape, r2 = compute_metrics(res_dict[key]['true'], res_dict[key]['pred'])
        rows.append({'Model': name, 'RMSE': rmse, 'MAE': mae, 'MAPE (%)': mape, 'R²': r2})
    return pd.DataFrame(rows).set_index('Model')

def styled_table(df):
    def colour_min(s): return ['background-color:#d4f0d4' if v == s.min() else '' for v in s]
    def colour_max(s): return ['background-color:#d4f0d4' if v == s.max() else '' for v in s]
    return (df.style
            .apply(colour_min, subset=['RMSE','MAE','MAPE (%)'])
            .apply(colour_max, subset=['R²'])
            .format({'RMSE':'{:.4f}','MAE':'{:.4f}','MAPE (%)':'{:.2f}','R²':'{:.4f}'}))

# ── Load data ─────────────────────────────────────────────────────
pjm     = load_pjm()
uci     = load_uci()
aep_df  = load_aep()
res_pjm = load_predictions()        # contains PJM + UCI univariate
res_aep = load_aep_predictions()    # None if notebook 06 not done
res_mv  = load_ucimv_predictions()  # None if notebook 07 not done
ablation= load_ablation()           # None if notebook 08 not done

# ── Dataset options (only show completed ones) ─────────────────
ds_options = ["PJM East (Regional Grid)", "UCI Household (Univariate)"]
if res_aep  is not None: ds_options.append("AEP Hourly (Regional Grid)")
if res_mv   is not None: ds_options.append("UCI Household (Multivariate)")

# ── Sidebar ───────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚡ Smart Energy\nForecasting")
    st.caption("Smart Energy Consumption Forecasting")
    st.divider()

    page = st.radio("Navigate", [
        "🏠  Home",
        "📊  Data Explorer",
        "🔮  Forecasts",
        "📈  Performance",
        "🔬  Research Results",
    ])
    st.divider()

    dataset = st.radio("Dataset", ds_options)
    if "PJM" in dataset:
        ds_tag, ds_label = "PJM", "PJM East"
        KEYS, res = PJM_KEYS, res_pjm
        df, target, unit, clr = pjm, "PJME_MW", "MW", "#4472C4"
        hist_suffix = "pjm"
    elif "AEP" in dataset:
        ds_tag, ds_label = "AEP", "AEP Hourly"
        KEYS, res = AEP_KEYS, res_aep
        df, target, unit, clr = aep_df, "AEP_MW", "MW", "#70AD47"
        hist_suffix = "aep"
    elif "Multivariate" in dataset:
        ds_tag, ds_label = "UCIMV", "UCI Household (Multivariate)"
        KEYS, res = UCIMV_KEYS, res_mv
        df, target, unit, clr = uci, "Global_active_power", "kW", "#7030A0"
        hist_suffix = "uci_mv"
    else:
        ds_tag, ds_label = "UCI", "UCI Household"
        KEYS, res = UCI_KEYS, res_pjm
        df, target, unit, clr = uci, "Global_active_power", "kW", "#ED7D31"
        hist_suffix = "uci"

    # For UCI univariate the predictions live in res_pjm (the combined pkl)
    if ds_tag == "UCI":
        res = res_pjm

# ═══════════════════════════════════════════════════════════════════
# PAGE: HOME
# ═══════════════════════════════════════════════════════════════════
if page == "🏠  Home":
    st.title("Smart Energy Consumption Forecasting")
    st.subheader("Deep Learning & Time Series Analytics Framework")
    st.markdown("""
    This dashboard presents a comparative evaluation of **five forecasting models** —
    Random Forest, XGBoost, LSTM, GRU, and CNN-LSTM — applied to **three publicly available
    energy consumption datasets**.  Use the sidebar to explore the data, compare predictions,
    and examine performance metrics across both univariate and multivariate settings.
    """)
    st.divider()

    # KPI cards
    cols = st.columns(5)
    cols[0].metric("PJM Records",     f"{len(pjm):,}",  "hourly (2002–2018)")
    cols[1].metric("AEP Records",     f"{len(aep_df):,}" if aep_df is not None else "⏳ training",
                   "hourly (2004–2018)" if aep_df is not None else "")
    cols[2].metric("UCI Records",     f"{len(uci):,}",  "hourly (2006–2010)")
    cols[3].metric("Models Trained",  f"{5 * (2 + (1 if res_aep else 0) + (1 if res_mv else 0))}",
                   "across all datasets")
    cols[4].metric("Best PJM R²",     "0.9962",          "Random Forest")

    st.divider()

    # Mini time series
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("PJM East — Weekly Average Demand")
        fig = px.line(pjm['PJME_MW'].resample('W').mean().reset_index(),
                      x='Datetime', y='PJME_MW', color_discrete_sequence=['#4472C4'])
        fig.update_layout(xaxis_title='', yaxis_title='MW', height=300, margin=dict(t=10))
        fig.update_traces(line_width=1.2)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.subheader("UCI Household — Weekly Average Power")
        fig = px.line(uci['Global_active_power'].resample('W').mean().reset_index(),
                      x='Datetime', y='Global_active_power',
                      color_discrete_sequence=['#ED7D31'])
        fig.update_layout(xaxis_title='', yaxis_title='kW', height=300, margin=dict(t=10))
        fig.update_traces(line_width=1.2)
        st.plotly_chart(fig, use_container_width=True)

    if aep_df is not None:
        st.subheader("AEP Hourly — Weekly Average Demand")
        fig = px.line(aep_df['AEP_MW'].resample('W').mean().reset_index(),
                      x='Datetime', y='AEP_MW', color_discrete_sequence=['#70AD47'])
        fig.update_layout(xaxis_title='', yaxis_title='MW', height=260, margin=dict(t=10))
        fig.update_traces(line_width=1.2)
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("Models Overview")
    model_info = pd.DataFrame({
        'Model': MODEL_NAMES,
        'Type': ['Traditional ML', 'Traditional ML', 'Deep Learning', 'Deep Learning', 'Deep Learning'],
        'Architecture': [
            'Ensemble of 100 decision trees',
            '200 gradient-boosted trees (lr=0.05)',
            'LSTM(64) → LSTM(32) → Dense(1)',
            'GRU(64)  → GRU(32)  → Dense(1)',
            'Conv1D(64) → MaxPool → Conv1D(32) → LSTM(50) → Dense(1)',
        ],
        'Input window': ['24 h'] * 5,
    })
    st.dataframe(model_info.set_index('Model'), use_container_width=True)


# ═══════════════════════════════════════════════════════════════════
# PAGE: DATA EXPLORER
# ═══════════════════════════════════════════════════════════════════
elif page == "📊  Data Explorer":
    st.title(f"📊 Data Explorer — {ds_label}")

    tab_ts, tab_pat, tab_heat = st.tabs([
        "📈 Time Series", "📉 Consumption Patterns", "🔥 Heatmap"
    ])

    with tab_ts:
        min_yr = int(df.index.year.min())
        max_yr = int(df.index.year.max())
        yr_range = st.slider("Year range", min_yr, max_yr, (min_yr, max_yr), key="yr_slider")
        freq_opt = st.selectbox("Resample frequency",
                                ["Hourly (raw)", "Daily", "Weekly", "Monthly"])
        freq_map = {"Hourly (raw)": None, "Daily": "D", "Weekly": "W", "Monthly": "ME"}
        fr = freq_map[freq_opt]
        df_filt  = df.loc[str(yr_range[0]):str(yr_range[1]), target]
        plot_ser = df_filt.resample(fr).mean() if fr else df_filt
        plot_df  = plot_ser.reset_index()
        plot_df.columns = ['Datetime', target]
        fig = px.line(plot_df, x='Datetime', y=target,
                      title=f"{ds_label} — {target} ({freq_opt})",
                      color_discrete_sequence=[clr])
        fig.update_traces(line_width=0.8 if fr is None else 1.5)
        fig.update_layout(xaxis_title='Date', yaxis_title=unit, height=450)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Summary statistics for selected period")
        st.dataframe(df_filt.describe().rename('Value').to_frame().T.round(4),
                     use_container_width=True)

    with tab_pat:
        c1, c2, c3 = st.columns(3)
        dow_labels = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
        mon_labels = ['Jan','Feb','Mar','Apr','May','Jun',
                      'Jul','Aug','Sep','Oct','Nov','Dec']
        with c1:
            h = df.groupby('hour')[target].mean().reset_index()
            fig = px.bar(h, x='hour', y=target, title='By Hour of Day',
                         color_discrete_sequence=[clr])
            fig.update_layout(xaxis_title='Hour', yaxis_title=f'Avg {unit}', height=350)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            d = df.groupby('day_of_week')[target].mean().reset_index()
            d['Day'] = [dow_labels[i] for i in d['day_of_week']]
            fig = px.bar(d, x='Day', y=target, title='By Day of Week',
                         color_discrete_sequence=[clr])
            fig.update_layout(xaxis_title='Day', yaxis_title=f'Avg {unit}', height=350)
            st.plotly_chart(fig, use_container_width=True)
        with c3:
            m = df.groupby('month')[target].mean().reset_index()
            m['Month'] = [mon_labels[i-1] for i in m['month']]
            fig = px.bar(m, x='Month', y=target, title='By Month',
                         color_discrete_sequence=[clr])
            fig.update_layout(xaxis_title='Month', yaxis_title=f'Avg {unit}', height=350)
            st.plotly_chart(fig, use_container_width=True)

        season_map = {0: 'Winter', 1: 'Spring', 2: 'Summer', 3: 'Autumn'}
        seas = df.groupby('season')[target].mean().reset_index()
        seas['Season'] = seas['season'].map(season_map)
        fig = px.bar(seas, x='Season', y=target, title='By Season',
                     color='Season',
                     color_discrete_sequence=['#4472C4','#70AD47','#ED7D31','#FFC000'])
        fig.update_layout(yaxis_title=f'Avg {unit}', showlegend=False, height=350)
        st.plotly_chart(fig, use_container_width=True)

    with tab_heat:
        pivot = df.pivot_table(values=target, index='hour',
                               columns='day_of_week', aggfunc='mean')
        pivot.columns = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
        fig = px.imshow(pivot, color_continuous_scale='YlOrRd',
                        title=f'{ds_label} — Average {unit} by Hour × Day of Week',
                        labels=dict(x='Day of Week', y='Hour of Day', color=f'Avg {unit}'),
                        aspect='auto',
                        text_auto='.1f' if ds_tag in ('PJM','AEP') else '.2f')
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
        peak = pivot.stack().idxmax()
        low  = pivot.stack().idxmin()
        col1, col2 = st.columns(2)
        col1.metric("Peak demand",   f"{pivot.stack().max():,.2f} {unit}", f"{peak[1]}, {peak[0]:02d}:00")
        col2.metric("Lowest demand", f"{pivot.stack().min():,.2f} {unit}", f"{low[1]}, {low[0]:02d}:00")


# ═══════════════════════════════════════════════════════════════════
# PAGE: FORECASTS
# ═══════════════════════════════════════════════════════════════════
elif page == "🔮  Forecasts":
    st.title(f"🔮 Forecasts — {ds_label}")

    if res is None:
        st.warning("Results not yet available for this dataset. Run the corresponding training notebook first.")
        st.stop()

    col_l, col_r = st.columns([2, 1])
    with col_l:
        selected = st.multiselect("Models to display", MODEL_NAMES,
                                  default=['Random Forest', 'XGBoost', 'LSTM'])
    with col_r:
        n = st.slider("Test samples", 100, min(2000, len(res[KEYS[0]]['true'])), 500)

    if not selected:
        st.warning("Select at least one model.")
        st.stop()

    fig = go.Figure()
    first_key = KEYS[MODEL_NAMES.index(selected[0])]
    y_true    = res[first_key]['true'][:n]
    fig.add_trace(go.Scatter(y=y_true, name='Actual',
                             line=dict(color='#222', width=1.8), opacity=0.9))
    for name, colour in zip(MODEL_NAMES, PALETTE):
        if name not in selected: continue
        key    = KEYS[MODEL_NAMES.index(name)]
        y_pred = res[key]['pred'][:n]
        fig.add_trace(go.Scatter(y=y_pred, name=name,
                                 line=dict(color=colour, width=1.3), opacity=0.85))
    fig.update_layout(
        title=f'{ds_label} — Actual vs Predicted (first {n} test samples)',
        xaxis_title='Test Sample Index', yaxis_title=unit, height=480,
        hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, x=0),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Metrics — Selected Models")
    rows = []
    for name in selected:
        key = KEYS[MODEL_NAMES.index(name)]
        rmse, mae, mape, r2 = compute_metrics(res[key]['true'], res[key]['pred'])
        rows.append({'Model': name, f'RMSE ({unit})': round(rmse, 4),
                     f'MAE ({unit})': round(mae, 4),
                     'MAPE (%)': round(mape, 2), 'R²': round(r2, 4)})
    st.dataframe(pd.DataFrame(rows).set_index('Model'), use_container_width=True)

    st.subheader("Residual Distribution (Actual − Predicted)")
    fig2 = go.Figure()
    for name, colour in zip(MODEL_NAMES, PALETTE):
        if name not in selected: continue
        key       = KEYS[MODEL_NAMES.index(name)]
        residuals = res[key]['true'] - res[key]['pred']
        fig2.add_trace(go.Histogram(x=residuals, name=name, nbinsx=80,
                                    marker_color=colour, opacity=0.55))
    fig2.update_layout(barmode='overlay', xaxis_title=f'Residual ({unit})',
                       yaxis_title='Count', height=360,
                       legend=dict(orientation='h', yanchor='bottom', y=1.02))
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Scatter — Actual vs Predicted")
    scatter_model = st.selectbox("Model for scatter", selected)
    key    = KEYS[MODEL_NAMES.index(scatter_model)]
    colour = PALETTE[MODEL_NAMES.index(scatter_model)]
    fig3   = go.Figure()
    fig3.add_trace(go.Scatter(x=res[key]['true'], y=res[key]['pred'],
                              mode='markers', marker=dict(color=colour, size=3, opacity=0.4),
                              name=scatter_model))
    lim = [min(res[key]['true'].min(), res[key]['pred'].min()),
           max(res[key]['true'].max(), res[key]['pred'].max())]
    fig3.add_trace(go.Scatter(x=lim, y=lim, mode='lines',
                              line=dict(color='black', dash='dash', width=1.5),
                              name='Perfect fit'))
    rmse, mae, mape, r2 = compute_metrics(res[key]['true'], res[key]['pred'])
    fig3.update_layout(title=f'{scatter_model} — Actual vs Predicted  (R²={r2:.4f})',
                       xaxis_title=f'Actual ({unit})', yaxis_title=f'Predicted ({unit})',
                       height=450)
    st.plotly_chart(fig3, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════
# PAGE: PERFORMANCE
# ═══════════════════════════════════════════════════════════════════
elif page == "📈  Performance":
    st.title(f"📈 Performance Comparison — {ds_label}")

    if res is None:
        st.warning("Results not yet available for this dataset.")
        st.stop()

    metrics_df = build_metrics_df(KEYS, res)
    best_rmse  = metrics_df['RMSE'].idxmin()
    best_r2    = metrics_df['R²'].idxmax()
    best_mape  = metrics_df['MAPE (%)'].idxmin()
    c1, c2, c3 = st.columns(3)
    c1.metric("Best RMSE", best_rmse, f"{metrics_df.loc[best_rmse,'RMSE']:.4f} {unit}")
    c2.metric("Best R²",   best_r2,   f"{metrics_df.loc[best_r2,'R²']:.4f}")
    c3.metric("Best MAPE", best_mape, f"{metrics_df.loc[best_mape,'MAPE (%)']:.2f}%")
    st.divider()

    st.subheader("Full Metrics Table")
    st.dataframe(styled_table(metrics_df), use_container_width=True)
    st.caption("Green = best value in column")
    st.divider()

    st.subheader("Metric Comparison")
    metric_sel       = st.selectbox("Select metric to plot", ['RMSE','MAE','MAPE (%)','R²'])
    lower_is_better  = metric_sel in ('RMSE','MAE','MAPE (%)')
    best_model       = (metrics_df[metric_sel].idxmin()
                        if lower_is_better else metrics_df[metric_sel].idxmax())
    bar_colours      = [PALETTE[i] if name != best_model else 'gold'
                        for i, name in enumerate(MODEL_NAMES)]
    fig = go.Figure(go.Bar(
        x=MODEL_NAMES, y=metrics_df[metric_sel].values,
        marker_color=bar_colours,
        text=[f'{v:.4f}' for v in metrics_df[metric_sel].values],
        textposition='outside',
    ))
    fig.update_layout(
        title=f'{metric_sel} — {"lower" if lower_is_better else "higher"} is better (gold = best)',
        yaxis_title=metric_sel, xaxis_title='Model', height=420)
    st.plotly_chart(fig, use_container_width=True)
    st.divider()

    st.subheader("Normalised Performance Radar  (all axes: higher = better)")
    norm = metrics_df.copy().astype(float)
    for m in ['RMSE','MAE','MAPE (%)']:
        lo, hi = norm[m].min(), norm[m].max()
        norm[m] = 1 - (norm[m] - lo) / (hi - lo + 1e-12)
    lo, hi = norm['R²'].min(), norm['R²'].max()
    norm['R²'] = (norm['R²'] - lo) / (hi - lo + 1e-12)
    cats   = ['RMSE','MAE','MAPE (%)','R²']
    fig_r  = go.Figure()
    for (model, row), colour in zip(norm.iterrows(), PALETTE):
        vals = list(row[cats]) + [row[cats[0]]]
        fig_r.add_trace(go.Scatterpolar(r=vals, theta=cats + [cats[0]],
                                        fill='toself', name=model,
                                        line_color=colour, opacity=0.55))
    fig_r.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,1])),
                        legend=dict(orientation='h', yanchor='bottom', y=-0.2),
                        height=520, margin=dict(t=40))
    st.plotly_chart(fig_r, use_container_width=True)
    st.divider()

    st.subheader("Deep Learning — Training Loss Curves")
    dl_keys  = ['lstm',   'gru',   'cnnlstm']
    dl_names = ['LSTM',   'GRU',   'CNN-LSTM']
    cols3    = st.columns(3)
    for col, hkey, hname in zip(cols3, dl_keys, dl_names):
        path = os.path.join(MODELS, f'hist_{hkey}_{hist_suffix}.pkl')
        try:
            with open(path, 'rb') as f: hist = pickle.load(f)
            fig_l = go.Figure()
            fig_l.add_trace(go.Scatter(y=hist['loss'],     name='Train', line_width=2))
            fig_l.add_trace(go.Scatter(y=hist['val_loss'], name='Val',
                                       line=dict(dash='dash', width=2)))
            fig_l.update_layout(title=f'{hname} — {ds_label}',
                                 xaxis_title='Epoch', yaxis_title='MSE Loss',
                                 height=300, margin=dict(t=40,b=30),
                                 legend=dict(orientation='h', y=1.15))
            col.plotly_chart(fig_l, use_container_width=True)
        except FileNotFoundError:
            col.info(f"Training history for {hname} not available yet.")


# ═══════════════════════════════════════════════════════════════════
# PAGE: RESEARCH RESULTS
# ═══════════════════════════════════════════════════════════════════
elif page == "🔬  Research Results":
    st.title("🔬 Research Results")
    st.markdown("""
    Publication-quality analysis: consolidated cross-dataset comparison,
    window-size ablation study, and statistical significance testing.
    """)

    # ── Tab layout ────────────────────────────────────────────────
    tab_cross, tab_uv_mv, tab_ablation, tab_dm = st.tabs([
        "📊 Cross-Dataset", "🔄 Uni- vs Multi-variate",
        "🪟 Window Ablation", "📐 Statistical Tests"
    ])

    # ── Cross-dataset comparison ──────────────────────────────────
    with tab_cross:
        st.subheader("Consolidated Results — All Datasets")
        all_rows = []
        if res_pjm:
            for name, key in zip(MODEL_NAMES, PJM_KEYS):
                rmse, mae, mape, r2 = compute_metrics(res_pjm[key]['true'], res_pjm[key]['pred'])
                all_rows.append({'Dataset':'PJM East','Model':name,
                                 'RMSE':rmse,'MAE':mae,'MAPE (%)':mape,'R²':r2})
        if res_aep:
            for name, key in zip(MODEL_NAMES, AEP_KEYS):
                rmse, mae, mape, r2 = compute_metrics(res_aep[key]['true'], res_aep[key]['pred'])
                all_rows.append({'Dataset':'AEP Hourly','Model':name,
                                 'RMSE':rmse,'MAE':mae,'MAPE (%)':mape,'R²':r2})
        if res_pjm:
            for name, key in zip(MODEL_NAMES, UCI_KEYS):
                rmse, mae, mape, r2 = compute_metrics(res_pjm[key]['true'], res_pjm[key]['pred'])
                all_rows.append({'Dataset':'UCI (Univariate)','Model':name,
                                 'RMSE':rmse,'MAE':mae,'MAPE (%)':mape,'R²':r2})
        if res_mv:
            for name, key in zip(MODEL_NAMES, UCIMV_KEYS):
                rmse, mae, mape, r2 = compute_metrics(res_mv[key]['true'], res_mv[key]['pred'])
                all_rows.append({'Dataset':'UCI (Multivariate)','Model':name,
                                 'RMSE':rmse,'MAE':mae,'MAPE (%)':mape,'R²':r2})

        if all_rows:
            all_df = pd.DataFrame(all_rows)
            st.dataframe(all_df.round(4).set_index(['Dataset','Model']),
                         use_container_width=True)

            # R² comparison grouped bar
            fig = px.bar(all_df, x='Model', y='R²', color='Dataset',
                         barmode='group', title='R² Comparison Across All Datasets',
                         color_discrete_sequence=['#4472C4','#70AD47','#ED7D31','#7030A0'])
            fig.update_layout(height=420, xaxis_title='Model',
                               legend=dict(orientation='h', yanchor='bottom', y=1.02))
            st.plotly_chart(fig, use_container_width=True)

            # RMSE comparison
            fig2 = px.bar(all_df[all_df['Dataset'].isin(['UCI (Univariate)','UCI (Multivariate)'])],
                          x='Model', y='RMSE', color='Dataset', barmode='group',
                          title='UCI RMSE — Univariate vs Multivariate',
                          color_discrete_sequence=['#ED7D31','#7030A0'])
            fig2.update_layout(height=380)
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("Run the training notebooks to populate this section.")

    # ── Univariate vs Multivariate ────────────────────────────────
    with tab_uv_mv:
        st.subheader("Univariate vs Multivariate — UCI Household")
        if res_mv is None:
            st.info("Run notebook 07 (07_multivariate_uci.ipynb) to see this comparison.")
        else:
            rows_mv = []
            for name, uk, mk in zip(MODEL_NAMES, UCI_KEYS, UCIMV_KEYS):
                _, _, _, r1  = compute_metrics(res_pjm[uk]['true'], res_pjm[uk]['pred'])
                rmse2, _, _, r2v = compute_metrics(res_mv[mk]['true'], res_mv[mk]['pred'])
                rmse1, _, _, _   = compute_metrics(res_pjm[uk]['true'], res_pjm[uk]['pred'])
                rows_mv.append({'Model':name, 'UV R²':round(r1,4), 'MV R²':round(r2v,4),
                                 'Δ R²':round(r2v-r1,4),
                                 'UV RMSE':round(rmse1,4),'MV RMSE':round(rmse2,4)})
            mv_df = pd.DataFrame(rows_mv).set_index('Model')
            st.dataframe(mv_df, use_container_width=True)

            fig = go.Figure()
            fig.add_trace(go.Bar(name='Univariate', x=MODEL_NAMES,
                                 y=mv_df['UV R²'], marker_color='#ED7D31'))
            fig.add_trace(go.Bar(name='Multivariate', x=MODEL_NAMES,
                                 y=mv_df['MV R²'], marker_color='#7030A0'))
            fig.update_layout(barmode='group', title='R² — Univariate vs Multivariate (UCI)',
                               yaxis_title='R²', height=400)
            st.plotly_chart(fig, use_container_width=True)

    # ── Window ablation ───────────────────────────────────────────
    with tab_ablation:
        st.subheader("Window-Size Ablation Study — PJM East")
        if ablation is None:
            st.info("Run notebook 08 (08_window_ablation.ipynb) to see this analysis.")
        else:
            WINDOWS = [24, 48, 168]
            abl_rows = []
            for W in WINDOWS:
                for model in ['RF','XGB','LSTM']:
                    d = ablation[W][model]
                    abl_rows.append({'Window': f'{W}h', 'Model': model,
                                     'RMSE': round(d['RMSE'],2),
                                     'MAE':  round(d['MAE'],2),
                                     'MAPE (%)': round(d['MAPE'],2),
                                     'R²':   round(d['R2'],4)})
            abl_df = pd.DataFrame(abl_rows)
            st.dataframe(abl_df.set_index(['Window','Model']), use_container_width=True)

            # Line plot: R² vs window
            fig = go.Figure()
            colours_m = {'RF':'#4472C4','XGB':'#ED7D31','LSTM':'#70AD47'}
            markers_m = {'RF':'circle','XGB':'square','LSTM':'triangle-up'}
            for model in ['RF','XGB','LSTM']:
                sub = abl_df[abl_df['Model']==model]
                r2s = [ablation[W][model]['R2'] for W in WINDOWS]
                fig.add_trace(go.Scatter(x=[f'{W}h' for W in WINDOWS], y=r2s,
                                         name=model, mode='lines+markers',
                                         line=dict(color=colours_m[model], width=2.5),
                                         marker=dict(symbol=markers_m[model], size=10)))
            fig.update_layout(title='R² vs Look-back Window Size — PJM East',
                               xaxis_title='Window', yaxis_title='R²', height=420)
            st.plotly_chart(fig, use_container_width=True)

            fig2 = go.Figure()
            for model in ['RF','XGB','LSTM']:
                rmses = [ablation[W][model]['RMSE'] for W in WINDOWS]
                fig2.add_trace(go.Scatter(x=[f'{W}h' for W in WINDOWS], y=rmses,
                                          name=model, mode='lines+markers',
                                          line=dict(color=colours_m[model], width=2.5),
                                          marker=dict(symbol=markers_m[model], size=10)))
            fig2.update_layout(title='RMSE vs Look-back Window Size — PJM East',
                                xaxis_title='Window', yaxis_title='RMSE (MW)', height=380)
            st.plotly_chart(fig2, use_container_width=True)

    # ── Statistical tests ─────────────────────────────────────────
    with tab_dm:
        st.subheader("Diebold-Mariano Statistical Significance Tests")
        st.markdown("""
        Tests whether the best-performing model's forecast accuracy is **statistically
        significantly better** than each competitor (Harvey, Leybourne & Newbold, 1997 correction).
        H₀: equal predictive accuracy.
        """)

        from scipy import stats

        def dm_test(y_true, pred1, pred2):
            e1 = (y_true - pred1) ** 2
            e2 = (y_true - pred2) ** 2
            d  = e1 - e2
            T  = len(d)
            d_bar = np.mean(d)
            var_d = np.var(d, ddof=1) / T
            dm    = d_bar / np.sqrt(max(var_d, 1e-12))
            dm_c  = dm * np.sqrt((T+1)/T)
            p     = float(2 * stats.t.sf(np.abs(dm_c), df=T-1))
            return float(dm_c), p

        available = {}
        if res_pjm: available['PJM East'] = (PJM_KEYS, res_pjm)
        if res_aep: available['AEP Hourly'] = (AEP_KEYS, res_aep)
        if res_pjm: available['UCI (Univariate)'] = (UCI_KEYS, res_pjm)
        if res_mv:  available['UCI (Multivariate)'] = (UCIMV_KEYS, res_mv)

        if not available:
            st.info("No results available yet.")
        else:
            for ds_name, (keys, res_d) in available.items():
                st.markdown(f"**{ds_name}**")
                rmses = {k: float(np.sqrt(mean_squared_error(res_d[k]['true'], res_d[k]['pred'])))
                         for k in keys}
                best_key  = min(rmses, key=rmses.get)
                best_name = MODEL_NAMES[keys.index(best_key)]
                best_pred = res_d[best_key]['pred']
                best_true = res_d[best_key]['true']

                dm_rows = []
                for name, key in zip(MODEL_NAMES, keys):
                    if key == best_key:
                        dm_rows.append({'Model':name,'DM stat':'—','p-value':'—',
                                        'Sig. (p<0.05)':'Reference (best)'})
                        continue
                    dm, p = dm_test(best_true, res_d[key]['pred'], best_pred)
                    dm_rows.append({'Model':name,
                                    'DM stat':f'{dm:.3f}',
                                    'p-value':f'{p:.4f}',
                                    'Sig. (p<0.05)':'✓ Yes' if p < 0.05 else '✗ No'})
                dm_df = pd.DataFrame(dm_rows).set_index('Model')
                st.dataframe(dm_df, use_container_width=True)
                st.caption(f"Reference model: **{best_name}** (lowest RMSE={rmses[best_key]:.4f})")
                st.markdown("---")
