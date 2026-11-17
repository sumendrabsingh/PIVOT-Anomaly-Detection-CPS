import os
import glob
import io
import streamlit as st
import pandas as pd
import numpy as np
import torch
import plotly.graph_objects as go
import json
import time
from datetime import datetime, timedelta

# ==========================================
# PAGE CONFIG
# ==========================================
st.set_page_config(page_title="PIVOT SCADA / HMI", layout="wide", page_icon="🛡️")

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
WINDOW_SIZE = 25

# Colors for UI
COLOR_NORMAL = "#00FF00"
COLOR_WARNING = "#FFA500"
COLOR_ALERT = "#FF0000"

TIME_COL_CANDIDATES = ["Timestamp", "timestamp", "Time", "time", "Datetime", "datetime", "Date", "date"]

hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

# ==========================================
# PLANT PRESETS
# ==========================================
def get_dataset_config(network_name):
    if "SWaT" in network_name:
        return {
            "default_data": "Datasets/swat_test_preprocessed.csv",
            "oracle_prefix": "subsystem",
            "label_col": "Normal/Attack",
            "label_attack_val": "attack",
            "default_start_time": "2015-12-28 10:00:00",
            "sample_period_s": 1.0,
        }
    elif "WADI" in network_name:
        return {
            "default_data": "Datasets/wadi_test_preprocessed.csv",
            "oracle_prefix": "wadi_subsystem",
            "label_col": "Normal/Attack",
            "label_attack_val": "1",
            "default_start_time": "2017-10-09 18:00:00",
            "sample_period_s": 1.0,
        }
    else:  # HAI
        return {
            "default_data": "Datasets/hai_test_engineered.csv",
            "oracle_prefix": "hai_subsystem",
            "label_col": "Label",
            "label_attack_val": "1",
            "default_start_time": "2023-05-01 00:00:00",
            "sample_period_s": 1.0,
        }

# ==========================================
# SIDEBAR — PLANT / DATA SOURCE / MODELS
# ==========================================
st.sidebar.title("🌐 Network Configuration")
selected_network = st.sidebar.selectbox("Select Plant Target", ["SWaT (Water Treatment)", "WADI (Water Distribution)", "HAI (Hardware-In-the-Loop)"])
CONFIG = get_dataset_config(selected_network)

# ---------- Data source ----------
with st.sidebar.expander("📁 Data Source", expanded=False):
    feed_mode = st.radio("Mode", ["Dataset Replay (offline)", "Live Plant Feed (online)"], key="feed_mode")

    available_csvs = sorted(glob.glob(os.path.join("Datasets", "*.csv")))
    if feed_mode.startswith("Dataset"):
        options = available_csvs + ["Custom path…"]
        default_idx = options.index(CONFIG["default_data"].replace("/", os.sep)) if CONFIG["default_data"].replace("/", os.sep) in options else 0
        picked = st.selectbox("Replay dataset (CSV)", options, index=default_idx)
        if picked == "Custom path…":
            data_path = st.text_input("Path to dataset CSV", CONFIG["default_data"])
        else:
            data_path = picked
        live_source_type = None
        live_endpoint = None
        poll_interval = None
        calib_path = data_path
    else:
        live_source_type = st.radio("Live source type", [
            "Streaming CSV (appended by gateway/historian)",
            "HTTP JSON endpoint (polled)"
        ])
        if "CSV" in live_source_type:
            live_endpoint = st.text_input("Path to streaming CSV", "live/plant_feed.csv",
                                          help="Any process (OPC-UA/Modbus gateway, historian export, "
                                               "or simulate_live_feed.py) that appends rows to this file.")
        else:
            live_endpoint = st.text_input("Endpoint URL", "http://localhost:9000/latest",
                                          help="GET returning a JSON record {col: value, ...} or a list of records "
                                               "for the newest samples.")
        poll_interval = st.number_input("Poll interval (seconds)", min_value=0.2, max_value=60.0, value=1.0, step=0.2)
        st.markdown("**Calibration reference** — a normal-period dataset used to fit the "
                    "scaler and per-subsystem alarm calibration (live data has no labels):")
        calib_options = available_csvs if available_csvs else ["<none found>"]
        default_idx = calib_options.index(CONFIG["default_data"].replace("/", os.sep)) if CONFIG["default_data"].replace("/", os.sep) in calib_options else 0
        calib_path = st.selectbox("Calibration dataset (CSV)", calib_options, index=default_idx)
        data_path = None

# ---------- Model source ----------
with st.sidebar.expander("🧠 Model Source", expanded=False):
    models_dir = st.text_input("Oracle bundles directory", "Oracles")
    oracle_prefix = st.text_input("Bundle filename prefix", CONFIG["oracle_prefix"],
                                  help="Bundles are matched as <prefix>_<SubsystemName>.pth")
    discovered = sorted([os.path.basename(p) for p in glob.glob(os.path.join(models_dir, oracle_prefix + "*.pth"))]) if os.path.isdir(models_dir) else []
    # Guard against the SWaT prefix "subsystem" swallowing wadi_/hai_ bundles is
    # unnecessary (startswith on the basename), but exclude exact cross-prefix hits:
    if oracle_prefix == "subsystem":
        discovered = [f for f in discovered if not (f.startswith("wadi_") or f.startswith("hai_"))]
    selected_bundles = st.multiselect("Oracle bundles to load", discovered, default=discovered)
    if not discovered:
        st.warning(f"No bundles matching `{oracle_prefix}*.pth` in `{models_dir}`")

# ---------- Plant clock ----------
with st.sidebar.expander("🕒 Plant Clock", expanded=False):
    st.caption("Used only when the dataset has no timestamp column (a synthesized "
               "clock is then derived as start time + index × sample period).")
    clock_start_str = st.text_input("Plant start time (YYYY-MM-DD HH:MM:SS)", CONFIG["default_start_time"])
    sample_period = st.number_input("Sample period (seconds)", min_value=0.001, value=float(CONFIG["sample_period_s"]), step=0.5)

st.sidebar.markdown("---")
defense_mode = st.sidebar.radio("Active Defense Mechanism", [
    "Dual-Threshold (Standard Pooled)",
    "Smoothed OR-Voting (W=25)",
    "Raw OR-Voting (Lockdown Mode)"
])

st.sidebar.markdown("---")
st.sidebar.markdown("**Sensitivity Controls**")

if 'mem_thresh_dual' not in st.session_state: st.session_state.mem_thresh_dual = 4.0
if 'mem_thresh_smooth' not in st.session_state: st.session_state.mem_thresh_smooth = 2.5
if 'mem_thresh_raw' not in st.session_state: st.session_state.mem_thresh_raw = 2.5

def save_dual(): st.session_state.mem_thresh_dual = st.session_state.thresh_dual
def save_smooth(): st.session_state.mem_thresh_smooth = st.session_state.thresh_smooth
def save_raw(): st.session_state.mem_thresh_raw = st.session_state.thresh_raw

if "Dual-Threshold" in defense_mode:
    threshold_limit = st.sidebar.slider("Global Alarm Threshold (Pooled)", min_value=1.0, max_value=15.0, value=st.session_state.mem_thresh_dual, step=0.5, key="thresh_dual", on_change=save_dual)
elif "Smoothed" in defense_mode:
    threshold_limit = st.sidebar.slider("Strict Lockdown Threshold (Smoothed)", min_value=1.0, max_value=15.0, value=st.session_state.mem_thresh_smooth, step=0.5, key="thresh_smooth", on_change=save_smooth)
else:
    threshold_limit = st.sidebar.slider("Strict Lockdown Threshold (Raw)", min_value=1.0, max_value=15.0, value=st.session_state.mem_thresh_raw, step=0.5, key="thresh_raw", on_change=save_raw)

# ==========================================
# MODEL DEFINITION & LOADING
# ==========================================
class SubsystemAutoencoder(torch.nn.Module):
    def __init__(self, input_dim, bottleneck_dim):
        super().__init__()
        if input_dim <= 8:
            hidden1 = 32; hidden2 = 16
        elif input_dim <= 16:
            hidden1 = 64; hidden2 = 32
        elif input_dim <= 32:
            hidden1 = 128; hidden2 = 64
        else:
            hidden1 = 256; hidden2 = 128

        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden1), torch.nn.BatchNorm1d(hidden1), torch.nn.ReLU(), torch.nn.Dropout(0.2),
            torch.nn.Linear(hidden1, hidden2), torch.nn.BatchNorm1d(hidden2), torch.nn.ReLU(), torch.nn.Dropout(0.2),
            torch.nn.Linear(hidden2, bottleneck_dim)
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(bottleneck_dim, hidden2), torch.nn.BatchNorm1d(hidden2), torch.nn.ReLU(), torch.nn.Dropout(0.2),
            torch.nn.Linear(hidden2, hidden1), torch.nn.BatchNorm1d(hidden1), torch.nn.ReLU(),
            torch.nn.Linear(hidden1, input_dim)
        )
    def forward(self, x):
        return self.decoder(self.encoder(x))

@st.cache_resource
def load_models(models_dir, prefix, bundle_files):
    models = {}
    for file in bundle_files:
        name = file.replace(prefix + "_", "").replace(".pth", "")
        path = os.path.join(models_dir, file)
        checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)

        model = SubsystemAutoencoder(checkpoint['input_dim'], checkpoint['bottleneck_dim']).to(DEVICE)
        model.load_state_dict(checkpoint['model_state'])
        model.eval()

        features = checkpoint.get('features', None)

        # SWaT-style bundles may be trained on flattened SEQ_LEN-step windows
        # (input_dim = seq_len * n_features); WADI/HAI are point-wise.
        seq_len = 1
        if features and checkpoint['input_dim'] % len(features) == 0:
            seq_len = checkpoint['input_dim'] // len(features)

        models[name] = {
            'model': model,
            'features': features,
            'seq_len': seq_len,
            'file': file,
        }
    return models

MODELS = load_models(models_dir, oracle_prefix, tuple(selected_bundles))
MAX_SEQ_LEN = max([md['seq_len'] for md in MODELS.values()], default=1)
MIN_ROWS = WINDOW_SIZE + MAX_SEQ_LEN + 1

if not MODELS:
    st.error("No oracle bundles loaded — pick at least one in **🧠 Model Source**.")
    st.stop()

# ==========================================
# DATA LOADING / TIMESTAMPS
# ==========================================
def detect_time_col(columns):
    for c in TIME_COL_CANDIDATES:
        if c in columns:
            return c
    return None

def attach_labels(df, label_col, attack_val):
    if label_col in df.columns:
        lbl = df[label_col].astype(str).str.strip().str.replace(" ", "").str.lower()
        df['Is_Attack'] = lbl.apply(lambda x: 1 if attack_val.lower() in x else 0)
        df['Has_Labels'] = True
    else:
        df['Is_Attack'] = 0
        df['Has_Labels'] = False
    return df

@st.cache_data
def load_dataset_cached(path, label_col, attack_val):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return attach_labels(df, label_col, attack_val)

def timestamp_for(df, idx, time_col, clock_start_str, sample_period):
    """Actual timestamp from the dataset when available, else synthesized."""
    if time_col is not None:
        return str(df.iloc[idx][time_col]).strip(), False
    try:
        start = pd.Timestamp(clock_start_str)
        return str(start + pd.Timedelta(seconds=float(idx) * sample_period)), True
    except Exception:
        return f"index {idx}", True

# ---------- Reference (calibration) dataset ----------
try:
    df_ref = load_dataset_cached(calib_path, CONFIG["label_col"], CONFIG["label_attack_val"])
except Exception as e:
    st.error(f"Could not load {'calibration' if data_path is None else 'replay'} dataset "
             f"`{calib_path}`: {e}\n\nGenerate it with the preprocessing scripts (see README).")
    st.stop()

missing_feats = sorted({f for md in MODELS.values() for f in md['features'] if f not in df_ref.columns})
if missing_feats:
    st.error(f"The selected dataset is missing {len(missing_feats)} feature(s) required by the "
             f"loaded oracle bundles (e.g. {', '.join(missing_feats[:8])}). "
             f"Check that the plant target, dataset, and bundles belong together.")
    st.stop()

@st.cache_data
def get_baseline_stats(path, label_col, attack_val):
    """Per-feature mean/std over the reference dataset's normal rows
    (ALIGN_BASELINE convention from the research eval runners)."""
    df = load_dataset_cached(path, label_col, attack_val)
    normal = df[df['Is_Attack'] == 0]
    if len(normal) == 0:
        normal = df
    mean = normal.mean(numeric_only=True)
    std = normal.std(numeric_only=True)
    std[std < 1e-4] = 1.0
    return mean, std

BASELINE_MEAN, BASELINE_STD = get_baseline_stats(calib_path, CONFIG["label_col"], CONFIG["label_attack_val"])

@st.cache_resource
def calibrate_subsystems(path, label_col, attack_val, bundle_key):
    """Per-subsystem normal-period P99.5 of the per-timestep max feature
    reconstruction error on the reference dataset. Anomaly scores are relative
    to this, so 'score > 1' means 'worse than the normal period ever gets'."""
    df = load_dataset_cached(path, label_col, attack_val)
    mean, std = get_baseline_stats(path, label_col, attack_val)
    sample = df.iloc[::10]
    normal_mask = (sample['Is_Attack'].values == 0)
    if normal_mask.sum() == 0:
        normal_mask = np.ones(len(sample), dtype=bool)
    cals = {}
    with torch.no_grad():
        for name, md in MODELS.items():
            feats = md['features']
            seq_len = md['seq_len']
            scaled = (sample[feats].fillna(0).values - mean[feats].values) / std[feats].values
            if seq_len > 1:
                # Approximate steady-state windows by repeating each point
                scaled = np.tile(scaled, (1, seq_len))
            errs = []
            X_all = torch.tensor(scaled, dtype=torch.float32)
            for i in range(0, len(X_all), 8192):
                X = X_all[i:i + 8192].to(DEVICE)
                R = md['model'](X)
                sq = ((X - R) ** 2)
                if seq_len > 1:
                    sq = sq.view(-1, seq_len, len(feats)).mean(dim=1)
                errs.append(sq.max(dim=1).values.cpu().numpy())
            e = np.concatenate(errs)[normal_mask]
            cals[name] = float(max(np.percentile(e, 99.5), 1e-6))
    return cals

CALIBRATION = calibrate_subsystems(calib_path, CONFIG["label_col"], CONFIG["label_attack_val"],
                                   tuple(sorted(md['file'] for md in MODELS.values())))

# ==========================================
# LIVE FEED READERS
# ==========================================
def read_csv_tail(path, n_rows):
    """Efficiently read the header + last n_rows of a (possibly large) growing CSV."""
    with open(path, 'rb') as f:
        header = f.readline()
        f.seek(0, os.SEEK_END)
        size = f.tell()
        back = min(size, 8_000_000)
        f.seek(size - back)
        chunk = f.read().split(b"\n")
    # Drop the first element: it is either the header (small file) or a partial line
    body = [ln for ln in chunk[1:] if ln.strip()]
    body = body[-n_rows:]
    df = pd.read_csv(io.BytesIO(header + b"\n".join(body)))
    df.columns = df.columns.str.strip()
    return df

def poll_http_endpoint(url):
    """GET the newest record(s) from a JSON endpoint; returns a DataFrame."""
    import requests
    r = requests.get(url, timeout=5)
    r.raise_for_status()
    payload = r.json()
    if isinstance(payload, dict):
        payload = [payload]
    df = pd.DataFrame(payload)
    df.columns = df.columns.str.strip()
    return df

# ==========================================
# PROCESSING LOGIC
# ==========================================
def preprocess_window(df_window, model_data):
    features = model_data['features']

    raw_vals = df_window[features].fillna(0).values
    # ALIGN_BASELINE convention (same as the research eval runners): standardize
    # against the reference dataset's normal-period statistics. This also absorbs
    # features that were constant during training but drifted afterwards.
    scaled_vals = (raw_vals - BASELINE_MEAN[features].values) / BASELINE_STD[features].values

    seq_len = model_data['seq_len']
    if seq_len > 1:
        # Build flattened sliding windows matching the model's training format
        if len(scaled_vals) < seq_len:
            return None
        windows = np.stack([
            scaled_vals[i:i + seq_len].flatten()
            for i in range(len(scaled_vals) - seq_len + 1)
        ])
        return torch.tensor(windows, dtype=torch.float32).to(DEVICE)
    return torch.tensor(scaled_vals, dtype=torch.float32).to(DEVICE)

def evaluate_window(df_window, mode, threshold_limit):
    """Score the newest timestep given its trailing window of samples."""
    if len(df_window) < min(MIN_ROWS, WINDOW_SIZE + 1):
        return None, None, False

    subsystem_errors = {}
    sensor_errors = {}

    with torch.no_grad():
        for name, model_data in MODELS.items():
            model = model_data['model']
            features = model_data['features']
            data_tensor = preprocess_window(df_window, model_data)

            if data_tensor is not None and len(data_tensor) >= WINDOW_SIZE:
                reconstructed_batch = model(data_tensor)
                squared_diff = ((data_tensor - reconstructed_batch) ** 2)

                num_features = len(features)
                seq_len = model_data['seq_len']
                if seq_len > 1:
                    # Collapse the flattened (seq_len * n_features) window error
                    # back to a per-feature error by averaging over time
                    squared_diff = squared_diff.view(-1, seq_len, num_features).mean(dim=1)

                # Take the last WINDOW_SIZE frames for smoothing
                current_sensor_errors_batch = squared_diff[-WINDOW_SIZE:].cpu().numpy()

                smoothed_sensor_errors = np.mean(current_sensor_errors_batch, axis=0)
                raw_sensor_errors = current_sensor_errors_batch[-1]

                active_errors = raw_sensor_errors if "Raw" in mode else smoothed_sensor_errors
                max_error = float(np.max(active_errors))

                # Score = error relative to this subsystem's normal-period
                # P99.5 reconstruction error (calibrated once at startup)
                z_score = max_error / CALIBRATION.get(name, 1.0)
                subsystem_errors[name] = z_score

                if seq_len > 1:
                    last_actual = data_tensor[-1].view(seq_len, num_features)[-1]
                    last_expected = reconstructed_batch[-1].view(seq_len, num_features)[-1]
                else:
                    last_actual = data_tensor[-1]
                    last_expected = reconstructed_batch[-1]
                sensor_data = {}
                for i, feat in enumerate(features):
                    sensor_data[feat] = {
                        "error": float(active_errors[i]),
                        "actual": float(last_actual[i].item()),
                        "expected": float(last_expected[i].item())
                    }
                sensor_errors[name] = sensor_data

    is_alert = False
    if subsystem_errors:
        if "Dual-Threshold" in mode:
            if np.mean(list(subsystem_errors.values())) > threshold_limit:
                is_alert = True
        else:
            for score in subsystem_errors.values():
                if score > threshold_limit:
                    is_alert = True
                    break

    return subsystem_errors, sensor_errors, is_alert

# ==========================================
# SHARED RESULT PANELS
# ==========================================
def render_verdict(subsystem_errors, is_alert):
    st.markdown("---")
    if subsystem_errors is None:
        st.warning("Not enough data to form a detection window yet.")
    elif is_alert:
        st.markdown(f"<h1 style='text-align: center; color: {COLOR_ALERT}; background-color: #330000; padding: 20px; border-radius: 10px;'>CRITICAL ALERT: ANOMALY DETECTED</h1>", unsafe_allow_html=True)
    else:
        st.markdown(f"<h1 style='text-align: center; color: {COLOR_NORMAL}; background-color: #003300; padding: 20px; border-radius: 10px;'>SYSTEM SECURE: ALL CLEAR</h1>", unsafe_allow_html=True)

def render_overview(subsystem_errors):
    st.markdown("---")
    st.subheader("Plant Overview (Subsystem Health)")
    if subsystem_errors is not None:
        stages = list(subsystem_errors.keys())
        scores = list(subsystem_errors.values())
        colors = [COLOR_ALERT if s > threshold_limit else COLOR_NORMAL for s in scores]

        fig = go.Figure(data=[go.Bar(name='Score', x=stages, y=scores, marker_color=colors)])
        fig.add_hline(y=threshold_limit, line_dash="dash", line_color="red", annotation_text="Alarm Threshold")
        fig.update_layout(title="Real-Time Subsystem Stress Levels", yaxis_title="Anomaly Score (× normal-period P99.5 error)", template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

def render_xai(subsystem_errors, sensor_errors, is_alert):
    st.markdown("---")
    if is_alert and subsystem_errors is not None:
        st.subheader("Explainable AI (XAI) Drill-Down")
        failing_stages = [stage for stage, score in subsystem_errors.items() if score > threshold_limit]
        if not failing_stages: failing_stages = [max(subsystem_errors, key=subsystem_errors.get)]

        st.error(f"📍 **Failure Locations Detected**: {', '.join(failing_stages)}")

        st.markdown("### Exact Physical Deviations")
        for stage in failing_stages:
            drifting_sensors = sensor_errors[stage]
            top_sensors = sorted(drifting_sensors.items(), key=lambda x: x[1]['error'], reverse=True)[:3]
            for sensor, data in top_sensors:
                st.warning(f"⚠️ **[{stage}] {sensor}**: MSE Reconstruction Deviation: **{data['error']:.4f}** "
                           f"(actual {data['actual']:.3f} vs expected {data['expected']:.3f}, standardized units)")

        all_drifting = []
        for stage in failing_stages:
            for s_name, s_data in sensor_errors[stage].items():
                all_drifting.append((f"{s_name} ({stage.split('_')[0]})", s_data['error']))

        top_overall = sorted(all_drifting, key=lambda x: x[1], reverse=True)[:5]
        fig2 = go.Figure(data=[go.Bar(x=[x[0] for x in top_overall], y=[x[1] for x in top_overall], marker_color=COLOR_WARNING)])
        fig2.update_layout(title="Top 5 Drifting Sensors Overall", yaxis_title="Mean Squared Error", template="plotly_dark")
        st.plotly_chart(fig2, use_container_width=True)

# ==========================================
# MODE: DATASET REPLAY
# ==========================================
if feed_mode.startswith("Dataset"):
    df_full = df_ref
    time_col = detect_time_col(df_full.columns)

    st.title(f"🛡️ PIVOT Plant Security: {selected_network.split(' ')[0]}")
    st.markdown(f"Subsystem PINN Cyber-Physical Anomaly Detection Dashboard — "
                f"replaying `{data_path}` against {len(MODELS)} oracle bundle(s)")

    st.markdown("**Timeline Simulator**")

    if 'current_step' not in st.session_state: st.session_state.current_step = 500
    if 'is_playing' not in st.session_state: st.session_state.is_playing = False

    def toggle_play(): st.session_state.is_playing = not st.session_state.is_playing

    col_play, col_speed, col_search = st.columns([1, 1, 2])
    with col_play:
        st.button("▶️ Play / ⏸️ Pause", on_click=toggle_play)
    with col_speed:
        playback_speed = st.selectbox("Playback Speed", ["1x (Real-time)", "5x Fast Forward", "10x Ultra Fast"])
        speed_map = {"1x (Real-time)": 1.0, "5x Fast Forward": 0.2, "10x Ultra Fast": 0.05}
    with col_search:
        search_label = time_col if time_col else "index"
        search_time = st.text_input(f"Search Exact {search_label} (e.g., Row Index or Timestamp)", "")
        if search_time:
            if time_col:
                matches = df_full[df_full[time_col].astype(str).str.contains(search_time.strip(), case=False, na=False, regex=False)]
            else:
                matches = df_full[df_full.index.astype(str) == search_time.strip()]
            if not matches.empty:
                st.success(f"✅ Found match at Index: **{matches.index[0]}**")

    if 'num_val' not in st.session_state: st.session_state.num_val = st.session_state.current_step
    if 'slider_val' not in st.session_state: st.session_state.slider_val = st.session_state.current_step

    def on_num_change():
        st.session_state.current_step = st.session_state.num_val
        st.session_state.slider_val = st.session_state.current_step
    def on_slider_change():
        st.session_state.current_step = st.session_state.slider_val
        st.session_state.num_val = st.session_state.current_step

    st.session_state.current_step = min(st.session_state.current_step, len(df_full) - 1)
    st.session_state.num_val = st.session_state.current_step
    st.session_state.slider_val = st.session_state.current_step

    col_sl1, col_sl2 = st.columns([1, 4])
    with col_sl1:
        st.number_input("Jump to Index", min_value=MIN_ROWS, max_value=len(df_full)-1, key="num_val", on_change=on_num_change)
    with col_sl2:
        st.slider("Scrub Timeline", min_value=MIN_ROWS, max_value=len(df_full)-1, key="slider_val", on_change=on_slider_change)

    timestep = st.session_state.current_step
    ts_str, synthesized = timestamp_for(df_full, timestep, time_col, clock_start_str, sample_period)
    is_ground_truth_attack = df_full.iloc[timestep]['Is_Attack'] == 1
    has_labels = bool(df_full['Has_Labels'].iloc[0])

    col_stat1, col_stat2 = st.columns([1, 1])
    with col_stat1:
        suffix = " *(synthesized)*" if synthesized else ""
        st.info(f"🕒 **Plant Clock:** index `{timestep}` — `{ts_str}`{suffix}")
    with col_stat2:
        if not has_labels:
            st.info("⚪ REALITY: no ground-truth labels in this dataset.")
        elif is_ground_truth_attack:
            st.error(f"🔴 REALITY: Cyber Attack occurring at index {timestep}!")
        else:
            st.success(f"🟢 REALITY: Systems operating normally.")

    start_idx = max(0, timestep - WINDOW_SIZE - MAX_SEQ_LEN)
    subsystem_errors, sensor_errors, is_alert = evaluate_window(
        df_full.iloc[start_idx: timestep + 1], defense_mode, threshold_limit)

    render_verdict(subsystem_errors, is_alert)
    render_overview(subsystem_errors)
    render_xai(subsystem_errors, sensor_errors, is_alert)

    if st.session_state.is_playing:
        time.sleep(speed_map[playback_speed])
        if st.session_state.current_step < len(df_full) - 1:
            st.session_state.current_step += 1
            st.rerun()
        else:
            st.session_state.is_playing = False

# ==========================================
# MODE: LIVE PLANT FEED
# ==========================================
else:
    st.title(f"🛡️ PIVOT Plant Security: {selected_network.split(' ')[0]} — LIVE")
    st.markdown(f"Monitoring **{live_endpoint}** — scaler/alarms calibrated on `{calib_path}` "
                f"({len(MODELS)} oracle bundle(s))")

    if 'live_on' not in st.session_state: st.session_state.live_on = False
    if 'live_buffer' not in st.session_state: st.session_state.live_buffer = pd.DataFrame()
    if 'live_history' not in st.session_state: st.session_state.live_history = []
    if 'live_rows_seen' not in st.session_state: st.session_state.live_rows_seen = 0

    def toggle_live():
        st.session_state.live_on = not st.session_state.live_on
        if st.session_state.live_on:
            st.session_state.live_history = []

    col_btn, col_status = st.columns([1, 3])
    with col_btn:
        st.button("📡 Start / ⏹️ Stop Monitoring", on_click=toggle_live)
    with col_status:
        if st.session_state.live_on:
            st.success("Monitoring ACTIVE — polling every %.1fs" % poll_interval)
        else:
            st.info("Monitoring stopped. Press Start to begin polling the live source.")

    err_box = st.empty()
    df_window = None
    if st.session_state.live_on:
        try:
            if "CSV" in live_source_type:
                df_window = read_csv_tail(live_endpoint, MIN_ROWS + 5)
            else:
                new_rows = poll_http_endpoint(live_endpoint)
                buf = pd.concat([st.session_state.live_buffer, new_rows], ignore_index=True)
                st.session_state.live_buffer = buf.tail(max(MIN_ROWS * 4, 500)).reset_index(drop=True)
                df_window = st.session_state.live_buffer
            st.session_state.live_rows_seen = len(df_window)
        except FileNotFoundError:
            err_box.error(f"Live source `{live_endpoint}` not found. Start your gateway/historian export, "
                          f"or demo with: `python simulate_live_feed.py --source {CONFIG['default_data']} --out {live_endpoint}`")
        except Exception as e:
            err_box.error(f"Failed to read live source: {e}")

    if df_window is not None and len(df_window) > 0:
        missing = sorted({f for md in MODELS.values() for f in md['features'] if f not in df_window.columns})
        if missing:
            err_box.error(f"Live feed is missing required feature(s): {', '.join(missing[:8])}…")
        else:
            time_col = detect_time_col(df_window.columns)
            last_idx = len(df_window) - 1
            ts_str, synthesized = timestamp_for(df_window, last_idx, time_col, clock_start_str, sample_period)
            if time_col is None:
                ts_str, synthesized = datetime.now().strftime("%Y-%m-%d %H:%M:%S"), True

            subsystem_errors, sensor_errors, is_alert = evaluate_window(df_window, defense_mode, threshold_limit)

            m1, m2, m3 = st.columns(3)
            m1.metric("Samples in window", st.session_state.live_rows_seen)
            m2.metric("Latest plant sample", ts_str + (" (receive time)" if synthesized and time_col is None else ""))
            m3.metric("Alarm state", "🔴 ALERT" if is_alert else "🟢 CLEAR")

            if subsystem_errors:
                st.session_state.live_history.append(
                    {"t": ts_str, **{k: float(v) for k, v in subsystem_errors.items()}})
                st.session_state.live_history = st.session_state.live_history[-300:]

            render_verdict(subsystem_errors, is_alert)
            render_overview(subsystem_errors)

            if len(st.session_state.live_history) > 1:
                st.markdown("---")
                st.subheader("Live Score Trend")
                hist = pd.DataFrame(st.session_state.live_history)
                figt = go.Figure()
                for col in [c for c in hist.columns if c != "t"]:
                    figt.add_trace(go.Scatter(x=hist["t"], y=hist[col], mode="lines", name=col))
                figt.add_hline(y=threshold_limit, line_dash="dash", line_color="red", annotation_text="Alarm Threshold")
                figt.update_layout(yaxis_title="Anomaly Score (× normal-period P99.5 error)",
                                   template="plotly_dark", height=350)
                st.plotly_chart(figt, use_container_width=True)

            render_xai(subsystem_errors, sensor_errors, is_alert)

    if st.session_state.live_on:
        time.sleep(float(poll_interval))
        st.rerun()
