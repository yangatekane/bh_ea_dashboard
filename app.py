from flask import Flask, render_template, request, send_from_directory, url_for
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
import os
from processing.ert_processor import process_ert_data

# Initialize Flask app
app = Flask(__name__)

# ✅ Cloud Run only allows writing to /tmp (use it safely)
UPLOAD_FOLDER = "/tmp/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50 MB max

# --- Default thresholds (editable from dashboard) ---
COST_GOLD_MAX = 1700
YIELD_GOLD_MIN = 1.7
COST_TROUBLE_MIN = 4000
YIELD_TROUBLE_MAX = 0.8


# Demo dataset defaults
default_data = {
    "District": ["Amathole", "BCM", "Chris Hani"] * 2,
    "Borehole_Type": ["Production", "Production", "Production", "Domestic", "Domestic", "Domestic"],
    "Depth_m": [120, 110, 125, 60, 55, 65],
    "Yield_Lps": [5.2, 4.8, 6.1, 1.8, 2.1, 2.3],
    "Cost_USD": [7285, 7200, 7350, 3723, 3700, 3740]
}
df = pd.DataFrame(default_data)

ert_processed_img = None
ert_processed_csv = None
ert_uploaded_img = None


def _percentile_thresholds(x: pd.Series, low_q=0.25, high_q=0.75):
    """Return (low_threshold, high_threshold), ignoring NaNs."""
    x = pd.to_numeric(x, errors="coerce").dropna()
    if len(x) == 0:
        return (np.nan, np.nan)
    return (np.nanpercentile(x, low_q * 100), np.nanpercentile(x, high_q * 100))


def _ellipse_shape(x_center, y_center, width, height, line_color, fill_rgba):
    """Plotly circle shape used as an ellipse in data coords."""
    # Plotly draws 'circle' with sizex/sizey in data units (acts like an ellipse if axes are scaled differently)
    return dict(
        type="circle",
        xref="x", yref="y",
        x0=x_center - width / 2.0, x1=x_center + width / 2.0,
        y0=y_center - height / 2.0, y1=y_center + height / 2.0,
        line=dict(color=line_color, width=2, dash="dot"),
        fillcolor=fill_rgba,
        layer="below"
    )


def build_dashboard(dataframe):
    """
    Generate dashboard metrics and Plotly visuals.
    - Robust to missing columns (toast alert instead of 500).
    - Adds dotted elliptical overlays for:
        🟩 Goldilocks (Low Cost / High Yield)
        🟥 Trouble (High Cost / Low Yield)
    """
    # --- Normalize headers ---
    dataframe = dataframe.copy()
    dataframe.columns = [c.strip().lower() for c in dataframe.columns]

    # --- Required columns ---
    required_cols = ['yield_lps', 'cost_usd', 'depth_m']
    alerts = []

    missing = [c for c in required_cols if c not in dataframe.columns]
    if missing:
        alerts.append(f"⚠ Missing required fields: {', '.join(missing)} — using defaults.")

    for c in required_cols:
        if c not in dataframe.columns:
            dataframe[c] = np.nan

    # --- Compute headline metrics safely ---
    total_bh = len(dataframe)
    avg_yield = round(pd.to_numeric(dataframe['yield_lps'], errors="coerce").mean(skipna=True) or 0, 2)
    avg_cost  = round(pd.to_numeric(dataframe['cost_usd'], errors="coerce").mean(skipna=True) or 0, 2)
    proj_savings = round(total_bh * (avg_cost or 0) * 0.25, 2)

    # --- Clean copies for plotting ---
    plot_df = dataframe.copy()
    plot_df['yield_lps'] = pd.to_numeric(plot_df['yield_lps'], errors="coerce")
    plot_df['cost_usd']  = pd.to_numeric(plot_df['cost_usd'], errors="coerce")
    plot_df['depth_m']   = pd.to_numeric(plot_df['depth_m'], errors="coerce")

    # --- Base scatter ---
    fig = px.scatter(
        plot_df.fillna(0),
        x='cost_usd',
        y='yield_lps',
        color='borehole_type' if 'borehole_type' in plot_df.columns else None,
        size='depth_m',
        hover_data=[c for c in plot_df.columns if c not in ['cost_usd', 'yield_lps', 'depth_m']],
        title='Yield vs Cost by Borehole Type'
    )
    fig.update_layout(template='plotly_white', height=520, margin=dict(l=40, r=20, t=60, b=40))

    # --- Domain thresholds ---
    global COST_GOLD_MAX, YIELD_GOLD_MIN, COST_TROUBLE_MIN, YIELD_TROUBLE_MAX

    # Assign readable threshold aliases (for plotting guide lines etc.)
    cost_lo, cost_hi = COST_GOLD_MAX, COST_TROUBLE_MIN
    yld_lo, yld_hi   = YIELD_TROUBLE_MAX, YIELD_GOLD_MIN

    # We’ll only draw if we have at least one valid numeric pair
    if not np.isnan(cost_lo) and not np.isnan(cost_hi) and not np.isnan(yld_lo) and not np.isnan(yld_hi):
        # Region masks
        gold_mask = (plot_df['cost_usd'] < COST_GOLD_MAX) & (plot_df['yield_lps'] > YIELD_GOLD_MIN)
        trou_mask = (plot_df['cost_usd'] > COST_TROUBLE_MIN) & (plot_df['yield_lps'] < YIELD_TROUBLE_MAX)


        # Helper to compute ellipse center/size from subset
        def region_params(mask, pad=0.15):
            sub = plot_df.loc[mask, ['cost_usd', 'yield_lps']].dropna()
            if len(sub) == 0:
                return None
            xc = sub['cost_usd'].mean()
            yc = sub['yield_lps'].mean()
            # Width/height ~ spread of the subset (robust)
            w = max(sub['cost_usd'].quantile(0.90) - sub['cost_usd'].quantile(0.10), 1e-9)
            h = max(sub['yield_lps'].quantile(0.90) - sub['yield_lps'].quantile(0.10), 1e-9)
            # Pad a bit so the ellipse comfortably encloses the cluster
            return (xc, yc, w * (1 + pad), h * (1 + pad))

        gold = region_params(gold_mask)
        trou = region_params(trou_mask)

        shapes = []
        # --- Always show corner annotations for both zones (no ellipses) ---
        annotations = [
            dict(
                xref="paper", yref="paper", x=0.02, y=0.95,
                showarrow=False,
                text="🟩 Goldilocks<br><span style='font-size:11px'>(low cost, high yield)</span>",
                font=dict(color="#1e7e34", size=12),
                align="left",
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="#2ecc71", borderwidth=1, borderpad=4
            ),
            dict(
                xref="paper", yref="paper", x=0.98, y=0.05,
                xanchor="right", yanchor="bottom",
                showarrow=False,
                text="🟥 Trouble<br><span style='font-size:11px'>(high cost, low yield)</span>",
                font=dict(color="#7a1c16", size=12),
                align="right",
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="#e74c3c", borderwidth=1, borderpad=4
            )
        ]
        fig.update_layout(annotations=annotations)

        # Apply shapes + annotations
        fig.update_layout(shapes=fig.layout.shapes + tuple(shapes) if fig.layout.shapes else tuple(shapes))
        fig.update_layout(annotations=list(fig.layout.annotations) + annotations if fig.layout.annotations else annotations)

        # Also add faint guide lines at thresholds
        fig.add_hline(y=yld_hi, line_dash="dot", line_color="#2ecc71", opacity=0.4)
        fig.add_vline(x=cost_lo, line_dash="dot", line_color="#2ecc71", opacity=0.4)
        fig.add_hline(y=yld_lo, line_dash="dot", line_color="#e74c3c", opacity=0.4)
        fig.add_vline(x=cost_hi, line_dash="dot", line_color="#e74c3c", opacity=0.4)
    else:
        alerts.append("ℹ Not enough numeric data to compute Goldilocks/Trouble regions yet.")

    # --- Toast container (persisting, non-layout shifting) ---
    alert_html = ""
    if alerts:
        alert_html = (
            '<div id="toast" style="position:fixed;top:16px;right:16px;z-index:9999;'
            'background:rgba(10,30,80,0.92);border-left:6px solid #00b4d8;'
            'padding:12px 16px;border-radius:10px;color:#f8f9fa;'
            'font-family:Arial,sans-serif;box-shadow:0 2px 8px rgba(0,0,0,0.25);max-width:420px">'
            + "<br>".join(alerts) +
            '</div>'
            '<script>setTimeout(()=>{const t=document.getElementById("toast");if(t){t.style.opacity=0;t.style.transition="opacity .6s";setTimeout(()=>t.remove(),700)}},8000);</script>'
        )

    graph_html = pio.to_html(fig, full_html=False)
    return dict(
        total_bh=total_bh,
        avg_yield=avg_yield,
        avg_cost=avg_cost,
        proj_savings=proj_savings,
        plot_html=alert_html + graph_html
    )


def process_bh_ea_csv(file_path):
    """
    Cleans and augments Borehole Exploration/Surveying CSV data for BH-EA dashboard.
    Handles semicolon delimiters, spacing, and missing derived fields.
    """
    # --- Load & Detect Separator ---
    try:
        df = pd.read_csv(file_path, delimiter=';')
    except Exception:
        df = pd.read_csv(file_path)  # fallback to comma

    # --- Clean Headers ---
    df.columns = [c.strip().replace(" ", "_") for c in df.columns]

    # --- Convert numeric fields safely ---
    numeric_cols = [
        "Depth_m", "Static_WL_m_bgl", "Dynamic_WL_m_bgl",
        "Yield_Lps", "Drawdown_m", "Specific_Capacity_Lps_per_m",
        "Cost_USD"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = (
                df[col].astype(str)
                .str.replace(",", ".", regex=False)
                .str.extract(r"([\d\.]+)", expand=False)
                .astype(float)
            )

    # --- Compute missing fields ---
    # Drawdown
    if "Drawdown_m" not in df.columns or df["Drawdown_m"].isnull().any():
        if "Static_WL_m_bgl" in df.columns and "Dynamic_WL_m_bgl" in df.columns:
            df["Drawdown_m"] = df["Dynamic_WL_m_bgl"] - df["Static_WL_m_bgl"]

    # Specific Capacity
    if "Specific_Capacity_Lps_per_m" not in df.columns or df["Specific_Capacity_Lps_per_m"].isnull().any():
        if "Yield_Lps" in df.columns and "Drawdown_m" in df.columns:
            df["Specific_Capacity_Lps_per_m"] = df["Yield_Lps"] / df["Drawdown_m"].replace(0, np.nan)

    # --- Clean outliers or negatives ---
    if "Drawdown_m" in df.columns:
        df.loc[df["Drawdown_m"] < 0, "Drawdown_m"] = np.nan
    if "Specific_Capacity_Lps_per_m" in df.columns:
        df.loc[df["Specific_Capacity_Lps_per_m"] < 0, "Specific_Capacity_Lps_per_m"] = np.nan

    # --- Optional derived metric: Cost per meter ---
    if "Depth_m" in df.columns and "Cost_USD" in df.columns:
        df["Cost_per_m_USD"] = df["Cost_USD"] / df["Depth_m"].replace(0, np.nan)

    return df


@app.route('/', methods=['GET', 'POST'])
def index():
    global df, ert_processed_img, ert_processed_csv, ert_uploaded_img
    message = None
    global COST_GOLD_MAX, YIELD_GOLD_MIN, COST_TROUBLE_MIN, YIELD_TROUBLE_MAX

    # --- Threshold update from form ---
    if 'YIELD_GOLD_MIN' in request.form:
        try:
            YIELD_GOLD_MIN = float(request.form.get('YIELD_GOLD_MIN', YIELD_GOLD_MIN))
            COST_GOLD_MAX = float(request.form.get('COST_GOLD_MAX', COST_GOLD_MAX))
            YIELD_TROUBLE_MAX = float(request.form.get('YIELD_TROUBLE_MAX', YIELD_TROUBLE_MAX))
            COST_TROUBLE_MIN = float(request.form.get('COST_TROUBLE_MIN', COST_TROUBLE_MIN))
        except ValueError:
            pass


    if request.method == 'POST':
        # CSV data
        if 'file' in request.files:
            f = request.files['file']
            if f and f.filename.endswith('.csv'):
                try:
                    save_path = os.path.join(app.config['UPLOAD_FOLDER'], f.filename)
                    f.save(save_path)
                    df = process_bh_ea_csv(save_path)  # ✅ smart parser
                    message = (message or '') + f' ✅ Loaded and processed {len(df)} records from {f.filename}'
                except Exception as e:
                    message = (message or '') + f' ⚠️ CSV error: {e}'

        # ERT raw data (.dat/.xyz/.csv grid)
        if 'ert_data' in request.files:
            dat = request.files['ert_data']
            if dat and dat.filename:
                save_path = os.path.join(app.config['UPLOAD_FOLDER'], dat.filename)
                dat.save(save_path)
                img_path, model_path = process_ert_data(save_path, output_dir=app.config['UPLOAD_FOLDER'])
                if img_path:
                    ert_processed_img = img_path
                    ert_processed_csv = model_path
                    message = (message or '') + f' ✅ ERT data processed: {os.path.basename(dat.filename)}'
                else:
                    message = (message or '') + ' ⚠️ ERT processing failed.'

        # ERT display image (simple thumbnail)
        if 'ert_image' in request.files:
            img = request.files['ert_image']
            if img and img.filename:
                img_path = os.path.join(app.config['UPLOAD_FOLDER'], img.filename)
                img.save(img_path)
                ert_uploaded_img = img_path
                message = (message or '') + f' ✅ ERT-I image uploaded: {img.filename}'

    metrics = build_dashboard(df)

    def make_url(p):
        return url_for('serve_uploads', filename=os.path.basename(p)) if p else None

    return render_template('index.html',
                           message=message,
                           ert_processed_img_url=make_url(ert_processed_img),
                           ert_processed_csv_url=make_url(ert_processed_csv),
                           ert_uploaded_img_url=make_url(ert_uploaded_img),
                           yield_gold_min=YIELD_GOLD_MIN,
                           cost_gold_max=COST_GOLD_MAX,
                           yield_trouble_max=YIELD_TROUBLE_MAX,
                           cost_trouble_min=COST_TROUBLE_MIN,
                           **metrics)


@app.route('/uploads/<path:filename>')
def serve_uploads(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
