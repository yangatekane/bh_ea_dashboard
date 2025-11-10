from flask import Flask, render_template, request, send_from_directory, url_for
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
import os
from processing.ert_processor import process_ert_data
from google.cloud import storage
from datetime import datetime, timezone
from processing.ai_studio_client import analyze_with_ai_studio
from processing.contour_report import generate_contour_report
import json


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

# Initialize GCS client
GCS_BUCKET = "bh-ea-dashboard-uploads"
storage_client = storage.Client()


# Demo dataset defaults
default_data = {
    "District": ["Amathole", "BCM", "Chris Hani"] * 2,
    "Borehole_Type": ["Production", "Production", "Production", "Domestic", "Domestic", "Domestic"],
    "Depth_m": [120, 110, 125, 60, 55, 65],
    "Yield_Lps": [5.2, 4.8, 6.1, 1.8, 2.1, 2.3],
    "Cost_USD": [7285, 7200, 7350, 3723, 3700, 3740],
    "Pumping_Hours": [8, 6, 10, 4, 3, 5],
    "Recovery_Hours": [6, 5, 8, 3, 2, 4],
    "Transmissivity_m2_per_day": [120, 135, 140, 45, 50, 55],
    "Storage_Coefficient": [0.002, 0.0015, 0.0025, 0.0008, 0.0010, 0.0012]
}
df = pd.DataFrame(default_data)

ert_processed_img = None
ert_processed_csv = None
ert_uploaded_img = None

def upload_to_gcs(local_path, remote_name=None):
    """Upload file to Cloud Storage and return the public URL."""
    try:
        bucket = storage_client.bucket(GCS_BUCKET)
        if not remote_name:
            filename = os.path.basename(local_path)
            timestamp = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%S")
            remote_name = f"latest/{timestamp}_{filename}"
        blob = bucket.blob(remote_name)
        blob.upload_from_filename(local_path)
        blob.make_public()
        return blob.public_url
    except Exception as e:
        print("⚠️ GCS Upload failed:", e)
        return None

def upload_json_to_gcs(data: dict, remote_name: str):
    local = os.path.join(UPLOAD_FOLDER, os.path.basename(remote_name))
    with open(local, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return upload_to_gcs(local, remote_name)


def build_dashboard(dataframe):
    """
    Generate dashboard metrics and Plotly visuals.
    Adds hydrogeological cycle fields and stacked Monthly Volume vs Cost plot.
    """
    dataframe = dataframe.copy()
    dataframe.columns = [c.strip().lower() for c in dataframe.columns]

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
    avg_cost = round(pd.to_numeric(dataframe['cost_usd'], errors="coerce").mean(skipna=True) or 0, 2)
    proj_savings = round(total_bh * (avg_cost or 0) * 0.25, 2)

    # --- New averages ---
    avg_transmissivity = (
        round(pd.to_numeric(dataframe.get('transmissivity_m2_per_day'), errors="coerce").mean(skipna=True) or 0, 2)
        if 'transmissivity_m2_per_day' in dataframe.columns else 0
    )
    avg_storage = (
        round(pd.to_numeric(dataframe.get('storage_coefficient'), errors="coerce").mean(skipna=True) or 0, 5)
        if 'storage_coefficient' in dataframe.columns else 0
    )
    avg_volume = (
        round(pd.to_numeric(dataframe.get('monthly_volume_m3'), errors="coerce").mean(skipna=True) or 0, 1)
        if 'monthly_volume_m3' in dataframe.columns else 0
    )

    # --- Base scatter: Yield vs Cost ---
    plot_df = dataframe.copy()
    plot_df['yield_lps'] = pd.to_numeric(plot_df['yield_lps'], errors="coerce")
    plot_df['cost_usd'] = pd.to_numeric(plot_df['cost_usd'], errors="coerce")
    plot_df['depth_m'] = pd.to_numeric(plot_df['depth_m'], errors="coerce")

    main_fig = px.scatter(
        plot_df.fillna(0),
        x='cost_usd', y='yield_lps',
        color='borehole_type' if 'borehole_type' in plot_df.columns else None,
        size='depth_m',
        hover_data=[c for c in plot_df.columns if c not in ['cost_usd', 'yield_lps', 'depth_m']],
        title='Yield vs Cost by Borehole Type'
    )
    main_fig.update_layout(template='plotly_white', height=480)

    # --- Goldilocks / Trouble thresholds (manual logic) ---
    gold_mask = (plot_df['yield_lps'] > YIELD_GOLD_MIN) & (plot_df['cost_usd'] < COST_GOLD_MAX)
    trou_mask = (plot_df['yield_lps'] < YIELD_TROUBLE_MAX) & (plot_df['cost_usd'] > COST_TROUBLE_MIN)


    # --- Add labeled regions (no ellipse, only text annotations) ---
    annotations = []
    if gold_mask.any():
        annotations.append(dict(
            xref="paper", yref="paper",
            x=0.02, y=0.95, showarrow=False,
            text="🟩 Goldilocks<br><span style='font-size:11px'>(low cost, high yield)</span>",
            font=dict(color="#1e7e34", size=12),
            align="left",
            bgcolor="rgba(255,255,255,0.75)",
            bordercolor="#2ecc71", borderwidth=1, borderpad=4
        ))
    if trou_mask.any():
        annotations.append(dict(
            xref="paper", yref="paper",
            x=0.98, y=0.05, xanchor="right", yanchor="bottom", showarrow=False,
            text="🟥 Trouble<br><span style='font-size:11px'>(high cost, low yield)</span>",
            font=dict(color="#7a1c16", size=12),
            align="right",
            bgcolor="rgba(255,255,255,0.75)",
            bordercolor="#e74c3c", borderwidth=1, borderpad=4
        ))
    main_fig.update_layout(annotations=annotations)

    # --- Add dotted reference lines for zones ---
# --- Add dynamic dotted reference lines for user thresholds ---
    main_fig.add_hline(y=YIELD_GOLD_MIN, line_dash="dot", line_color="#2ecc71", opacity=0.5)
    main_fig.add_vline(x=COST_GOLD_MAX, line_dash="dot", line_color="#2ecc71", opacity=0.5)
    main_fig.add_hline(y=YIELD_TROUBLE_MAX, line_dash="dot", line_color="#e74c3c", opacity=0.5)
    main_fig.add_vline(x=COST_TROUBLE_MIN, line_dash="dot", line_color="#e74c3c", opacity=0.5)



    main_plot_html = pio.to_html(main_fig, full_html=False)

    # --- Second scatter: Monthly Volume vs Cost ---
    if {'monthly_volume_m3', 'cost_usd'} <= set(plot_df.columns):
        cycle_fig = px.scatter(
            plot_df,
            x='cost_usd', y='monthly_volume_m3',
            color='transmissivity_m2_per_day' if 'transmissivity_m2_per_day' in plot_df.columns else None,
            size='depth_m',
            hover_data=['yield_lps', 'transmissivity_m2_per_day', 'storage_coefficient'] if 'transmissivity_m2_per_day' in plot_df.columns else None,
            title='Monthly Volume vs Cost (Colored by Transmissivity)',
            color_continuous_scale='Viridis'
        )
        cycle_fig.update_layout(template='plotly_white', height=480)
        cycle_plot_html = pio.to_html(cycle_fig, full_html=False)
    else:
        cycle_plot_html = "<p style='color:gray;font-style:italic;'>No Monthly Volume data available.</p>"

        # --- Third scatter: Cycle Duration vs Yield (Efficiency Map) ---
    if {'cycle_duration_hr', 'yield_lps'} <= set(plot_df.columns):
        efficiency_fig = px.scatter(
            plot_df,
            x='cycle_duration_hr', y='yield_lps',
            color='efficiency_index' if 'efficiency_index' in plot_df.columns else None,
            size='storage_coefficient' if 'storage_coefficient' in plot_df.columns else None,
            hover_data=['cost_usd', 'transmissivity_m2_per_day', 'storage_coefficient'],
            title='Cycle Duration vs Yield (Efficiency Map)',
            color_continuous_scale='Turbo'
        )
        efficiency_fig.update_layout(template='plotly_white', height=480)
        efficiency_plot_html = pio.to_html(efficiency_fig, full_html=False)
    else:
        efficiency_plot_html = "<p style='color:gray;font-style:italic;'>No Pumping Cycle data available.</p>"


    # --- Toast alert (persisting, top-right) ---
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

    # --- Return all dashboard elements ---
    return dict(
        total_bh=total_bh,
        avg_yield=avg_yield,
        avg_cost=avg_cost,
        avg_transmissivity=avg_transmissivity,
        avg_storage=avg_storage,
        avg_volume=avg_volume,
        proj_savings=proj_savings,
        plot_html=alert_html + main_plot_html,
        cycle_plot_html=cycle_plot_html,
        efficiency_plot_html=efficiency_plot_html 
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
    # --- Ensure new cycle & hydraulic columns are numeric before computations ---
    for col in ["Pumping_Hours", "Recovery_Hours", "Transmissivity_m2_per_day",
                "Storage_Coefficient", "Monthly_Volume_m3"]:
        if col in df.columns:
            df[col] = (
                pd.to_numeric(
                    df[col]
                    .astype(str)
                    .str.replace(",", ".", regex=False)
                    .str.extract(r"([\d\.]+)", expand=False),
                    errors="coerce"
                )
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

        df["Cycle_Duration_hr"] = df["Pumping_Hours"] + df["Recovery_Hours"]
        df["Monthly_Volume_m3"] = df["Yield_Lps"] * 3600 * 24 * 30 / 1000  # m³/month
        df["Efficiency_Index"] = (df["Transmissivity_m2_per_day"] / df["Cost_USD"]) * 1000
        df["Storage_Index"] = df["Storage_Coefficient"] * df["Yield_Lps"]

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
                    gcs_url = upload_to_gcs(save_path)
                    if gcs_url:
                        message += f' ☁️ Uploaded to Cloud Storage: <a href="{gcs_url}" target="_blank">Open</a>'
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
                gcs_url = upload_to_gcs(save_path)
                if gcs_url:
                    message += f' ☁️ Uploaded to Cloud Storage: <a href="{gcs_url}" target="_blank">Open</a>'
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
                gcs_url = upload_to_gcs(img_path)
                if gcs_url:
                    message += f' ☁️ Uploaded to Cloud Storage: <a href="{gcs_url}" target="_blank">Open</a>'
        
        # === AI Studio auto-run (only if we have a DF and at least an image to analyze) ===
        try:
            # Prefer processed ERT image; else fall back to user-uploaded ERT-I image if available
            base_img_path = ert_processed_img or ert_uploaded_img
            report_url = None

            if base_img_path:
                contour_out = os.path.join(app.config['UPLOAD_FOLDER'], "ai_contour_report.png")
                generate_contour_report(base_img_path, contour_out)
                report_url = upload_to_gcs(contour_out, remote_name=f"reports/ai_contour_report_{int(datetime.now(timezone.utc).timestamp())}.png")

            # Build compact dataset summary + thresholds (as strings for prompt context)
            stats = df.describe(include='all').to_dict()
            thresholds = {
                "YIELD_GOLD_MIN": YIELD_GOLD_MIN,
                "COST_GOLD_MAX": COST_GOLD_MAX,
                "YIELD_TROUBLE_MAX": YIELD_TROUBLE_MAX,
                "COST_TROUBLE_MIN": COST_TROUBLE_MIN
            }
            meta = {
                "project": "practical-day-179721",
                "service": "bh-ea-dashboard",
                "utc": datetime.now(timezone.utc).isoformat(),
                "counts": {"records": len(df)},
                "thresholds": thresholds,
                "headline": {
                    "avg_yield_lps": float(pd.to_numeric(df.get('yield_lps'), errors='coerce').mean(skipna=True) or 0),
                    "avg_cost_usd": float(pd.to_numeric(df.get('cost_usd'), errors='coerce').mean(skipna=True) or 0),
                    "avg_transmissivity_m2_per_day": float(pd.to_numeric(df.get('transmissivity_m2_per_day'), errors='coerce').mean(skipna=True) or 0),
                    "avg_storage_coeff": float(pd.to_numeric(df.get('storage_coefficient'), errors='coerce').mean(skipna=True) or 0),
                },
                "columns": list(df.columns),
                "stats": stats
            }

            meta_url = upload_json_to_gcs(meta, remote_name=f"metadata/ai_meta_{int(datetime.now(timezone.utc).timestamp())}.json")
            dataset_summary_json = json.dumps({"thresholds": thresholds, "headlines": meta["headline"]}, ensure_ascii=False)

            if meta_url and report_url:
                ai_out = analyze_with_ai_studio(meta_url, report_url, dataset_summary_json)
                if isinstance(ai_out, dict):
                    # Show a concise HTML card in the message area
                    if "interpretation_summary" in ai_out or "raw_text" in ai_out:
                        summary_txt = ai_out.get("interpretation_summary") or ai_out.get("raw_text", "")
                        gold_n = len(ai_out.get("goldilocks_sites", [])) if isinstance(ai_out.get("goldilocks_sites"), list) else 0
                        trou_n = len(ai_out.get("trouble_sites", [])) if isinstance(ai_out.get("trouble_sites"), list) else 0
                        recs = ai_out.get("recommendations") or []
                        message = (message or "") + (
                            "<div style='margin-top:10px;padding:10px;border-radius:10px;background:#0b2447;color:#fff'>"
                            "<b>🤖 AI Studio Analysis</b><br>"
                            f"<i>{summary_txt}</i><br>"
                            f"🟩 Goldilocks Sites: {gold_n} &nbsp;|&nbsp; 🟥 Trouble Sites: {trou_n}<br>"
                            f"💡 Recs: {', '.join(recs) if recs else '—'}<br>"
                            f"☁️ <a href='{meta_url}' target='_blank' style='color:#9be7ff'>Metadata JSON</a> &nbsp;|&nbsp; "
                            f"🗺️ <a href='{report_url}' target='_blank' style='color:#9be7ff'>Contour Report</a>"
                            "</div>"
                        )
        except Exception as e:
            message = (message or "") + f" ⚠️ AI Studio step skipped: {e}"

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

# --- Health & Status Endpoints for Cloud Run / Monitoring ---

@app.route("/healthz")
def health_check():
    """Cloud Run health/status endpoint (UTC aware)"""
    return {"status": "ok", "service": "bh-ea-dashboard"}, 200


@app.route('/status')
def status():
    """Cloud Run health/status endpoint (UTC aware)"""
    return {
        "project": "practical-day-179721",
        "service": "bh-ea-dashboard",
        "status": "running",
        "time_utc": datetime.now(timezone.utc).isoformat()
    }, 200

@app.route('/uploads/<path:filename>')
def serve_uploads(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
