from flask import Flask, render_template, request, send_from_directory, url_for
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.io as pio
import os, io
from processing.ert_processor import process_ert_data

# Initialize Flask app
app = Flask(__name__)

# ✅ Cloud Run only allows writing to /tmp (use it safely)
UPLOAD_FOLDER = "/tmp/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50 MB max

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

def build_dashboard(dataframe):
    """
    Generate dashboard metrics and Plotly visuals.
    Detects missing columns and reports user-facing alerts instead of crashing.
    """
    dataframe.columns = [c.lower() for c in dataframe.columns]
    required_cols = ['yield_lps', 'cost_usd', 'depth_m']
    missing_cols = [col for col in required_cols if col not in dataframe.columns]
    alerts = []

    # --- Add alerts for missing columns ---
    if missing_cols:
        alerts.append(
            f"⚠️ Missing required fields: {', '.join(missing_cols)} — using defaults."
        )

    # Ensure all required columns exist for downstream calculations
    for col in required_cols:
        if col not in dataframe.columns:
            dataframe[col] = np.nan

    # --- Compute metrics safely ---
    total_bh = len(dataframe)
    avg_yield = round(dataframe['yield_lps'].mean(skipna=True) if 'yield_lps' in dataframe.columns else 0, 2)
    avg_cost = round(dataframe['cost_usd'].mean(skipna=True) if 'cost_usd' in dataframe.columns else 0, 2)

    proj_savings = round(total_bh * avg_cost * 0.25, 2)

    # --- Plotly visualization ---
    fig = px.scatter(
        dataframe.fillna(0),
        x='cost_usd', y='yield_lps',
        color='borehole_type' if 'borehole_type' in dataframe.columns else None,
        size='depth_m',
        hover_data=[c for c in dataframe.columns if c not in ['cost_usd', 'yield_lps', 'depth_m']],
        title='Yield vs Cost by Borehole Type'
    )
    fig.update_layout(template='plotly_white', height=480)
    graph_html = pio.to_html(fig, full_html=False)

    # --- Format alerts as floating toast notification (dark theme styling) ---
    alert_html = ""
    if alerts:
        alert_html = (
            '<div id="toast-container" '
            'style="position:fixed;top:20px;right:20px;z-index:9999;'
            'background:rgba(10,30,80,0.9);border-left:6px solid #00b4d8;'
            'padding:14px 18px;border-radius:10px;color:#f8f9fa;'
            'font-family:Arial,sans-serif;font-size:14px;'
            'box-shadow:0 4px 10px rgba(0,0,0,0.25);'
            'opacity:0;transition:opacity 0.6s ease-in-out;">'
            + "<br>".join(alerts) +
            '</div>'
            '<script>'
            'var toast=document.getElementById("toast-container");'
            'setTimeout(function(){toast.style.opacity=1;},200);'
            'setTimeout(function(){toast.style.opacity=0;},6000);'
            'setTimeout(function(){toast.remove();},7000);'
            '</script>'
        )

    return dict(
        total_bh=total_bh,
        avg_yield=avg_yield,
        avg_cost=avg_cost,
        proj_savings=proj_savings,
        plot_html=alert_html + graph_html  # combine alerts + chart
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
    df.loc[df["Drawdown_m"] < 0, "Drawdown_m"] = np.nan
    df.loc[df["Specific_Capacity_Lps_per_m"] < 0, "Specific_Capacity_Lps_per_m"] = np.nan
    
    # --- Optional derived metric: Cost per meter ---
    if "Depth_m" in df.columns and "Cost_USD" in df.columns:
        df["Cost_per_m_USD"] = df["Cost_USD"] / df["Depth_m"].replace(0, np.nan)
    
    # --- Return clean DataFrame ---
    return df

@app.route('/', methods=['GET','POST'])
def index():
    global df, ert_processed_img, ert_processed_csv, ert_uploaded_img
    message = None

    if request.method == 'POST':
        # CSV data
        if 'file' in request.files:
            f = request.files['file']
            if f and f.filename.endswith('.csv'):
                try:
                    save_path = os.path.join(app.config['UPLOAD_FOLDER'], f.filename)
                    f.save(save_path)
                    df = process_bh_ea_csv(save_path)  # ✅ use the smart parser
                    message = (message or '') + f' ✅ Loaded and processed {len(df)} records from {f.filename}'
                except Exception as e:
                    message = (message or '') + f' ⚠️ CSV error: {e}'

        # ERT raw data (.dat/.xyz/.csv grid) → process with PyGIMLi if available, else fallback renderer
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
                           **metrics)

@app.route('/uploads/<path:filename>')
def serve_uploads(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
