from flask import Flask, render_template, request, send_from_directory, url_for
import pandas as pd
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
    total_bh = len(dataframe)
    avg_yield = round(dataframe['Yield_Lps'].mean(), 2)
    avg_cost = round(dataframe['Cost_USD'].mean(), 2)
    proj_savings = round(total_bh * avg_cost * 0.25, 2)
    fig = px.scatter(
        dataframe, x='Cost_USD', y='Yield_Lps',
        color='Borehole_Type', size='Depth_m',
        hover_data=['District','Depth_m'],
        title='Yield vs Cost by Borehole Type'
    )
    fig.update_layout(template='plotly_white', height=480)
    graph_html = pio.to_html(fig, full_html=False)
    return dict(total_bh=total_bh, avg_yield=avg_yield, avg_cost=avg_cost,
                proj_savings=proj_savings, plot_html=graph_html)

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
                    content = f.read()
                    df = pd.read_csv(io.BytesIO(content))
                    message = (message or '') + f' ✅ Loaded {len(df)} records from {f.filename}'
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
