# AI Studio Prompt Summary – Borehole Exploration/Surveying Analytics (BH-EA)

## Overview
BH-EA is a serverless groundwater analytics platform deployed on **Google Cloud Run**.  
It was developed collaboratively using **Google AI Studio (Gemini)** to generate code components, refine data processing logic, and design visual dashboards for borehole exploration and yield testing.

---

## AI-Generated Components

### 🧩 1. Flask + Plotly Dashboard Layout
**Prompt Example:**
> "Generate a Flask application that visualizes borehole data (District, Depth, Yield, and Cost) using Plotly. Include routes for CSV upload, data summary, and cost-yield correlation plots."

**AI Contribution:**
- Produced the initial Flask app structure (`app.py`)
- Added `UPLOAD_FOLDER` logic for `/tmp` runtime storage
- Generated responsive Plotly visual templates

---

### ⚙️ 2. PyGIMLi ERT-I Processing Pipeline
**Prompt Example:**
> "Create a Python script using PyGIMLi to process 2-D Electrical Resistivity Tomography (ERT) data and export it as a surface grid or contour image suitable for Flask/Plotly dashboards."

**AI Contribution:**
- Generated the prototype of `processing/ert_processor.py`
- Suggested data-to-image pipeline compatible with Cloud Run stateless architecture
- Integrated image upload + thumbnail logic for `/ERT-I`

---

### 📊 3. Dynamic CSV Loader and Plotly Update
**Prompt Example:**
> "Extend the dashboard to allow dynamic CSV uploads and automatically refresh Plotly graphs for cost-yield impact visualization."

**AI Contribution:**
- Built the upload endpoint and dynamic Plotly callbacks
- Implemented summary statistics (average yield, total cost per district)
- Helped tune data schema (`District,Borehole_Type,Depth_m,Yield_Lps,Cost_USD`)

---

### 🧠 4. Gemini-Powered Dataset Insight Summarizer (planned)
**Prompt Example:**
> "Using google-generativeai, summarize borehole yield and cost trends by district in a natural-language report."

**AI Contribution:**
- Proposed integration with Gemini 1.5 Pro API
- Designed `/insight` route to return a concise, human-readable summary

---

## AI Studio Deployment Context
- Code snippets were generated and refined within **Google AI Studio** sessions between 2–6 Nov 2025.  
- Final code integrated manually into the GitHub repo:  
  [`https://github.com/yangatekane/bh_ea_dashboard`](https://github.com/yangatekane/bh_ea_dashboard)

---

## Compliance Declaration
This file is submitted as proof that **AI Studio was used to generate a portion of the application**, fulfilling the AI Studio Category requirement for the Google Cloud Run Hackathon 2025.

---

*Prepared by:*  
**Yanga Tekane (YGE Borehole Investigators (Pty) Ltd)**  
*In collaboration with Ta Thing (Gemini AI Assistant)*
