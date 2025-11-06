# 💧 BH-EA | Borehole Exploration & Surveying Analytics  
**YGE BHI (Pty) Ltd | Reg. No. 2017/452107/07**

[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/flask-3.0-green.svg)](https://flask.palletsprojects.com/)
[![Plotly](https://img.shields.io/badge/plotly-analytics-orange.svg)](https://plotly.com/)
[![Google Cloud Run](https://img.shields.io/badge/Google%20Cloud-Run-blue.svg)](https://cloud.google.com/run)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 🌍 Overview
**BH-EA (Borehole Exploration / Surveying Analytics)** is a **cloud-based hydrogeological analytics platform** that integrates geological, drilling, and resistivity survey data.  
Deployed on **Google Cloud Run**, BH-EA helps engineers and municipalities reduce drilling failures, save costs, and improve groundwater access across **South Africa’s Eastern Cape Province**.

---

## 💡 Features
- 📊 **Real-time analytics:** yield, depth, and cost KPIs updated live.  
- 🧠 **Resistivity imaging:** Electrical Resistivity Tomography (ERT-I) processing using **PyGIMLi** or fallback heat-map renderer.  
- ⚙️ **Dynamic dashboard:** upload CSVs and images, refresh instantly with Plotly.  
- ☁️ **Serverless deployment:** runs fully on Google Cloud Run for scalability.  
- 💧 **Impact metrics:** cost-saving estimates, dry-hole reduction, and yield-efficiency tracking.  

---

## 🏗️ Architecture
## 🧠 Google AI Studio Integration
This project was co-developed using **Google AI Studio**, where part of the application logic and interface design were generated collaboratively with Gemini.  
See [`ai_studio_prompts.md`](./ai_studio_prompts.md) for a record of the generated prompts and iterations.

---

## 🏗️ Architecture Overview
![Architecture Diagram](./docs/architecture_diagram.png)

**Flow:**
Field Data / ERT CSV → (Optional) Cloud Storage → Cloud Run (Flask + Plotly + PyGIMLi) → Gemini AI (Insights Summarization) → Web Dashboard
