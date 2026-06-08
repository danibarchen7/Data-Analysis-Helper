# 🚀 Gemini AI Data Science Sandbox & Analytics Agent

An intelligent, multi-agent automated data analytics dashboard built with **FastAPI**, **Tailwind CSS**, and the new **Google GenAI SDK**. This application allows users to upload raw datasets (`.csv`, `.xlsx`), immediately view structural and statistical data profiles, and leverage Gemini's code execution sandbox to dynamically generate analytical summaries, actionable recommendations, and data visualizations.

---

## 🌟 Key Features

* **Dual-Engine Dataset Profiling:** Instantly captures and formats structural layouts using Pandas `.info()` along with comprehensive descriptive metrics via `.describe()`.
* **3-Step Multi-Agent Architecture:**
    1.  **Planning:** Formulates hypothesis boundaries and step-by-step processing paths based on structural data shapes.
    2.  **Code Execution Sandbox:** Safely spawns Python compilation routines to run calculations and export visual graphs.
    3.  **Synthesis:** Translates console terminal logs and outputs into clean, structured executive summaries.
* **Automated Plot Extraction:** Extracts inline `Blob` image data segments directly out of Gemini's execution framework to display **Boxplots** and **Scatter Diagrams** dynamically on the web UI.
* **Actionable Cleansing Matrix:** Provides one-click interface buttons mapping directly to strategic data-prep options (*Strict Row Pruning*, *Imputation & Winsorization*, or *Anomalous Feature Flagging*).

---

## 🏗️ Core Architecture Workflow

```text
📁 Upload Dataset (.csv / .xlsx) 
       │
       ├──► 📋 Native Pandas Profiles (df.info() & df.describe())
       │
       └──► 🤖 Gemini 2.5 Flash Agent Pipeline
                │
                ├─── Step 1: Generate Data Analysis Plan & Hypotheses
                ├─── Step 2: Run Secure Code Execution Sandbox ──► 📉 Generate Boxplot & Scatter
                └─── Step 3: Synthesize Terminal Logs into Structured JSON Data
