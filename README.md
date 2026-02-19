# 🚀 MLOps 
> **End-to-End Machine Learning Engineering: From Ingestion to Cloud Deployment.**

![Deadline](https://img.shields.io/badge/Deadline-March%2015-red?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-In%20Progress-orange?style=for-the-badge)
![Tech](https://img.shields.io/badge/Stack-FastAPI%20%7C%20Docker%20%7C%20Cloud-blue?style=for-the-badge)

---

## 🏗 System Architecture
This project implements a robust MLOps lifecycle, moving beyond simple scripts into a containerized, production-grade ecosystem.



---

## 🛠 Tech Stack
| Layer | Technology |
| :--- | :--- |
| **Orchestration** | DVC (Data Version Control), MLflow |
| **API Backend** | FastAPI (Python) |
| **Security** | OAuth2, JWT Authentication, Pydantic Validation |
| **Deployment** | Docker & Docker Compose |
| **Hosting** | Cloud (AWS/GCP/Azure) |
| **Frontend** | Streamlit / Dash (Interactive Data Viz) |

---

## 🛤 Execution Roadmap
The project follows a prioritized "Production-First" sequence. Note the strategic swap of **Expt 6 & 5** to ensure the model was optimized *before* being finalized for the registry.

### 📡 Phase 1: Data & Experimentation
* **Expt 1 - 3:** Data Ingestion, Preprocessing, and Versioning (DVC).
* **Expt 4:** Experiment Tracking and Logging with MLflow.

### 🧪 Phase 2: Optimization (The Pivot)
* **Expt 6:** ⚡ **Hyperparameter Tuning** (Optimization phase).
* **Expt 5:** Final Model Training & Registration (Post-Optimization).

### 🚢 Phase 3: Production & Delivery
* **Expt 7:** Developing the **FastAPI** backend with Global Error Handling & JWT Auth.
* **Expt 8:** **Containerization** (Multistage Docker builds).
* **Expt 9:** Cloud Deployment & CI/CD Pipeline.
* **Expt 10:** **Frontend Integration** (User input forms & real-time Plotly visuals).

---

## ✨ Key Features
* **🔒 Secured Inference:** The API isn't just open; it requires authenticated access via JWT.
* **🛡 Error Resilience:** Custom middleware handles 404s, 500s, and validation errors gracefully.
* **📦 Write Once, Run Anywhere:** The entire stack is Dockerized to eliminate "environment drift."
* **📊 Insightful UI:** A dedicated frontend that allows users to input data and see model confidence scores and feature importance.

---

## 🚀 Setup & Installation

### 1. Clone the Repo
```bash
git clone [https://github.com/your-username/mlops-practicals.git](https://github.com/your-username/mlops-practicals.git)
cd mlops-practicals
