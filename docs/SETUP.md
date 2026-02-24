# Sanash — Development Setup Guide

## Prerequisites

| Tool | Version | Purpose |
|------|---------|---------|
| Python | 3.11+ | Backend, ML, analysis |
| Node.js | 18+ | Frontend, mobile |
| PostgreSQL | 14+ | Primary database |
| Redis | 7+ | Cache & session store |
| Git | any | Version control |

---

## 1. Clone & Configure

```bash
git clone <repository-url>
cd sana
cp .env.example .env
# Edit .env with your PostgreSQL/Redis credentials
```

---

## 2. Backend API (FastAPI)

```bash
cd api
python -m venv .venv

# Activate:
source .venv/bin/activate        # Linux/macOS
.venv\Scripts\activate           # Windows

pip install -r requirements.txt

# Start PostgreSQL and Redis first, then:
uvicorn main:app --reload --port 8000
```

Verify: `curl http://localhost:8000/api/v1/health`

---

## 3. Celery Worker (background tasks)

```bash
# In a separate terminal (same venv):
cd api
celery -A workers.celery_app worker --loglevel=info
```

---

## 4. Frontend (React Dashboard)

```bash
cd frontend
npm install
npm run dev
# Opens at http://localhost:5173
```

---

## 5. Mobile App (React Native / Expo)

```bash
cd mobile
npm install
npx expo start
# Scan QR code with Expo Go app, or press 'w' for browser
```

---

## 6. Computer Vision Pipeline

```bash
# From project root, with venv active:
pip install ultralytics simpy statsmodels scipy matplotlib seaborn pandas

# Run person detection on a video:
python bus_tracker/detector.py --input input/video.mp4

# Run crowd density estimation visualization:
python scripts/visualize_predictions.py
```

---

## 7. Research Pipeline

```bash
# Survey analysis (uses synthetic data if no CSV):
python survey/analysis/survey_analysis.py

# Run simulation (quick 10-rep test):
python simulation/bus_simulation.py --replications 10

# Generate scatter plot:
python scripts/generate_scatter_plot.py
```

---

## 8. Run Tests

```bash
pip install pytest scipy numpy matplotlib pandas statsmodels
pytest tests/ -v --tb=short
```

---

## Environment Variables (`.env`)

```ini
# Database
DATABASE_URL=postgresql://user:password@localhost:5432/sanash

# Redis
REDIS_URL=redis://localhost:6379/0

# Security
SECRET_KEY=your-secret-key-here
JWT_ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# API
API_V1_PREFIX=/api/v1
BACKEND_CORS_ORIGINS=["http://localhost:5173","http://localhost:3000"]

# ML / Edge
CSRNET_MODEL_PATH=models/csrnet/
YOLO_MODEL_PATH=models/yolov8/yolov8n.pt
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `ImportError: ultralytics` | `pip install ultralytics` |
| `ImportError: simpy` | `pip install simpy` |
| DB connection refused | Ensure PostgreSQL is running on port 5432 |
| Redis connection refused | Ensure Redis is running: `redis-server` |
| Port 8000 already in use | `uvicorn main:app --port 8001` |
| Expo QR code not scanning | Use `npx expo start --tunnel` |
