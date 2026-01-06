# Step 0: Start Redis
echo "Step 0: Starting Redis..."
docker-compose up -d redis

# Step 1: Run full DVC pipeline (download → transform → setup_feast → train)
echo "==========================================="
echo "Running full DVC pipeline..."
echo "==========================================="
dvc repro

dvc add model/model_info.json
git add .
git commit -m "Update pipeline outputs and retrain model"

# Step 2: Start FastAPI service in background
echo "==========================================="
echo "Starting FastAPI backend..."
echo "==========================================="
uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload &
API_PID=$!

# Wait for backend to start
sleep 8

# Step 3: Start Streamlit dashboard
echo "==========================================="
echo "Starting Streamlit dashboard..."
echo "==========================================="
echo ""
echo "🎉 FeastFlow Demo is ready!"
echo "📊 Streamlit Dashboard: http://localhost:8501"
echo "🔌 API Documentation: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop all services"
echo ""

streamlit run UI_app.py

# Cleanup
echo "Stopping FastAPI backend..."
kill $API_PID 2>/dev/null