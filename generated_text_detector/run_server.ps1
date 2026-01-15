$env:PYTHONPATH = "."
$env:DETECTOR_CONFIG_PATH = "etc/configs/detector_config.json"
Write-Host "Server starting..."
Write-Host "Access the server at: https://localhost:8080"
.\venv\Scripts\uvicorn --host 0.0.0.0 --port 8080 --ssl-keyfile=./key.pem --ssl-certfile=./cert.pem generated_text_detector.fastapi_app:app
