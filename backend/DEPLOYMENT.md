# Docker Deployment Guide

## Prerequisites

### On Ubuntu Cloud Server
1. **NVIDIA GPU** (L4 recommended)
2. **NVIDIA Driver** (535+ for CUDA 12.1)
   ```bash
   nvidia-smi  # Verify GPU is detected
   ```
3. **Docker** with **NVIDIA Container Toolkit**
   ```bash
   # Install Docker
   curl -fsSL https://get.docker.com | sh
   
   # Install NVIDIA Container Toolkit
   distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
   curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
   curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
     sudo tee /etc/apt/sources.list.d/nvidia-docker.list
   
   sudo apt-get update
   sudo apt-get install -y nvidia-container-toolkit
   sudo systemctl restart docker
   ```

---

## Deployment Steps

### 1. Upload Backend to Server
```bash
# On local machine
cd d:/DTLEL
scp -r backend/ user@your-server-ip:/home/user/dtlel-backend/

# Or use git
cd backend
git init
git add .
git commit -m "Production-ready backend"
git push origin main
```

### 2. Build Docker Image
```bash
# On server
cd /home/user/dtlel-backend
docker build -t dtlel-backend:latest .
```

**Build time**: ~10-15 minutes (downloads models)

### 3. Run Container

#### Option A: Docker Compose (Recommended)
```bash
docker compose up -d
```

#### Option B: Docker Run
```bash
docker run -d \
  --name dtlel-backend \
  --gpus all \
  -p 8000:8000 \
  -v $(pwd)/vector_store:/app/vector_store \
  -v $(pwd)/onnx_models:/app/onnx_models \
  --restart unless-stopped \
  dtlel-backend:latest
```

### 4. Verify Deployment
```bash
# Check container logs
docker logs dtlel-backend

# Test health endpoint
curl http://localhost:8000/health

# Test API
curl -X POST http://localhost:8000/api/v1/analyze/text \
  -H "Content-Type: application/json" \
  -d '{"text": "This is a test."}'
```

---

## Important Notes

### ONNX Models (Auto-Conversion)
The ONNX models are **NOT included in the Docker image** to keep it lightweight.

**On first startup**, the container will:
1. Detect missing ONNX models
2. Automatically run `export_onnx.py` to convert PyTorch → ONNX (~2-3 minutes)
3. Save models to the mounted `onnx_models/` volume

**Subsequent startups** will skip conversion and load existing ONNX models instantly.

**To pre-convert locally** (optional):
```bash
# On your development machine
cd backend
python export_onnx.py

# Then upload onnx_models/ to server
scp -r onnx_models/ user@server:/path/to/backend/
```

### Vector Store Persistence
The `vector_store/` directory is **mounted as a volume** to persist plagiarism data across container restarts.

### GPU Verification
Inside the container:
```bash
docker exec -it dtlel-backend nvidia-smi
```

---

## Monitoring

### View Logs
```bash
docker logs -f dtlel-backend
```

### Container Stats
```bash
docker stats dtlel-backend
```

### Health Check
The container has a built-in health check that runs every 30s.

---

## Troubleshooting

### GPU Not Detected
```bash
# Verify NVIDIA runtime
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

### Out of Memory
If models don't fit in GPU memory, reduce batch sizes in `app/engine/detector.py` and `app/engine/plagiarism.py`.

### Slow Startup
First startup takes 2-3 minutes while models load into GPU memory. This is normal.

---

## Production Recommendations

1. **Reverse Proxy**: Use Nginx or Caddy for HTTPS
2. **Firewall**: Only expose port 8000 to application servers
3. **Monitoring**: Use Prometheus + Grafana for metrics
4. **Backups**: Regularly backup `vector_store/` directory
5. **Updates**: Use image tags (not `latest`) for version control
