import sys
import os

# Add backend to path
sys.path.append(os.getcwd())

print("Attempting to import app.routers.analyze...")
try:
    from app.routers import analyze
    print("✅ Import successful!")
except Exception as e:
    print(f"❌ Import failed: {e}")
    import traceback
    traceback.print_exc()
