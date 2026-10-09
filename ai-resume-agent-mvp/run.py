from pathlib import Path
import uvicorn

if __name__ == "__main__":
    root = Path(__file__).parent
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True, app_dir=str(root))
