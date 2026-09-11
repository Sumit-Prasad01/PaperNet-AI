"""PaperNet-AI Application Entrypoint.

Allows running the FastAPI application via:
    python app.py
or
    uvicorn app:app --reload
"""

import uvicorn
from app.app import app

if __name__ == "__main__":
    uvicorn.run("app.app:app", host="0.0.0.0", port=8000, reload=True)
