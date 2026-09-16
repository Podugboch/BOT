import sys
from pathlib import Path

# Add project root to sys.path so modules can be imported in Vercel serverless environment
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from web_server import app

# Vercel serverless ASGI handler
app = app
