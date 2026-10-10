import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

try:
    from app.rag import build_database
except ImportError:
    from rag import build_database


if __name__ == "__main__":
    print("Refreshing Foximize knowledge base...")
    build_database()
    print("Knowledge base refresh completed.")