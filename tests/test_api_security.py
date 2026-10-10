import pytest
import sys
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# We can test the security helper logic and FastAPI endpoints
from app.api import app, AUDIO_TEMP_DIR
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    return TestClient(app)


def test_home_page(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "Foximize AI Voice Assistant" in response.text


def test_audio_path_traversal_prevention(client):
    # Attempt directory traversal with ..
    response = client.get("/audio/../../requirements.txt")
    # Should be rejected with 400 (invalid filename) or 404
    assert response.status_code in (400, 404)

    # Attempt path traversal with encoded or slash
    response2 = client.get("/audio/..%2F..%2Fmain.py")
    assert response2.status_code in (400, 404)


def test_audio_missing_file_returns_404(client):
    # Non-existent valid filename must return 404, not 200 with an error dict
    response = client.get("/audio/non_existent_file_99999.wav")
    assert response.status_code == 404
    assert response.json()["detail"] == "Audio file not found"


def test_chat_empty_message(client):
    response = client.post("/chat", json={"message": "   "})
    assert response.status_code == 400


def test_chat_valid_message(client):
    with patch("app.api.search_knowledge", return_value=[{"text": "Python Full Stack fee is 40000", "source": "fees.txt"}]), \
         patch("app.api.ask_llm", return_value="The course fee is forty thousand rupees."):
        response = client.post("/chat", json={"message": "What is the fee?"})
        assert response.status_code == 200
        data = response.json()
        assert data["response"] == "The course fee is forty thousand rupees."
        assert "fees.txt" in data["sources"]
