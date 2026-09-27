import pytest
from app.core.config import settings
from app.core.auth import authenticated_brand
from app.main import app
from app.api.v1 import endpoints
from app.services.conversation_store import ConversationStore
from app.services.vector_store import CatalogStore


@pytest.fixture(autouse=True)
def isolated_services(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, 'CHROMA_PERSIST_DIR', str(tmp_path / 'catalog'))
    monkeypatch.setattr(settings, 'CONVERSATION_DB_PATH', str(tmp_path / 'conversations.sqlite3'))
    monkeypatch.setattr(settings, 'GEMINI_API_KEY', '')
    monkeypatch.setattr(settings, 'DATABASE_URL', '')
    monkeypatch.setattr(settings, 'AI_SERVICE_TOKEN', '')
    monkeypatch.setattr(settings, 'SERVICE_API_KEYS', {'key-a': 'test_brand', 'key-b': 'other_brand'})
    monkeypatch.setattr(endpoints, 'conversation_store', ConversationStore())
    monkeypatch.setattr(endpoints, 'catalog_store', CatalogStore())
    app.dependency_overrides[authenticated_brand] = lambda: 'test_brand'
    monkeypatch.setattr(endpoints.gemini_service, 'generate', lambda *_: {
        'public_reply': 'Happy to help with this product.',
        'private_dm': 'Which size would you like?', 'intent': 'product_details',
    })
    yield
    app.dependency_overrides.clear()
