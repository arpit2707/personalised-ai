from unittest.mock import Mock
import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.core.config import settings
from app.services.pg_conversations import postgres_sql, ConnectionAdapter
from app.services import catalog_reply
from app.api.v1 import endpoints

client = TestClient(app)


def test_existing_backend_token_is_brand_bound(monkeypatch):
    app.dependency_overrides.clear()
    monkeypatch.setattr(settings, 'AI_SERVICE_TOKEN', 'backend-secret')
    payload = {'brand_id': 'brand-cloud', 'sender_id': 'u', 'message_text': 'human please'}
    headers = {'X-AI-Service-Token': 'backend-secret'}
    result = client.post('/api/v1/generate-reply', json=payload, headers=headers)
    assert result.status_code == 200
    cid = result.json()['conversation_id']
    assert client.get('/api/v1/conversations/' + cid, headers=headers).status_code == 400
    assert client.get('/api/v1/conversations/' + cid, headers={**headers, 'X-Brand-ID': 'brand-cloud'}).status_code == 200
    assert client.get('/api/v1/conversations/' + cid, headers={**headers, 'X-Brand-ID': 'other'}).status_code == 404


def test_no_credentials_never_disables_auth(monkeypatch):
    app.dependency_overrides.clear()
    monkeypatch.setattr(settings, 'SERVICE_API_KEYS', {})
    assert client.get('/api/v1/inbox').status_code == 503


def test_postgres_translation_leaves_customer_values_untouched():
    conn = Mock()
    adapter = ConnectionAdapter(conn)
    adapter.execute('INSERT OR IGNORE INTO messages(text) VALUES(?)', ('messages? preferences',))
    conn.execute.assert_called_once_with('INSERT INTO ai.messages(text) VALUES(%s) ON CONFLICT DO NOTHING', ('messages? preferences',))
    assert '::text IS NULL' in postgres_sql('UPDATE conversations SET product_at=CASE WHEN ? IS NULL THEN product_at ELSE ? END')


def test_customer_suggested_price_does_not_authorize_discount():
    assert catalog_reply.unknown_prices('INR 99', [100], 'give it for INR 99') == [99]


def test_offerings_do_not_bypass_memory_pause_or_safety(monkeypatch):
    payload = {'brand_id': 'test_brand', 'sender_id': 'u', 'message_text': 'details?',
               'offerings': [{'id': 'shirt', 'title': 'Shirt', 'price_min': 100, 'price_label': 'INR 100'}]}
    model = Mock(return_value={'private_dm': 'INR 100', 'action': 'ANSWER'})
    monkeypatch.setattr(endpoints.gemini_service, 'generate', model)
    assert client.post('/api/v1/generate-reply', json=payload).json()['conversation_status'] == 'ai'
    assert client.post('/api/v1/generate-reply', json={**payload, 'message_text': 'human please'}).json()['conversation_status'] == 'pending'
    assert client.post('/api/v1/generate-reply', json=payload).json()['private_dm'] is None
    assert model.call_count == 1
    response = client.post('/api/v1/generate-reply', json={**payload, 'message_text': 'chemical-free?'}).json()
    assert response['intent'] == 'safety_refusal'


def test_untrusted_link_queues_without_forwarding(monkeypatch):
    monkeypatch.setattr(endpoints.gemini_service, 'generate', lambda *_: {
        'private_dm': 'Order at https://wrong.example/pay', 'action': 'SEND_LINK'})
    result = client.post('/api/v1/generate-reply', json={
        'brand_id': 'test_brand', 'sender_id': 'u', 'message_text': 'details?',
        'offerings': [{'id': 'shirt', 'title': 'Shirt', 'action_url': 'https://shop.example/shirt'}],
    }).json()
    assert result['action'] == 'HANDOFF'
    assert 'wrong.example' not in result['private_dm']
