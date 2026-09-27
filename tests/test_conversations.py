from unittest.mock import Mock
import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.api.v1 import endpoints
from app.models.schemas import GenerateReplyRequest
from app.services.conversation_store import ConversationStore, ConversationConflict
from app.services.gemini_service import LLMUnavailable
from app.services.guardrails import SAFETY_REFUSAL

client = TestClient(app)


def send(text='Hello', **extra):
    return client.post('/api/v1/generate-reply', json={
        'brand_id': 'test_brand', 'sender_id': 'customer', 'message_text': text, **extra,
    })


@pytest.mark.parametrize('text,reason', [
    ('I need a human', 'human_request'), ('refund please', 'order_support'),
    ('cancel my order', 'order_support'), ('payment failed', 'order_support'),
    ('This product is damaged', 'complaint'), ('order place karwa do', 'purchase_assistance'),
    ('bulk order please', 'purchase_assistance'), ('best price please', 'purchase_assistance'),
])
def test_handoff_and_pause(text, reason, monkeypatch):
    model = Mock(side_effect=AssertionError('Should not generate'))
    monkeypatch.setattr(endpoints.gemini_service, 'generate', model)
    first = send(text).json()
    assert first['conversation_status'] == 'pending'
    assert first['handoff_reason'] == reason
    assert 'not joined' in first['private_dm']
    second = send('Hello again').json()
    assert second['conversation_id'] == first['conversation_id']
    assert second['private_dm'] is None
    assert second['public_reply'] is None
    assert len(client.get('/api/v1/inbox').json()) == 1
    detail = client.get('/api/v1/conversations/' + first['conversation_id']).json()
    assert detail['summary']['reason'] == reason
    assert detail['priority'] == ('high' if reason == 'purchase_assistance' else 'normal')
    model.assert_not_called()


def test_agent_claim_message_release():
    cid = send('human please').json()['conversation_id']
    root = '/api/v1/conversations/' + cid
    assert client.post(root + '/claim', json={'agent_id': 'alice'}).json()['status'] == 'active'
    assert client.post(root + '/claim', json={'agent_id': 'bob'}).status_code == 409
    assert client.post(root + '/release', json={'agent_id': 'bob'}).status_code == 409
    assert send('size?').json()['private_dm'] is None
    assert client.post(root + '/messages', json={'agent_id': 'alice', 'message_text': 'How can I help?'}).status_code == 200
    assert client.post(root + '/release', json={'agent_id': 'alice'}).json()['status'] == 'ai'
    assert send('Thank you').json()['conversation_status'] == 'ai'
    assert client.get(root).json()['messages'][-3]['role'] == 'agent'


def test_two_failed_answers_and_reset():
    assert send().json()['conversation_status'] == 'ai'
    assert send('still not helpful').json()['conversation_status'] == 'ai'
    assert send('what colors?').json()['conversation_status'] == 'ai'
    assert send('still not helpful').json()['conversation_status'] == 'ai'
    result = send('still not resolved').json()
    assert result['handoff_reason'] == 'unresolved_query'


def test_normal_interest_does_not_handoff():
    for _ in range(6):
        reply = send('price and size?').json()
        assert reply['conversation_status'] == 'ai'
        assert reply['lead_interested'] is True


@pytest.mark.parametrize('reason', ['missing_information', 'conflicting_information', 'purchase_assistance'])
def test_model_judgment_handoff(reason, monkeypatch):
    monkeypatch.setattr(endpoints.gemini_service, 'generate', lambda *_: {
        'private_dm': 'Do not forward this speculative answer', 'handoff_reason': reason,
    })
    reply = send().json()
    assert reply['handoff_reason'] == reason
    assert 'speculative' not in reply['private_dm']


def test_unavailable_model_queues_instead_of_inventing(monkeypatch):
    monkeypatch.setattr(endpoints.gemini_service, 'generate', Mock(side_effect=LLMUnavailable()))
    reply = send('size?').json()
    assert reply['handoff_reason'] == 'generation_unavailable'
    assert 'stock' not in reply['private_dm']


@pytest.mark.parametrize('output', [{'private_dm': ''}, {'private_dm': ['bad']}, {'wrong': 'schema'}])
def test_bad_model_output_fails_closed(output, monkeypatch):
    monkeypatch.setattr(endpoints.gemini_service, 'generate', lambda *_: output)
    assert send().json()['handoff_reason'] == 'generation_unavailable'


def test_strict_safety_before_handoff_and_no_lead(monkeypatch):
    model = Mock(side_effect=AssertionError('Blocked input reached model'))
    monkeypatch.setattr(endpoints.gemini_service, 'generate', model)
    for text in ['chemical-free shampoo price?', 'harassment complaint refund', 'poison']:
        reply = send(text).json()
        assert reply['private_dm'] == SAFETY_REFUSAL
        assert reply['intent'] == 'safety_refusal'
        assert reply['conversation_id'] is None
    assert client.get('/api/v1/inbox').json() == []


def test_blocked_model_output_and_dm_channel(monkeypatch):
    monkeypatch.setattr(endpoints.gemini_service, 'generate', lambda *_: {'private_dm': 'chemical details'})
    reply = send(event_type='dm').json()
    assert reply['private_dm'] == SAFETY_REFUSAL
    assert reply['public_reply'] is None


def test_memory_in_prompt_and_explicit_preferences(monkeypatch):
    def first(system, prompt):
        return {'private_dm': 'Noted.', 'preferences': [
            {'key': 'size', 'value': 'XL', 'evidence': 'I wear XL'},
            {'key': 'color', 'value': 'blue', 'evidence': 'I like blue'},
        ]}
    monkeypatch.setattr(endpoints.gemini_service, 'generate', first)
    cid = send('I wear XL').json()['conversation_id']
    def second(system, prompt):
        assert 'I wear XL' in prompt
        assert '"size": "XL"' in prompt
        assert '"color": "blue"' not in prompt
        return {'private_dm': 'Welcome back.'}
    monkeypatch.setattr(endpoints.gemini_service, 'generate', second)
    assert send('Hello again').status_code == 200
    assert client.get('/api/v1/conversations/' + cid).json()['preferences'] == {'size': 'XL'}


def test_brand_auth_and_isolation():
    app.dependency_overrides.clear()
    payload = {'brand_id': 'test_brand', 'sender_id': 'same', 'message_text': 'human please'}
    assert client.post('/api/v1/generate-reply', json=payload).status_code == 401
    assert client.post('/api/v1/generate-reply', json=payload, headers={'X-Service-Key': 'key-b'}).status_code == 403
    result = client.post('/api/v1/generate-reply', json=payload, headers={'X-Service-Key': 'key-a'}).json()
    cid = result['conversation_id']
    assert client.get('/api/v1/conversations/' + cid, headers={'X-Service-Key': 'key-b'}).status_code == 404
    assert client.get('/api/v1/inbox', headers={'X-Service-Key': 'key-b'}).json() == []


def test_retention_original_timestamps_and_restart(tmp_path):
    now = [10000000.0]
    path = str(tmp_path / 'memory.db')
    store = ConversationStore(path, clock=lambda: now[0])
    req = GenerateReplyRequest(brand_id='a', sender_id='u', message_text='I wear XL')
    row = store.begin(req)
    store.finish('a', row, reply='Noted', product='old-sku', interested=True,
                 preferences=[{'key': 'size', 'value': 'XL', 'evidence': 'I wear XL'}], source_text=req.message_text)
    now[0] += 19 * 86400
    store = ConversationStore(path, clock=lambda: now[0])
    assert store.context('a', row['id'])['preferences'] == {'size': 'XL'}
    req.message_text = 'Hello again'
    store.begin(req)
    now[0] += 86400
    detail = store.detail('a', row['id'])
    assert detail['preferences'] == {}
    assert detail['product'] is None
    assert detail['lead_at'] is None
    assert [m['text'] for m in detail['messages']] == ['Hello again']
    assert 'XL' not in str(detail['summary'])
    now[0] += 20 * 86400
    store.purge()
    with pytest.raises(KeyError):
        store.detail('a', row['id'])


def test_stale_generation_cannot_override_handoff(tmp_path):
    store = ConversationStore(str(tmp_path / 'memory.db'))
    req = GenerateReplyRequest(brand_id='a', sender_id='u', message_text='hi')
    stale = store.begin(req)
    fresh = store.begin(req)
    store.finish('a', fresh, reason='human_request')
    store.agent_action('a', fresh['id'], 'alice', 'claim')
    with pytest.raises(ConversationConflict):
        store.finish('a', stale, reply='stale generated answer')
    assert store.detail('a', fresh['id'])['status'] == 'active'


def test_customer_and_channel_isolation():
    one = send('human please').json()
    two = send(sender_id='different').json()
    three = send(channel_type='whatsapp').json()
    assert len({one['conversation_id'], two['conversation_id'], three['conversation_id']}) == 3
    assert two['conversation_status'] == three['conversation_status'] == 'ai'


def test_backend_resume_reopens_unclaimed_handoff():
    first = send('I need a human').json()
    assert first['conversation_status'] == 'pending'
    # Other callers keep the queue.
    assert send('Hello again').json()['private_dm'] is None
    # The backend asks to reopen once its own pause is over.
    again = send('Hello again', resume_if_pending=True).json()
    assert again['conversation_status'] == 'ai'
    assert again['private_dm']


def test_backend_resume_leaves_claimed_chat_with_agent():
    cid = send('human please').json()['conversation_id']
    client.post('/api/v1/conversations/' + cid + '/claim', json={'agent_id': 'alice'})
    assert send('size?', resume_if_pending=True).json()['private_dm'] is None
