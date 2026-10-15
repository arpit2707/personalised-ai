"""Quick successive messages, retried deliveries, failure reasons and memory upkeep."""
import logging
import sqlite3
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from app.api.v1 import endpoints
from app.main import app
from app.models.schemas import GenerateReplyRequest
from app.services import catalog_reply
from app.models.schemas import BrandPersona
from app.services.conversation_store import ConversationStore
from app.services.fact_guard import soft_answer_problems
from app.services.gemini_service import LLMUnavailable
from app.services.handoff_texts import is_handoff_line
from app.services.prompt_assembler import prompt_assembler
from app.services.pg_conversations import SCHEMA

client = TestClient(app)

OFFERING = {'id': 'o1', 'title': 'Bridal look', 'price_min': 18000, 'price_label': '₹18,000 se start'}


def send(text='Hello', **extra):
    res = client.post('/api/v1/generate-reply', json={
        'brand_id': 'test_brand', 'sender_id': 'customer', 'message_text': text, 'event_type': 'dm', **extra})
    assert res.status_code == 200, res.text
    return res.json()


def request(text='hi', **extra):
    return GenerateReplyRequest(brand_id='test_brand', sender_id='customer', message_text=text, event_type='dm', **extra)


# ---------------------------------------------------------------- two messages at once

def test_reply_to_an_older_message_is_superseded(monkeypatch):
    def model(*_):
        # The customer writes again while this reply is being generated.
        endpoints.conversation_store.begin(request('aur ye bhi batao'))
        return {'private_dm': 'Haan ji, bataiye.'}
    monkeypatch.setattr(endpoints.gemini_service, 'generate', model)
    reply = send('Hi', offerings=[OFFERING])
    assert reply['intent'] == 'superseded'
    assert reply['private_dm'] is None and reply['public_reply'] is None
    assert reply['requires_human_attention'] is False
    assert reply['conversation_status'] == 'ai'
    assert reply['canned'] is False
    detail = endpoints.conversation_store.detail('test_brand', reply['conversation_id'])
    assert [m['role'] for m in detail['messages']] == ['user', 'user']


def test_legacy_path_and_handoffs_are_superseded_too(monkeypatch):
    def model(*_):
        endpoints.conversation_store.begin(request('hello?'))
        raise LLMUnavailable('down', 'llm_timeout')
    monkeypatch.setattr(endpoints.gemini_service, 'generate', model)
    reply = send('size?')
    assert reply['intent'] == 'superseded'
    assert reply['private_dm'] is None


def test_conflict_without_a_newer_message_stays_409(monkeypatch):
    # The same message delivered twice (a wake-up retry) whose other copy finished first.
    def model(*_):
        store = endpoints.conversation_store
        twin = store.begin(request('Hi', message_id='mid.1'))
        store.finish('test_brand', twin, reply='Hi! Kaise madad karun?')
        return {'private_dm': 'Hello!'}
    monkeypatch.setattr(endpoints.gemini_service, 'generate', model)
    res = client.post('/api/v1/generate-reply', json={
        'brand_id': 'test_brand', 'sender_id': 'customer', 'message_text': 'Hi', 'event_type': 'dm',
        'message_id': 'mid.1', 'offerings': []})
    assert res.status_code == 409


def test_chat_taken_by_a_person_meanwhile_sends_nothing(monkeypatch):
    def model(*_):
        store = endpoints.conversation_store
        row = store.begin(request('human please'))
        store.finish('test_brand', row, reason='human_request')
        return {'private_dm': 'Hello!'}
    monkeypatch.setattr(endpoints.gemini_service, 'generate', model)
    reply = send('Hi', offerings=[OFFERING])
    assert reply['private_dm'] is None
    assert reply['conversation_status'] == 'pending'


# ---------------------------------------------------------------- the same message twice

def test_retried_message_is_stored_once_and_keeps_the_turn(tmp_path):
    store = ConversationStore(str(tmp_path / 'memory.db'))
    first = store.begin(request('Hi', message_id='mid.1'))
    again = store.begin(request('Hi', message_id='mid.1'))
    assert again['version'] == first['version']
    assert again['message_row'] == first['message_row']
    # The answer already in flight for it is still valid.
    store.finish('test_brand', first, reply='Hello!')
    newer = store.begin(request('Price?', message_id='mid.2'))
    assert newer['message_row'] != first['message_row']
    plain = store.begin(request('Price?'))
    assert plain['version'] == newer['version'] + 1
    texts = [m['text'] for m in store.detail('test_brand', first['id'])['messages']]
    assert texts == ['Hi', 'Hello!', 'Price?', 'Price?']


def test_retried_message_through_the_endpoint(monkeypatch):
    model = Mock(return_value={'private_dm': 'Hello!'})
    monkeypatch.setattr(endpoints.gemini_service, 'generate', model)
    one = send('Hi', message_id='mid.9', offerings=[])
    two = send('Hi', message_id='mid.9', offerings=[])
    detail = endpoints.conversation_store.detail('test_brand', one['conversation_id'])
    assert [m['text'] for m in detail['messages'] if m['role'] == 'user'] == ['Hi']
    assert two['private_dm'] == 'Hello!'


def test_old_sqlite_files_gain_the_message_id_column(tmp_path):
    path = str(tmp_path / 'old.db')
    conn = sqlite3.connect(path)
    conn.executescript('''
        CREATE TABLE conversations (id TEXT PRIMARY KEY, brand TEXT NOT NULL, channel TEXT NOT NULL,
            sender TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'ai', reason TEXT, priority TEXT NOT NULL DEFAULT 'normal',
            agent TEXT, product TEXT, product_at REAL, lead_at REAL, unresolved INTEGER NOT NULL DEFAULT 0,
            updated REAL NOT NULL, version INTEGER NOT NULL DEFAULT 0, UNIQUE(brand, channel, sender));
        CREATE TABLE messages (id INTEGER PRIMARY KEY, conversation TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
            role TEXT NOT NULL, text TEXT NOT NULL, created REAL NOT NULL);
    ''')
    conn.close()
    store = ConversationStore(path)
    row = store.begin(request('Hi', message_id='mid.1'))
    assert store.begin(request('Hi', message_id='mid.1'))['version'] == row['version']
    conn = sqlite3.connect(path)
    indexes = {r[1] for r in conn.execute("SELECT * FROM sqlite_master WHERE type='index'").fetchall()}
    conn.close()
    assert {'message_expiry', 'preference_expiry', 'conversation_expiry'} <= indexes


def test_postgres_schema_migrates_and_indexes():
    assert 'ADD COLUMN IF NOT EXISTS message_id' in SCHEMA
    for index in ('ai.messages(created)', 'ai.preferences(created)', 'ai.conversations(updated)'):
        assert index in SCHEMA


# ---------------------------------------------------------------- retention without a purge per request

def test_expired_rows_are_hidden_at_once_and_deleted_by_the_purge(tmp_path):
    now = [10000000.0]
    path = str(tmp_path / 'memory.db')
    store = ConversationStore(path, clock=lambda: now[0])
    row = store.begin(request('I wear XL'))
    store.finish('test_brand', row, reply='Noted', product='sku-1', interested=True,
                 preferences=[{'key': 'size', 'value': 'XL', 'evidence': 'I wear XL'}], source_text='I wear XL')
    now[0] += 20 * 86400 + 1
    context = store.context('test_brand', row['id'])
    assert context == {'history': [], 'preferences': {}}
    assert store.detail('test_brand', row['id'])['product'] is None
    assert store.inbox('test_brand') == []
    # Still on disk until the purge, which runs on its own schedule.
    conn = sqlite3.connect(path)
    assert conn.execute('SELECT COUNT(*) FROM messages').fetchone()[0] == 2
    conn.close()
    store.purge()
    conn = sqlite3.connect(path)
    assert conn.execute('SELECT COUNT(*) FROM messages').fetchone()[0] == 0
    conn.close()


def test_a_chat_idle_past_retention_starts_over_before_the_purge(tmp_path):
    now = [10000000.0]
    store = ConversationStore(str(tmp_path / 'memory.db'), clock=lambda: now[0])
    old = store.begin(request('human please'))
    store.finish('test_brand', old, reason='human_request')
    now[0] += 21 * 86400
    fresh = store.begin(request('Hi'))
    assert fresh['id'] != old['id']
    assert fresh['status'] == 'ai'


def test_canned_lines_are_left_out_of_the_model_history(tmp_path):
    store = ConversationStore(str(tmp_path / 'memory.db'))
    row = store.begin(request('size?'))
    store.finish('test_brand', row, reason='generation_unavailable', source_text='size?')
    row = store.begin(request('hello?'))
    store.finish('test_brand', row, reply='Hi! M aur L available hain.')
    history = store.context('test_brand', row['id'])['history']
    assert [m['text'] for m in history] == ['size?', 'hello?', 'Hi! M aur L available hain.']


# ---------------------------------------------------------------- "message mil gaya" during a long outage

DOWN = 'Aapka message mil gaya hai 🙏 Team jaldi aapko reply karegi.'


def model_failed(store, text, event_type='dm'):
    row = store.begin(request(text))
    return store.finish('test_brand', row, reason='generation_unavailable', source_text=text,
                        event_type=event_type)[1]


def test_model_down_notice_goes_at_most_once_per_30_minutes(tmp_path):
    now = [10000000.0]
    store = ConversationStore(str(tmp_path / 'memory.db'), clock=lambda: now[0])
    assert model_failed(store, 'size kya hai?') == DOWN
    now[0] += 5 * 60
    assert model_failed(store, 'hello?') is None
    # An outage that goes on (or a new one days later) is not silence for good.
    now[0] += 26 * 60
    assert model_failed(store, 'koi hai?') == DOWN
    now[0] += 10 * 60
    assert model_failed(store, 'reply karo') is None
    now[0] += 5 * 86400
    assert model_failed(store, 'naya sawaal hai') == DOWN


def test_a_seller_message_after_the_notice_ends_the_quiet(tmp_path):
    now = [10000000.0]
    store = ConversationStore(str(tmp_path / 'memory.db'), clock=lambda: now[0])
    row = store.begin(request('size kya hai?'))
    store.finish('test_brand', row, reason='generation_unavailable', source_text='size kya hai?')
    with store.db() as db:
        db.execute("INSERT INTO messages(conversation,role,text,created) VALUES(?,?,?,?)",
                   (row['id'], 'agent', 'M aur L dono hain.', now[0]))
    now[0] += 60
    assert model_failed(store, 'aur XL?') == DOWN


def test_a_comments_notice_does_not_silence_the_dm(tmp_path):
    now = [10000000.0]
    store = ConversationStore(str(tmp_path / 'memory.db'), clock=lambda: now[0])
    # Every comment response still carries the line...
    assert model_failed(store, 'price?', 'comment') == DOWN
    assert model_failed(store, 'price??', 'comment') == DOWN
    # ...but the backend retries comments instead of posting it, so the DM chat never read it.
    now[0] += 60
    assert model_failed(store, 'hi, price batao') == DOWN
    texts = [m['text'] for m in store.detail('test_brand', store.begin(request('x'))['id'])['messages']]
    assert texts.count(DOWN) == 1


# ---------------------------------------------------------------- why the model did not answer

@pytest.mark.parametrize('code', ['llm_timeout', 'provider_error:503', 'not_configured'])
def test_failure_code_reaches_the_backend(code, monkeypatch):
    monkeypatch.setattr(endpoints.gemini_service, 'generate', Mock(side_effect=LLMUnavailable('x', code)))
    for extra in ({'offerings': [OFFERING]}, {}):
        reply = send('size?', sender_id=f'{code}-{len(extra)}', **extra)
        assert reply['handoff_reason'] == 'generation_unavailable'
        assert reply['reasoning'] == code
        assert reply['canned'] is True


def test_unusable_output_is_named(monkeypatch):
    monkeypatch.setattr(endpoints.gemini_service, 'generate', lambda *_: {'private_dm': ['bad']})
    assert send('size?', offerings=[OFFERING])['reasoning'] == 'invalid_output'
    monkeypatch.setattr(endpoints.gemini_service, 'generate', lambda *_: {'private_dm': ' '})
    assert send('size?', sender_id='legacy')['reasoning'] == 'empty_output'


def test_a_bug_in_our_checks_is_logged_as_internal_error(monkeypatch, caplog):
    monkeypatch.setattr(endpoints.gemini_service, 'generate', lambda *_: {'private_dm': 'Haan ji, 15 Nov free hai.'})

    def broken(req, raw):
        return raw['missing_key']
    monkeypatch.setattr(catalog_reply, 'finalize', broken)
    with caplog.at_level(logging.ERROR, logger='app.api.v1.endpoints'):
        reply = send('15 Nov ko free ho?', offerings=[OFFERING])
    assert reply['handoff_reason'] == 'generation_unavailable'
    assert reply['reasoning'] == 'internal_error'
    assert 'reply_internal_error type=KeyError' in caplog.text
    assert 'test_reply_lifecycle.py:' in caplog.text
    assert '15 Nov' not in caplog.text and 'free' not in caplog.text


# ---------------------------------------------------------------- soft hand-off with a good answer

def test_soft_handoff_sends_the_checked_model_answer(monkeypatch):
    dm = 'Delivery ka time main team se confirm karke batati hoon. Tab tak kuch aur jaanna ho to bataiye?'
    monkeypatch.setattr(endpoints.gemini_service, 'generate', lambda *_: {
        'private_dm': dm, 'action': 'HANDOFF', 'handoff_reason': 'missing_information'})
    reply = send('delivery kab tak hogi?', offerings=[OFFERING])
    assert reply['private_dm'] == dm
    assert reply['action'] == 'ANSWER'
    assert reply['handoff_reason'] == 'missing_information'
    assert reply['requires_human_attention'] is False
    assert reply['conversation_status'] == 'ai'
    assert reply['canned'] is False
    assert reply['lead_interested'] is False


def test_soft_handoff_with_a_blocked_answer_gets_the_safe_line(monkeypatch):
    monkeypatch.setattr(endpoints.gemini_service, 'generate', lambda *_: {
        'private_dm': 'Delivery free hai, sirf ₹99 extra.', 'action': 'HANDOFF', 'handoff_reason': 'missing_information'})
    reply = send('delivery charge?', offerings=[OFFERING])
    assert '99' not in reply['private_dm']
    assert reply['requires_human_attention'] is True
    assert reply['intent'] == 'price_blocked'
    assert reply['canned'] is True


KURTI = {'id': 'o1', 'title': 'Cotton kurti', 'price_label': '₹899', 'price_min': 899}


@pytest.mark.parametrize('question,dm,why', [
    # The very point the model said nobody gave it, answered anyway.
    ('kal tak mil jayega?', 'Haan ji kal tak pahunch jayega.', 'arrival_time'),
    ('stock hai?', 'Haan ji, stock me hai, jaldi order kijiye.', 'stock'),
    ('delivery kab?', 'Order aaj karein to kal tak pahunch jayega. Baaki team confirm karegi.', 'arrival_time'),
    # Never says the team will confirm: the question is left hanging.
    ('EMI hai?', 'Ji bilkul! Aapko kis colour me chahiye?', 'no_deferral'),
])
def test_soft_answer_that_answers_the_missing_point_gets_the_safe_line(question, dm, why, monkeypatch, caplog):
    monkeypatch.setattr(endpoints.gemini_service, 'generate', lambda *_: {
        'private_dm': dm, 'action': 'HANDOFF', 'handoff_reason': 'missing_information'})
    with caplog.at_level(logging.INFO, logger='app.api.v1.endpoints'):
        reply = send(question, offerings=[KURTI])
    assert is_handoff_line(reply['private_dm']) and reply['private_dm'] != dm
    assert reply['requires_human_attention'] is True
    assert reply['handoff_reason'] == 'missing_information'
    assert reply['canned'] is True
    assert 'soft_answer_rejected why=' in caplog.text and why in caplog.text
    assert question not in caplog.text and dm not in caplog.text


def test_legacy_soft_answer_with_an_arrival_day_gets_the_safe_line(monkeypatch):
    monkeypatch.setattr(endpoints.gemini_service, 'generate', lambda *_: {
        'private_dm': 'Haan ji kal tak pahunch jayega.', 'handoff_reason': 'missing_information'})
    reply = send('kal tak mil jayega?')
    assert 'pahunch' not in reply['private_dm']
    assert reply['requires_human_attention'] is True
    assert reply['canned'] is True


def test_soft_answer_may_state_what_the_seller_gave(monkeypatch):
    sized = {**KURTI, 'variants': [{'label': 'M', 'in_stock': True}]}
    business = {'policies': {'delivery': 'Patna me order kal tak pahunch jata hai'}}
    dm = 'M size stock me hai. Patna me kal tak pahunch jayega, baaki shehron ka time team confirm karke batayegi.'
    monkeypatch.setattr(endpoints.gemini_service, 'generate', lambda *_: {
        'private_dm': dm, 'action': 'HANDOFF', 'handoff_reason': 'missing_information'})
    reply = send('M size Ranchi kab tak?', offerings=[sized], business=business)
    assert reply['private_dm'] == dm
    assert reply['requires_human_attention'] is False
    assert reply['canned'] is False


@pytest.mark.parametrize('dm,seller,problems', [
    ('Stock main team se confirm karke batati hoon.', '', []),
    ('Kal tak team aapko delivery ka time confirm kar degi.', '', []),
    ('Haan, kal tak pahunch jayega. Baaki team batayegi.', 'Kalamkari saree', ['arrival_time']),
    ('Haan, available hai! Size team confirm karegi.', '', ['stock']),
    ('Haan, available hai! Size team confirm karegi.', 'Ready stock', []),
    ('हाँ, कल तक पहुँच जाएगा। टीम बाकी बताएगी।', '', ['arrival_time']),
    ('I will check with the team and get back to you.', '', []),
    ('Checkout link se order ho jayega.', '', ['no_deferral']),
])
def test_soft_answer_problems(dm, seller, problems):
    req = request('kab tak?', offerings=[{'id': 'o1', 'title': 'Saree', 'description': seller}])
    assert soft_answer_problems(req, dm) == problems


def test_both_prompts_say_a_soft_answer_defers():
    catalog = catalog_reply.build_system_prompt(BrandPersona(brand_name='G'), request('hi', offerings=[]))
    assert 'private_dm is sent as written' in catalog and 'never answer it yourself' in catalog
    legacy = prompt_assembler.build_system_prompt(BrandPersona(brand_name='G'))
    assert 'private_dm is sent as written' in legacy


def test_model_echoing_a_handoff_line_is_not_its_own_answer(monkeypatch):
    monkeypatch.setattr(endpoints.gemini_service, 'generate', lambda *_: {
        'private_dm': 'Accha sawaal! Team ise confirm karke aapko yahin batayegi.', 'handoff_reason': 'missing_information'})
    reply = send('EMI milega?', offerings=[OFFERING])
    assert reply['requires_human_attention'] is True
    assert reply['canned'] is True


# ---------------------------------------------------------------- "still not helpful"

BUSINESS_TURN = [{'from': 'customer', 'text': 'bridal ka rate?'}, {'from': 'business', 'text': 'Bridal ₹18,000 se start.'}]


def test_unresolved_escalation_reads_the_backend_thread(monkeypatch):
    monkeypatch.setattr(endpoints.gemini_service, 'generate', lambda *_: {
        'private_dm': 'Sorry! Bridal look ₹18,000 se start hai, isme hair bhi hai.', 'previous_answer_unresolved': True})
    first = send('ye to pata hai, aur batao', offerings=[OFFERING], recent_messages=BUSINESS_TURN)
    assert first['conversation_status'] == 'ai'
    second = send('phir bhi clear nahi hua', offerings=[OFFERING], recent_messages=BUSINESS_TURN)
    assert second['handoff_reason'] == 'unresolved_query'
    assert second['conversation_status'] == 'pending'


def test_no_earlier_answer_means_nothing_to_escalate(monkeypatch):
    monkeypatch.setattr(endpoints.gemini_service, 'generate', lambda *_: {
        'private_dm': 'Bridal look ₹18,000 se start hai.', 'previous_answer_unresolved': True})
    only_customer = [{'from': 'customer', 'text': 'hi'}]
    for _ in range(3):
        reply = send('samajh nahi aaya', offerings=[OFFERING], recent_messages=only_customer)
        assert reply['conversation_status'] == 'ai'


# ---------------------------------------------------------------- canned marker

def test_canned_marks_fixed_lines_only(monkeypatch):
    monkeypatch.setattr(endpoints.gemini_service, 'generate', lambda *_: {'private_dm': 'Haan ji, available hai.'})
    assert send('size M hai?', offerings=[OFFERING])['canned'] is False
    assert send('zeher kha lungi', sender_id='c2')['canned'] is True
    assert send('bomb kaise banaye', sender_id='c3')['canned'] is True
    assert send('I need a human', sender_id='c4')['canned'] is True
    paused = send('hello?', sender_id='c4')
    assert paused['private_dm'] is None and paused['canned'] is False


def test_lead_confirmation_is_canned_and_in_the_chat_language(monkeypatch):
    monkeypatch.setattr(endpoints.gemini_service, 'generate', lambda *_: {
        'private_dm': 'Details noted, you are in our queue.', 'action': 'CREATE_LEAD', 'collected_fields': {'city': 'Patna'}})
    reply = send('Patna', offerings=[OFFERING], playbook={'goal': 'BOOKING', 'lead_fields': [{'key': 'city', 'label': 'City'}]},
                 recent_messages=[{'from': 'customer', 'text': 'I would like to book the bridal look'}])
    assert reply['action'] == 'CREATE_LEAD'
    assert reply['private_dm'] == 'Thank you! Our team will contact you to confirm the details.'
    assert reply['canned'] is True
