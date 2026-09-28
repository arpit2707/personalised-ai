"""Crisis, friendly hand-offs, lead confirmation and history use."""
from unittest.mock import Mock
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.api.v1 import endpoints
from app.services import crisis
from app.services.gemini_service import LLMUnavailable

client = TestClient(app)


def send(text='Hello', **extra):
    return client.post('/api/v1/generate-reply', json={
        'brand_id': 'test_brand', 'sender_id': 'customer', 'message_text': text, **extra,
    }).json()


@pytest.mark.parametrize('text', [
    'I want to kill myself', 'thinking about suicide', "i don't want to live anymore",
    'mujhe marna hai', 'jeena nahi hai ab', 'zeher kha lungi', 'khud ko khatam kar dunga',
    'मुझे मरना है', 'आत्महत्या',
])
def test_crisis_detection(text):
    assert crisis.is_crisis(text)


@pytest.mark.parametrize('text', [
    'killer look!', 'price kya hai', 'dying to buy this', 'mar gaye itna sundar', 'this dress is to die for',
])
def test_crisis_ignores_everyday_phrases(text):
    assert not crisis.is_crisis(text)


def test_crisis_reply_in_the_customer_language():
    assert 'Tele-MANAS: 14416' in crisis.crisis_reply('I want to die')
    assert 'kariye' in crisis.crisis_reply('mujhe marna hai')
    assert 'टेली-मानस' in crisis.crisis_reply('मुझे मरना है')


def test_crisis_runs_before_safety_filter_and_pauses_once(monkeypatch):
    model = Mock(side_effect=AssertionError('Should not generate'))
    monkeypatch.setattr(endpoints.gemini_service, 'generate', model)
    first = send('zeher kha lungi', event_type='comment')
    assert first['intent'] == 'crisis'
    assert '14416' in first['private_dm']
    assert first['public_reply'] == crisis.CRISIS_PUBLIC
    assert first['conversation_status'] == 'pending'
    assert first['handoff_reason'] == 'crisis'
    again = send('zeher kha lungi')
    assert again['private_dm'] is None
    model.assert_not_called()


def test_soft_line_is_not_sent_twice_in_a_row(monkeypatch):
    monkeypatch.setattr(endpoints.gemini_service, 'generate', lambda *_: {
        'private_dm': 'x', 'handoff_reason': 'missing_information'})
    first = send('delivery kab tak?')
    assert first['private_dm'] and first['conversation_status'] == 'ai'
    assert first['requires_human_attention'] is True
    second = send('aur COD?')
    assert second['private_dm'] is None
    assert second['conversation_status'] == 'ai'


def test_model_down_line_is_not_repeated(monkeypatch):
    monkeypatch.setattr(endpoints.gemini_service, 'generate', Mock(side_effect=LLMUnavailable()))
    assert 'mil gaya' in send('size kya hai?')['private_dm']
    assert send('hello?')['private_dm'] is None


def test_pausing_handoff_is_friendly_and_in_hinglish(monkeypatch):
    reply = send('mujhe kisi se baat karni hai, insaan chahiye')
    assert reply['handoff_reason'] == 'human_request'
    assert 'team' in reply['private_dm'].lower()
    assert 'queue' not in reply['private_dm'].lower()


OFFERING = {'id': 'o1', 'title': 'Bridal look', 'price_min': 18000, 'price_label': '₹18,000 se start'}


def test_create_lead_confirms_and_keeps_ai_on(monkeypatch):
    monkeypatch.setattr(endpoints.gemini_service, 'generate', lambda *_: {
        'private_dm': "Your conversation is in the team's queue for review.", 'action': 'CREATE_LEAD'})
    reply = send('12 dec, Patna', offerings=[OFFERING], event_type='dm',
                 playbook={'goal': 'BOOKING', 'lead_fields': [{'key': 'city', 'label': 'City'}]})
    assert reply['action'] == 'CREATE_LEAD'
    assert reply['conversation_status'] == 'ai'
    assert reply['lead_interested'] is True
    assert 'confirm' in reply['private_dm'] and 'queue' not in reply['private_dm']


def test_backend_history_and_goal_state_reach_the_prompt(monkeypatch):
    seen = {}

    def model(system, prompt, *_):
        seen['prompt'] = prompt
        seen['system'] = system
        return {'private_dm': 'Haan', 'action': 'ANSWER'}
    monkeypatch.setattr(endpoints.gemini_service, 'generate', model)
    send('isme hair included hai?', offerings=[OFFERING], event_type='dm',
         recent_messages=[{'from': 'business', 'text': 'Seller: bridal ka rate 18k'}],
         goal_state={'fields': {'city': 'Patna'}},
         playbook={'goal': 'BOOKING', 'lead_fields': [{'key': 'city', 'label': 'City'}]})
    assert 'Seller: bridal ka rate 18k' in seen['prompt']
    assert '"city": "Patna"' in seen['prompt']
    # The old clothing-store SKU prompt no longer rides along.
    assert 'Sales & Customer Engagement Specialist' not in seen['system']
