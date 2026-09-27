from fastapi import APIRouter, Depends, HTTPException, Query
from typing import List
from app.core.auth import authenticated_brand, require_brand
from app.models.schemas import (GenerateReplyRequest, GenerateReplyResponse, ProductInfo,
                                BrandPersona, ModelReply, AgentAction, AgentMessage)
from app.services.guardrails import contains_blocked_content, SAFETY_REFUSAL
from app.services.vector_store import catalog_store
from app.services.prompt_assembler import prompt_assembler
from app.services.gemini_service import gemini_service, LLMUnavailable
from app.services.conversation_store import conversation_store, ConversationConflict
from app.services.handoff_policy import handoff_reason, unresolved_feedback, interested

router = APIRouter()


@router.get("/health")
def health_check():
    return {"status": "healthy", "service": "personalised-ai", "vector_store": "ready"}


@router.post("/catalog/product", response_model=dict)
def upsert_product(brand_id: str, product: ProductInfo, brand: str = Depends(authenticated_brand)):
    require_brand(brand_id, brand)
    catalog_store.upsert_product(brand_id, product)
    return {"status": "success", "message": f"Product {product.sku} upserted for brand {brand_id}"}


@router.get("/catalog/search", response_model=List[ProductInfo])
def search_catalog(brand_id: str, query: str, limit: int = Query(default=2, ge=1, le=50),
                   brand: str = Depends(authenticated_brand)):
    require_brand(brand_id, brand)
    return catalog_store.search_products(brand_id, query, limit=limit)


def safety_refusal(req):
    return GenerateReplyResponse(public_reply=SAFETY_REFUSAL if req.event_type == "comment" else None,
                                 private_dm=SAFETY_REFUSAL, intent="safety_refusal")


def state_fields(row):
    return dict(conversation_id=row['id'], conversation_status=row['status'],
                handoff_reason=row['reason'], lead_interested=row['lead_at'] is not None,
                requires_human_attention=row['status'] != 'ai', detected_product_sku=row['product'])


def paused(row):
    return GenerateReplyResponse(intent="human_handoff", **state_fields(row))


def complete(req, row, **kwargs):
    try:
        return conversation_store.finish(req.brand_id, row, **kwargs)
    except ConversationConflict:
        current = conversation_store.detail(req.brand_id, row['id'])
        if current['status'] != 'ai':
            return current, None
        raise HTTPException(409, "Conversation changed during generation; discard this response and retry")


def handoff(req, row, reason, product=None):
    current, text = complete(req, row, reason=reason, product=product,
                             interested=reason == 'purchase_assistance')
    return GenerateReplyResponse(public_reply=text if req.event_type == 'comment' else None,
                                 private_dm=text, intent='human_handoff',
                                 sentiment='negative' if reason in ('complaint', 'order_support') else 'neutral',
                                 **state_fields(current))


@router.post("/generate-reply", response_model=GenerateReplyResponse)
def generate_reply(req: GenerateReplyRequest, brand: str = Depends(authenticated_brand)):
    require_brand(req.brand_id, brand)
    # Refusals never enter sales memory or become a lead.
    if contains_blocked_content(req.model_dump(mode="json")):
        return safety_refusal(req)
    row = conversation_store.begin(req)
    if row['status'] != 'ai':
        return paused(row)
    memory = conversation_store.context(brand, row['id'])
    # The current message is already supplied separately in the generation prompt.
    if memory['history'] and memory['history'][-1]['role'] == 'user':
        memory['history'].pop()
    failed = unresolved_feedback(req.message_text) and any(m['role'] == 'assistant' for m in memory['history'])
    reason = handoff_reason(req.message_text)
    sku = req.post_context.tagged_product_sku if req.post_context else None
    if reason:
        return handoff(req, row, reason, product=sku)
    if failed and row['unresolved'] >= 1:
        return handoff(req, row, 'unresolved_query')
    target = None
    if sku:
        target = catalog_store.get_product_by_sku(brand, sku)
        if target is None:
            return handoff(req, row, 'missing_information')
    elif row['product'] and (failed or any(word in req.message_text.casefold().split()
                                         for word in ('isme', 'iska', 'it', 'this', 'that', 'same'))):
        target = catalog_store.get_product_by_sku(brand, row['product'])
        if target is None:
            return handoff(req, row, 'missing_information')
    if target is None:
        matched = catalog_store.search_products(brand, req.message_text, limit=1)
        target = matched[0] if matched else None
    if target and contains_blocked_content(target.model_dump(mode='json')):
        return safety_refusal(req)
    if contains_blocked_content(memory):
        return safety_refusal(req)
    persona = req.brand_persona or BrandPersona(brand_name='Reel2Real Brand')
    user_prompt = prompt_assembler.build_user_prompt(
        req.message_text, req.channel_type, req.event_type, req.post_context, target, memory)
    try:
        raw = gemini_service.generate(prompt_assembler.build_system_prompt(persona), user_prompt)
        if raw.get('intent') == 'safety_refusal' or contains_blocked_content(raw):
            return safety_refusal(req)
        output = ModelReply.model_validate(raw)
        if not output.private_dm.strip() and not output.handoff_reason:
            raise ValueError('Empty reply')
    except (LLMUnavailable, ValueError, TypeError, AttributeError):
        return handoff(req, row, 'generation_unavailable', product=target.sku if target else None)
    if output.handoff_reason:
        return handoff(req, row, output.handoff_reason, product=target.sku if target else None)
    failed = failed or (output.previous_answer_unresolved and any(m['role'] == 'assistant' for m in memory['history']))
    current, text = complete(req, row, reply=output.private_dm, product=target.sku if target else None,
                             interested=output.buying_interest or interested(req.message_text), unresolved=failed,
                             preferences=[p.model_dump() for p in output.preferences], source_text=req.message_text)
    if current['status'] != 'ai':
        return GenerateReplyResponse(public_reply=text if req.event_type == 'comment' else None,
                                     private_dm=text, intent='human_handoff', **state_fields(current))
    return GenerateReplyResponse(public_reply=output.public_reply if req.event_type == 'comment' else None,
                                 private_dm=text, intent=output.intent, reasoning=output.reasoning,
                                 **state_fields(current))


@router.get('/inbox')
def inbox(limit: int = Query(default=50, ge=1, le=100), offset: int = Query(default=0, ge=0),
          brand: str = Depends(authenticated_brand)):
    return conversation_store.inbox(brand, limit, offset)


@router.get('/conversations/{conversation_id}')
def conversation_detail(conversation_id: str, brand: str = Depends(authenticated_brand)):
    try:
        return conversation_store.detail(brand, conversation_id)
    except KeyError:
        raise HTTPException(404, 'Conversation not found')


def agent_action(brand, cid, body, action):
    try:
        return conversation_store.agent_action(brand, cid, body.agent_id, action,
                                               getattr(body, 'message_text', None))
    except KeyError:
        raise HTTPException(404, 'Conversation not found')
    except ConversationConflict as exc:
        raise HTTPException(409, str(exc))


@router.post('/conversations/{conversation_id}/claim')
def claim(conversation_id: str, body: AgentAction, brand: str = Depends(authenticated_brand)):
    return agent_action(brand, conversation_id, body, 'claim')


@router.post('/conversations/{conversation_id}/release')
def release(conversation_id: str, body: AgentAction, brand: str = Depends(authenticated_brand)):
    return agent_action(brand, conversation_id, body, 'release')


@router.post('/conversations/{conversation_id}/messages')
def record_agent_message(conversation_id: str, body: AgentMessage, brand: str = Depends(authenticated_brand)):
    # This records an agent reply; channel delivery belongs to the caller.
    if contains_blocked_content(body.message_text):
        raise HTTPException(422, SAFETY_REFUSAL)
    return agent_action(brand, conversation_id, body, 'message')
