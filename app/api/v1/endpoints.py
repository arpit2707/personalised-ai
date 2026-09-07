from fastapi import APIRouter, HTTPException
from typing import List
from app.models.schemas import (
    GenerateReplyRequest,
    GenerateReplyResponse,
    ProductInfo,
    BrandPersona
)
from app.services.guardrails import guardrails
from app.services.vector_store import catalog_store
from app.services.prompt_assembler import prompt_assembler
from app.services.gemini_service import gemini_service

router = APIRouter()

@router.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "personalised-ai",
        "vector_store": "ready"
    }

@router.post("/catalog/product", response_model=dict)
def upsert_product(brand_id: str, product: ProductInfo):
    catalog_store.upsert_product(brand_id, product)
    return {"status": "success", "message": f"Product {product.sku} upserted for brand {brand_id}"}

@router.get("/catalog/search", response_model=List[ProductInfo])
def search_catalog(brand_id: str, query: str, limit: int = 2):
    return catalog_store.search_products(brand_id, query, limit=limit)

@router.post("/generate-reply", response_model=GenerateReplyResponse)
def generate_reply(req: GenerateReplyRequest):
    # Step 1: Guardrail & Sentiment Analysis
    sentiment, requires_human, deescalation = guardrails.evaluate_sentiment_and_safety(req.message_text)
    if requires_human:
        return GenerateReplyResponse(
            public_reply=deescalation if req.event_type == "comment" else None,
            private_dm=deescalation,
            intent="complaint",
            sentiment=sentiment,
            requires_human_attention=True,
            detected_product_sku=None,
            reasoning="Triggered negative/complaint sentiment guardrail. Flagged for human review."
        )

    # Step 2: Product Context Retrieval
    target_product = None
    if req.post_context and req.post_context.tagged_product_sku:
        target_product = catalog_store.get_product_by_sku(req.brand_id, req.post_context.tagged_product_sku)

    if not target_product:
        # Fallback to vector search on the message text
        matched = catalog_store.search_products(req.brand_id, req.message_text, limit=1)
        if matched:
            target_product = matched[0]

    # Step 3: Persona Preparation
    persona = req.brand_persona or BrandPersona(brand_name="Reel2Real Brand")

    # Step 4: Prompt Assembly
    system_prompt = prompt_assembler.build_system_prompt(persona)
    user_prompt = prompt_assembler.build_user_prompt(
        message_text=req.message_text,
        channel_type=req.channel_type,
        event_type=req.event_type,
        post_context=req.post_context,
        product=target_product
    )

    # Step 5: LLM Execution
    llm_output = gemini_service.generate(system_prompt, user_prompt)

    return GenerateReplyResponse(
        public_reply=llm_output.get("public_reply"),
        private_dm=llm_output.get("private_dm"),
        intent=llm_output.get("intent", "general"),
        sentiment="positive" if "love" in req.message_text.lower() or "nice" in req.message_text.lower() else "neutral",
        requires_human_attention=False,
        detected_product_sku=target_product.sku if target_product else None,
        reasoning=llm_output.get("reasoning")
    )
