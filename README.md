# Reel2Real Personalised AI Core Engine

FastAPI service for merchant-specific product and sales replies, using Google Gemini
(`DEFAULT_MODEL`) and Supabase/pgvector in cloud deployments (persistent Chroma locally). Order tracking, cancellation,
refund and payment actions are outside the AI's scope and go to the human queue.

## Setup

1. Activate `.venv` and install `requirements-local.txt` for local Chroma, or `requirements.txt` for cloud Supabase.
2. Copy `.env.example` to `.env` only if `.env` does not already exist.
3. Configure `GEMINI_API_KEY`, a model available to your account, and
   `SERVICE_API_KEYS={"long-random-backend-secret":"exact-brand-id"}` locally.
   Cloud: set `DATABASE_URL` to the backend Supabase database and `AI_SERVICE_TOKEN`
   to the same secret used by the backend. `render.yaml` preserves the existing Render blueprint.
4. Run `uvicorn app.main:app --reload --port 8000`.

Merchant endpoints accept per-brand `X-Service-Key` or the existing trusted-backend
`X-AI-Service-Token`. For the shared backend token, supply `brand_id` in the request
body/query, or `X-Brand-ID` for inbox/agent routes. Never expose this shared token to clients. The authenticated brand must match
`brand_id`. Missing key configuration returns 503; invalid keys return 401;
merchant mismatches return 403. Health and API docs remain public.
Keys belong in a trusted backend, never a browser. The backend must authenticate
its agents and supply their verified `agent_id`; this service does not implement
agent login or assign availability automatically.

## Conversation policy

`POST /api/v1/generate-reply` keeps the existing request shape. Conversations are
identified by exact `(brand_id, channel_type, sender_id)`. Responses additionally
include `conversation_id`, `conversation_status` (`ai`, `pending`, `active`),
`handoff_reason`, and `lead_interested`.

- Explicit human requests, complaints, order/payment issues, missing or conflicting
  information enter the human queue immediately.
- Purchase assistance, bulk orders and negotiation enter the queue at high priority.
- Ordinary product interest, price and size inquiries can continue with AI.
- Two consecutive customer-reported unsuccessful AI answers trigger a handoff.
  Productive follow-ups reset the counter. Raw reply count alone never triggers it.
- Rules cover common English/Hinglish phrases; model judgment supplements them.
- Invalid/empty model output or provider failure queues the conversation, without
  fabricated prices, stock, shipping promises or checkout links.
- The first handoff returns an honest pending message. Subsequent requests while
  pending/active return null reply fields. The caller must not send a message when
  these are null. A human claim/release race suppresses stale AI responses; other
  concurrent generation conflicts return 409 and should be retried serially.

A reply is generated, not sent: this service never sends DMs, cancels orders or
contacts agents itself. Channel delivery is the integrating backend's responsibility.

## Merchant inbox integration

| Endpoint | Purpose |
| --- | --- |
| `GET /api/v1/inbox?limit=50&offset=0` | Pending/active conversations, priority first |
| `GET /api/v1/conversations/{id}` | Recent transcript, preferences and derived handoff summary |
| `POST /api/v1/conversations/{id}/claim` | Body `{"agent_id":"agent-1"}`; pending to active |
| `POST /api/v1/conversations/{id}/messages` | Body includes `agent_id`, `message_text`; records an assigned agent's reply |
| `POST /api/v1/conversations/{id}/release` | Assigned agent returns conversation to AI and resets failure counter |

The dashboard polls the inbox and displays reason, product, priority and recent
customer messages. Agent replies must be delivered by the backend and then recorded
here. Only the assigned agent may record messages or release. There is no dashboard
UI or external support connector in this repository. If nobody claims a conversation,
it stays pending and AI stays paused (subject to the 20-day inactivity expiry).

## Memory and retention

With `DATABASE_URL`, Supabase tables `ai.conversations`, `ai.messages`, and
`ai.preferences` persist memory across Render restarts/deployments. Initialization
creates only these AI tables; the backend's product tables are not modified.
Without `DATABASE_URL`, local SQLite (`CONVERSATION_DB_PATH`) persists chat and explicit size/color/language
preferences across restarts. In both modes preferences require a supporting quote from the current
customer message; invented evidence is rejected. Memory is isolated by merchant,
channel and sender; identities across channels are not assumed to be the same person.

Each message and preference expires 20 days after it was saved. Reading a preference
or continuing the chat does not refresh its timestamp. Product/lead observations also
expire. Inactive conversation records expire after 20 days. Cleanup runs at startup,
every 60 seconds while running, and before every store operation; expired data is
never supplied to the model. If the service is off, physical cleanup resumes on
startup. SQLite secure deletion is enabled locally; Postgres uses row deletion and normal vacuum/WAL lifecycle. Independently managed backups must use
the same retention policy.

The model receives at most 20 recent messages and 12,000 history characters plus
unexpired preferences and current product context. Older unexpired messages remain
in storage but are not all resent. Handoff summaries are recent-message excerpts,
derived at read time; no permanent summary can retain expired text. Current request
text is supplied separately. Preferences are extracted during AI-handled turns.

## Strict reply safety

Only clear harm is refused: sexual or pornographic content, racism, terrorism, named
weapon agents, and making or using a bomb or poison ("bomb kaise banaye", "kisi ko
zeher"). Everyday shopping words that merely sound alarming are answered normally:
"Ethnic wear hai?", "Bombay delivery?", "isme chemical hai?", "ye lehenga bomb hai".
A customer reporting harassment goes to the seller as a complaint. The exact refusal is:

`sorry can't help you in that`

The customer's message and the generated output are checked; seller catalog text is
not. Crisis messages get helplines first. Refusal takes priority over complaint/sales
routing, uses no brand styling, and does not create a sales lead. DM refusals populate
only `private_dm`.

This remains lexical screening plus model instructions, not a certified semantic
moderator or a guarantee against every language, paraphrase or adversarial bypass.

## Facts the reply may state

Every reply is checked in code before it is returned (`catalog_reply.blocked`); a
reply that fails is replaced by the "team will confirm this" line and flagged for the seller:

- Prices: any amount (₹/Rs, "2500 ka", "sirf 1499 me", "2.5k", "2 hazar", "rupaye")
  must be a catalog price, an amount in the item text or business policies/FAQs, one
  the customer wrote, or a price times a quantity the customer gave ("200 log").
- Percentages ("20% off", "50% advance") must come from the catalog, business details,
  the post caption/note or the customer.
- Delivery times, COD, free/same-day delivery, returns/exchange and warranty must be in
  the seller's own text; the customer asking "COD hai?" is not a source.
- An option marked `in_stock: false` is never called available.
- Links (with or without `https://`, including `wa.me/` and bare shop domains) must be
  item links, Spotlight permalinks or links in the business details.
- Public comment replies carry no link, no price unless the commenter asked, and at
  most 25 words.
- A price for "ye wala" on a post with no linked item is held unless the customer,
  caption or chat names the item.

The backend runs the same price and percentage check again before sending.

## Catalog storage and compatibility

When `DATABASE_URL` is configured, products come directly from the backend's
`Product` table; Gemini embeddings are stored in `ai.product_embedding`. Changed
products refresh their embeddings; keyword search is used if embeddings are unavailable.
`POST /api/v1/catalog/reindex?brand_id=...` refreshes embeddings. Product writes to
this service return 409 in cloud mode; edit through the seller dashboard.

In local Chroma mode, products support persistent `upsert` by SKU. Descriptions are preserved on retrieval.
Collection names now hash the exact brand ID, preventing punctuation, case and
truncation collisions. Existing legacy `catalog_*` collections are left untouched;
re-ingest each merchant's source catalog through `/catalog/product` into `catalog_v2_*`
before switching traffic. Automatic migration cannot safely infer the original
merchant from the old lossy collection name.

The local Chroma embedding function is a lightweight word-hashing baseline; cloud
pgvector uses `EMBEDDING_MODEL` (default `gemini-embedding-001`). Retrieval quality and model handoff judgments still need
real merchant examples and evaluation before production rollout.

## Backend catalog and playbook compatibility

The remote catalog-playbooks branch is integrated: requests can carry `business`,
`playbook`, `offerings`, `goal_state`, and `recent_messages`. Supplied offerings are
authoritative for that request; this path does not query an unrelated product catalog.
Both paths enforce the same safety, 20-day memory, queue and AI-pause policy.
Responses include `action`, `offering_ids`, and `collected_fields` for backend compatibility.
Unknown prices, facts and links route to humans (see above). A collected lead detail
must be something the customer wrote, compared by kind: dates are read from their words
("15 Nov", "15/11", "kal", "agle Sunday") against `today` (the seller's date, sent by
the backend; India's date otherwise), phones by digits, budgets by amount and cities with
their old names (Bombay, Calcutta). `CREATE_LEAD` stands only when every lead detail is
known, else the reply asks for what is missing; it never places an order or confirms a
booking. Provider failure also hands off instead of claiming a DM was sent.

The backend's `recent_messages` and `goal_state` (details already collected, stage) are
part of the prompt; the service's own retained memory is added when the backend sends
no history. Backend-selected offering IDs
are returned to the caller; customer identity across channels is not merged.

## Validation and observability

Run `.\.venv\Scripts\python.exe -m pytest -q`. Tests use temporary databases and mocked
LLM calls, never the configured real API key. To see how the real model does, run the eval set
(`scripts/eval_cases.json`: boutique, salon, caterer and homestay cases) against it:

    GEMINI_API_KEY=... python scripts/eval_replies.py
    python scripts/eval_replies.py --provider CLAUDE --model <model> --api-key <key>

It prints each failed case with the reply, the pass rate, and how often a guard had to
stop a made-up price, fact or link. Run it after every prompt or guard change. Coverage includes expiry, isolation,
preference evidence, queue lifecycle, generation races, safety and provider failures.
Successful model calls log elapsed time and input/output token usage without customer
text. Pricing is not hardcoded; use measured tokens and the chosen model's pricing to
set a monthly budget. The provider timeout defaults to 15 seconds per attempt, with
at most two attempts. This is not an end-to-end response-time guarantee.

Postgres integration tests require an isolated `TEST_DATABASE_URL` containing the
backend schema. They modify test records and must never target a production database.
Without it, those tests are skipped; unit tests still exercise auth, SQL adaptation,
catalog validation, local persistence and the handoff lifecycle.
