# Frontend ↔ Backend Sync Review (AHVI)

Date: 2026-04-17

This review is based on the current backend in this repository. It lists concrete mismatches that should be updated so frontend and backend stay in sync.

## 1) Endpoint paths: standardize and lock a single API map

### Current backend path behavior
- Chat is mounted as `POST /api/text` (router has `/text`, app adds `/api` prefix).
- Boards are under `/api/boards/*`.
- Data CRUD is under `/api/data/*`.
- Utilities expose fully-qualified `/api/...` paths directly in the router.
- Ops metrics are at `/api/ops/metrics`.
- Vision/background have **multiple compatibility aliases**:
  - Analyze: `/api/analyze-image`, `/api/vision/analyze-image`, `/api/vision/analyze`, `/api/analyze`
  - Background: `/api/background/remove-bg`, `/api/remove-bg`

### What should be updated
- Frontend should consume one canonical path per feature and treat aliases as temporary fallback only.
- Add a shared `apiRoutes` constants file in frontend and map only canonical endpoints.
- Mark deprecated aliases and remove usage gradually.

---

## 2) Request field naming mismatch (`user_id` vs `userID` vs `userId`)

### Current backend inconsistency
- Chat accepts both `user_id` and `userID`.
- Vision compatibility endpoint accepts both `user_id` and `userId`.
- Vision router request model uses only `userId`.
- Wardrobe capture uses `user_id`.

### What should be updated
- Use one key everywhere: **`user_id`**.
- Frontend should send `user_id` in all payloads.
- Backend should continue temporary alias support and return a warning header/log when non-canonical keys are used.

---

## 3) Error contract mismatch across endpoints

### Current behavior
- Many endpoints return `{ "success": true, ... }` on success.
- Some failures raise `HTTPException(detail=...)`, returning FastAPI default error shape (`{ "detail": ... }`).
- Global exception handler returns a structured `{ success, request_id, error }` envelope for uncaught exceptions.

### What should be updated
- Frontend error parser must currently support both:
  1) `error.message` envelope
  2) `detail`
- Recommended backend follow-up: normalize all errors into one format (prefer the existing structured error envelope).

---

## 4) Compatibility endpoints can hide feature unavailability

### Current behavior
- Compatibility routes for vision/bg are always exposed in `main.py`.
- If optional modules are unavailable, these endpoints can return 503 with fallback/manual-input payloads.

### What should be updated
- Frontend should detect `503` and `requires_user_input: true` cases and route user to manual input flow.
- Do not assume `200` means full vision pipeline was executed; inspect `meta` fields.

---

## 5) Auth/rate-limit/request-id headers: frontend should read them

### Current behavior
- Auth middleware can be enabled globally.
- Rate-limit middleware adds headers like:
  - `X-RateLimit-Remaining`
  - `X-RateLimit-Limit`
  - `X-RateLimit-Window`
- Request tracing adds `X-Request-ID`.

### What should be updated
- Frontend network layer should surface `X-Request-ID` in error logs/support UI.
- Handle 429 with retry UX using `Retry-After`.
- Treat 401/403 as session-expired state and trigger re-auth path.

---

## 6) Immediate frontend checklist (high priority)

1. Update all API calls to canonical paths only.
2. Send `user_id` consistently in every request body/query param.
3. Add robust error parsing for both envelope + `detail` responses.
4. Add retry/backoff handler for 429 and temporary 503.
5. Log/display `X-Request-ID` for failed requests.

---

## 7) Recommended backend follow-up changes (to reduce future drift)

1. Add a documented `openapi-contract-version` response header.
2. Implement one Pydantic request alias policy (`user_id` only, with explicit aliases where needed).
3. Introduce a unified API response model for success/error.
4. Deprecate duplicate analyze/remove-bg aliases with sunset dates.
5. Generate and publish an endpoint contract file used by frontend CI.

---

## 8) Canonical endpoint list for frontend (proposed)

- `POST /api/text`
- `POST /api/ai/run`
- `GET /api/data/{resource}`
- `GET /api/data/{resource}/{document_id}`
- `POST /api/data`
- `PATCH /api/data/{document_id}`
- `DELETE /api/data`
- `POST /api/data/outfits/duplicate-check`
- `GET /api/boards`
- `POST /api/boards/save`
- `GET /api/boards/life`
- `POST /api/boards/life/save`
- `DELETE /api/boards/{document_id}`
- `POST /api/wardrobe/capture/analyze`
- `POST /api/vision/analyze-image` (canonical; keep other aliases fallback-only)
- `POST /api/background/remove-bg` (canonical; keep alias fallback-only)
- `GET /api/ops/metrics`
- `GET /health`

