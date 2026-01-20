## 2024-05-22 - [CRITICAL] Password Exposure in Query Parameters
**Vulnerability:** The `/auth/login` endpoint was accepting `email` and `password` as query parameters because the FastAPI route handler defined them as individual arguments instead of using a Pydantic model or Form body. The frontend was also sending them as query parameters.
**Learning:** In FastAPI, function arguments that are not `Body`, `Form`, `Header`, `Path`, `Query`, or Pydantic models are treated as query parameters by default. This can lead to accidental exposure of sensitive data in server logs and browser history.
**Prevention:** Always use Pydantic models or `Body()`/`Form()` for POST request payloads, especially for sensitive data. Verify API schemas to ensure sensitive fields are in the body.

## 2025-01-20 - [CRITICAL] Unauthenticated Ingest Endpoint Vulnerability
**Vulnerability:** The `/api/ingest/raw` endpoint accepted `businessId` in the payload without validating ownership or requiring authentication. This allowed unauthenticated actors to trigger background processing and potentially inject false data into a business's ML feedback loop.
**Learning:** Endpoints that accept an ID (like `businessId`) must always verify that the requestor is authorized to act on behalf of that ID. Reliance on "obscurity" (assuming only the extension sends the ID) is not security.
**Prevention:** Always use authentication dependencies (like `Depends(get_current_user)` or API Key validation) for any endpoint that performs actions on specific resources.
