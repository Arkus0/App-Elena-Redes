## 2024-05-22 - [CRITICAL] Password Exposure in Query Parameters
**Vulnerability:** The `/auth/login` endpoint was accepting `email` and `password` as query parameters because the FastAPI route handler defined them as individual arguments instead of using a Pydantic model or Form body. The frontend was also sending them as query parameters.
**Learning:** In FastAPI, function arguments that are not `Body`, `Form`, `Header`, `Path`, `Query`, or Pydantic models are treated as query parameters by default. This can lead to accidental exposure of sensitive data in server logs and browser history.
**Prevention:** Always use Pydantic models or `Body()`/`Form()` for POST request payloads, especially for sensitive data. Verify API schemas to ensure sensitive fields are in the body.

## 2025-01-20 - [CRITICAL] Unauthenticated Ingest Endpoint Vulnerability
**Vulnerability:** The `/api/ingest/raw` endpoint accepted `businessId` in the payload without validating ownership or requiring authentication. This allowed unauthenticated actors to trigger background processing and potentially inject false data into a business's ML feedback loop.
**Learning:** Endpoints that accept an ID (like `businessId`) must always verify that the requestor is authorized to act on behalf of that ID. Reliance on "obscurity" (assuming only the extension sends the ID) is not security.
**Prevention:** Always use authentication dependencies (like `Depends(get_current_user)` or API Key validation) for any endpoint that performs actions on specific resources.

## 2025-02-19 - [CRITICAL] Backdoor Authentication Dependency
**Vulnerability:** The `user_config.py` endpoints used a custom `get_current_user_id` dependency that defaulted to ID 1 or accepted a `user_id` query parameter for "testing purposes". This allowed full authentication bypass and IDOR.
**Learning:** Convenience functions for testing/development (like "mock auth") must strictly separate from production code or be gated behind explicit `DEBUG` flags. Leaving them as default dependencies opens critical backdoors.
**Prevention:** Use standard `get_current_user` dependencies everywhere. If testing mocks are needed, use `app.dependency_overrides` in the test suite, not in the application code.

## 2025-05-21 - [HIGH] Insecure Default Secrets in Production
**Vulnerability:** The application was configured with default insecure `SECRET_KEY` and `DYNAMIC_SALT` values in `config.py`. While meant for development, there was no validation preventing these defaults from being used in a production environment (`DEBUG=False`), posing a critical risk of session hijacking and PII exposure.
**Learning:** Instantiating settings objects at module level (e.g., `settings = get_settings()` in `config.py`) means validation logic executes *at import time*. This makes testing challenging because simply importing the module can crash the test runner if the environment isn't pre-configured.
**Prevention:** Implement strict startup validation using Pydantic's `@model_validator` to refuse startup if `DEBUG=False` and defaults are detected. Use `backend/tests/conftest.py` to force `DEBUG=True` during testing to bypass this check safely.
