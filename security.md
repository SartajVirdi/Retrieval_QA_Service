# Security Policy

This document describes how we handle and report security issues for this repository and the running service (if deployed).

---

## Reporting a Vulnerability

If you believe you’ve found a security issue, please **do not open a public GitHub issue**.

- Preferred: Open a **GitHub Security Advisory** (Security > Advisories > Report a vulnerability).
- Alternative: Email: **dragontech172@gmail.com** (PGP optional).
- Include: affected commit/branch or release, reproduction steps, impact, and any logs/PoCs.

We commit to the following **SLA** for incoming reports:

| Severity (CVSS-ish) | Triage Acknowledgement | Initial Fix ETA | Public Disclosure |
| --- | --- | --- | --- |
| Critical (RCE, key leakage, auth bypass) | 24 hours | 72 hours | After fix/mitigation |
| High (privilege escalation, data exfiltration) | 48 hours | 7 days | After fix/mitigation |
| Medium (info leak, SSRF with limits) | 3 business days | 14 days | After fix/mitigation |
| Low (best-practice deviation) | 5 business days | 30 days | After fix/mitigation |

We practice **coordinated disclosure**: we’ll work with you on timelines and credit.

Safe-harbor: If your testing follows the guidelines below and avoids privacy violations, data destruction, or service degradation, we will not pursue legal action.

---

## Scope

**In scope**
- Source code in this repository (FastAPI app, retrieval/indexing logic).
- Docker image built from the provided `Dockerfile`.
- Public deployments of this service under our control (e.g., Render).

**Out of scope**
- Third-party models and APIs (e.g., OpenRouter, Anthropic/GPT backends).
- DoS/volumetric attacks and automated fuzzing at destructive rates.
- Social engineering of maintainers, hosting providers, or users.

---

## Security Posture & Controls

### Branch & Merge Protections (Main)
We enforce the following on the default protected branch:
- Restrict updates (no direct pushes to `main`).
- Restrict deletions.
- Require pull request before merging.
- Require signed commits.
- Require conversation resolution before merging.
- Dismiss stale PR approvals on new commits.
- Require review from Code Owners (where applicable).
- Allowed merge methods: **Squash** only.
- Block force pushes.

> Tip: open PRs from feature branches; ensure your commits are GPG/Sigstore signed.

### Reviews & CI
- At least **1 approving review** is required.
- Status checks (tests/lint/type/security) should pass before merge (enable in repository settings when CI is added).
- Optional but recommended: enable **Code Scanning Alerts** (CodeQL) and **Secret Scanning** for private repos.

### Dependencies & Supply Chain
- Dependencies are version-pinned in `requirements.txt`.
- Recommended:
  - Enable **Dependabot** for Python and GitHub Actions updates.
  - Run `pip-audit` in CI to surface known CVEs.
  - Review FAISS/PyMuPDF changes before bumping.
- Only use official base images (`python:*-slim`). Avoid adding compilers/build tools in production layers.

### Secrets Management
- Secrets are **never committed** to the repo. Use environment variables at runtime.
- Required secrets:
  - `OPENROUTER_API_KEY`
- Rotation policy:
  - Rotate keys immediately if exposure is suspected.
  - Remove exposed keys from commit history (if any) and invalidate them upstream.
- Logs must never include API keys, tokens, or full user content. Redact on error paths.

### App-Level Hardening
- PDF ingestion has **size** and **page** caps; HTTP requests have **timeouts** and **retries**.
- Chunking and indexing happen **in-memory**; no PDFs persisted to disk in production images.
- CORS is **enabled by default** for development; set `ENABLE_CORS=false` or restrict `allow_origins` for production.
- External LLM calls use a fixed base URL and model identifier; handle upstream errors and timeouts.
- Evidence returned to clients is limited to matched chunks—avoid returning entire documents.

### Network & Runtime
- The container runs as a **non-root** user.
- Only port `8000` is exposed internally; on Render, the service binds to `$PORT`.
- Health endpoint: `/health` (no secrets, no DB access).
- Prefer HTTPS on the edge; do not terminate TLS inside the container.

---

## Responsible Testing Guidelines

You may:
- Send requests to your own deployment with PDFs you are authorized to test.
- Validate rate limiting, input validation, and context leaks within reasonable volumes.

You must not:
- Exfiltrate data you don’t own or have permission to access.
- Attempt persistent DoS, account takeover, or social engineering.
- Use other people’s API keys or induce the service to store secrets.

If in doubt, contact **security@yourdomain.example** before proceeding.

---

## Hardening Checklist (Maintainers)

- [ ] Protect default branch with rules above (done).
- [ ] Require status checks to pass (enable once CI exists).
- [ ] Enable Dependabot for Python and Docker.
- [ ] Enable Code Scanning (CodeQL) and Secret Scanning.
- [ ] Configure CODEOWNERS for critical paths (`app/`, `Dockerfile`, CI).
- [ ] Restrict CORS in production (`ENABLE_CORS=false` or set explicit origins).
- [ ] Ensure logs redact `Authorization` and environment variables.
- [ ] Rotate `OPENROUTER_API_KEY` at least quarterly or upon suspicion.
- [ ] Periodically rebuild images to pick up base-image security patches.

---

## Contact & Credits

- Security contact: **dragontech172@gmail.com**
- Please include “`[SECURITY] <short title>`” in your subject.
- We will credit reporters in the changelog or advisory unless you prefer to remain anonymous.
