# Agentic AI Data Governance Policy

## 1. Data Classification

- **Public Data**  
  Documentation, generic queries.  
  → Allowed to be sent to Public LLMs (OpenAI/Anthropic).

- **Internal Data**  
  Project codes, non-sensitive memos.  
  → Allowed to be sent to Public LLMs.

- **Restricted (PII)**  
  Names, Emails, Phone Numbers.  
  → **Must be redacted** (using Day 55 Agent) before leaving the secure perimeter.

- **Critical (PCI/PHI)**  
  Credit Cards, SSNs, Health Data.  
  → **Never processed.** The Agent must reject these queries entirely.

---

## 2. Data Residency & Architecture

### Vector Database (Long-Term Memory)

- Must never store raw PII in embeddings.
- If PII is required for search, it must be stored as hashed metadata fields  
  (e.g., `user_hash: "a1b2..."`), not in `page_content`.

### Session State (Short-Term Memory)

- The **Deanonymization Map (Day 56)** exists only in ephemeral memory (Redis/RAM).
- It is purged immediately after the conversation session ends.  
  **TTL = 1 hour**

---

## 3. The "Right to be Forgotten" (GDPR/CCPA)

### Requirement

- If a user requests deletion, all their data must be removed.

### Mechanism

- Using Metadata Filtering (Day 53):

```python
vector_store.delete(filter={"user_id": "JohnDoe"})
```
### Compliance Limitation of Standard RAG Systems

- Standard RAG systems without metadata tagging are **non-compliant**.
- User-specific chunks cannot be reliably identified or deleted.
- This prevents proper enforcement of data erasure requirements (GDPR/CCPA).

---

## 4. Human-in-the-Loop (HITL) Protocol

If the PII Redaction Agent returns a confidence score **below 0.60 (60%)**, the following actions must occur:

- The system must trigger a **Fallback Flow**.
- The user must be asked to confirm their input.
- The system must not guess, infer, or reconstruct sensitive data.