# Constraints and Trade-offs

Good architecture is mostly about choosing what to give up. This document lists the constraints PharmaLink is designed under and the main decisions they led to, including what each decision costs.

---

## 1. Constraints

| Constraint | Why it matters |
|---|---|
| **Patient safety** | Errors in a pharmacy can harm people. AI mistakes carry real consequences. |
| **Health-data regulation** | Strict rules on personal and medical data (Tunisian data-protection law, GDPR principles, health-data hosting standards). |
| **Limited local data** | Access to quality, representative Tunisian pharmacy data is restricted. |
| **Uneven digital maturity** | Pharmacies range from paper-and-invoice to fairly digital. |
| **Unreliable connectivity** | A pharmacy cannot stop serving patients because the network is down. |
| **Language** | Patients and staff use French and Tunisian Arabic, a dialect with few NLP resources. |
| **Change resistance** | Pharmacists must trust AI in a critical process before they use it. |
| **Startup resources** | Small team, limited budget, need to ship an MVP and learn from real pharmacies. |
| **National integration** | Links with insurer and regulator systems depend on official gateways that are not fully available yet. |
| **Sustainability** | Heavy AI workloads have an energy cost we want to keep in check. |

## 2. Decisions and trade-offs

### D1. Assistive AI, not automation

- **Decision:** the pharmacist validates every dispensing decision; major risks block and escalate.
- **Alternatives considered:** automatic dispensing for low-risk cases; a fully autonomous assistant.
- **Why:** safety, professional responsibility and user trust come first.
- **Cost accepted:** less time saved per prescription than full automation would give. We recover it elsewhere through pre-filled sheets, ranked alerts and fewer errors.

### D2. Modular, service-oriented architecture

- **Decision:** independent modules communicating through secured APIs and an event bus.
- **Alternatives considered:** a single monolithic application.
- **Why:** pharmacies adopt at different paces, modules need to evolve independently, and the system must scale from pilots to a network.
- **Cost accepted:** more operational complexity (deployment, monitoring, inter-service contracts). Mitigated with containerisation, CI/CD and centralised observability.

### D3. Hybrid edge + cloud processing

- **Decision:** run latency-sensitive and offline-critical work locally, and heavy analytics in the cloud.
- **Alternatives considered:** cloud-only.
- **Why:** lower latency, continuity during outages, and less sensitive data leaving the pharmacy.
- **Cost accepted:** a local server to deploy and maintain per pharmacy, and model and software updates to synchronise across sites.

### D4. Data minimisation over data richness

- **Decision:** no directly identifying data in analytics or model improvement; pseudonymisation when medical data is needed.
- **Alternatives considered:** training on full patient-level data.
- **Why:** privacy obligations and patient trust.
- **Cost accepted:** potentially lower model performance and harder debugging. Mitigated by domain expertise from pharmacists and by progressive enrichment with aggregated data.

### D5. Multi-tenant SaaS

- **Decision:** many pharmacies on shared infrastructure with logical data isolation.
- **Alternatives considered:** a separate deployment per pharmacy.
- **Why:** lower cost per pharmacy, faster rollout, easier updates.
- **Cost accepted:** isolation must be rigorously designed and tested, and one tenant's load must not affect another's.

### D6. Tamper-evident audit trail

- **Decision:** critical actions are recorded in an append-only, tamper-evident log, designed as a lightweight private ledger.
- **Alternatives considered:** conventional application logs only.
- **Why:** access to health data must be provable and trustworthy for patients, pharmacists and auditors.
- **Cost accepted:** extra engineering and storage. We keep the mechanism as simple as the guarantee allows rather than adopting heavy distributed-ledger infrastructure.

### D7. Bilingual by design

- **Decision:** French and Tunisian Arabic across interface, notifications and assistant.
- **Alternatives considered:** French only, or standard Arabic.
- **Why:** real usage in Tunisian pharmacies and better accessibility for patients.
- **Cost accepted:** more content and testing effort, and a harder NLP problem because of limited dialect resources. Sensitive requests are escalated to a human rather than answered by the assistant.

### D8. Low-carbon hosting and efficient engineering

- **Decision:** certified low-carbon cloud providers, caching, compression and lazy loading.
- **Alternatives considered:** default hyperscale hosting without sustainability criteria.
- **Why:** lower environmental impact and lower running costs.
- **Cost accepted:** a narrower choice of providers and managed services, and attention to portability.

## 3. Known limitations

- Real-time integration with national systems depends on official gateways that remain to be established.
- Models would benefit from training on local, anonymised data as it becomes available.
- User experience needs validation at larger scale with more pharmacies.
- Smart offline mode needs validation in real-world conditions.

See the [roadmap](roadmap.md) for how these are addressed.

## 4. Related documents

- [Architecture](architecture.md)
- [AI pipeline](ai-pipeline.md)
- [Roadmap](roadmap.md)
