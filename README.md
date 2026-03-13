# 🚀 UNSPSC Hybrid RAG POC – Simplified TODO

## 1️⃣ Prepare the Reference Data

* [ ] Export UNSPSC reference to clean CSV
* [ ] Add `full_path` column (breadcrumb string)
* [ ] Add `org_synonyms` column (manual keyword enrichment)
* [ ] Create `search_text = desc + path + synonyms`

---

## 2️⃣ Set Up Infrastructure (Docker Compose)

* [ ] Spin up **Meilisearch** (keyword search)
* [ ] Spin up **Qdrant** (vector search)
* [ ] Create FastAPI service
* [ ] (Optional) Add Postgres for logging + feedback

---

## 3️⃣ Index UNSPSC Data

* [ ] Load CSV into Meilisearch (for BM25)
* [ ] Generate embeddings for each UNSPSC row
* [ ] Store embeddings in Qdrant
* [ ] Verify both searches return results

---

## 4️⃣ Build Retrieval Logic

* [ ] Create query builder from PO fields
* [ ] Run BM25 search → top 50
* [ ] Run vector search → top 50
* [ ] Implement RRF fusion logic
* [ ] Return top 20 fused candidates

---

## 5️⃣ Add LLM Reranking

* [ ] Create structured prompt template
* [ ] Pass top 20 candidates to LLM
* [ ] Ask for Top 3 + confidence + rationale
* [ ] Return structured JSON

---

## 6️⃣ Add Guardrails

* [ ] Confidence threshold (e.g. <0.55 → human review)
* [ ] Simple keyword penalties (e.g. “refill” vs “marker”)
* [ ] Log retrieval + recommendation metadata

---

## 7️⃣ Evaluation

* [ ] Collect 200 historical POs with correct UNSPSC
* [ ] Measure Top-1 accuracy
* [ ] Measure Top-3 accuracy
* [ ] Analyze common misclassifications
* [ ] Improve synonyms + rules

---

## 8️⃣ Feedback Loop

* [ ] Store human overrides
* [ ] Update synonyms dictionary
* [ ] Add pattern-based rules
* [ ] Re-evaluate accuracy

---

# 🎯 Definition of POC Success

* Top-3 accuracy > 85%
* < 20% flagged for manual review
* Clear audit trail per recommendation
 # smc-cpg-classify
# procurement-classifier
# procurement-classifier
