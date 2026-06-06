# Agentic 🤖

A **production-grade agentic assistant** — built from scratch with a full orchestrator, planning layer, replanning, and smart memory management. Designed to work for any merchant, business, or personal use case.

---

## 💡 The Journey — Why I Built This

Started as a simple RAG chatbot that hallucinated constantly. Rebuilt it four times:

- **V1** — Advanced RAG pipeline — fixed hallucination
- **V2** — ReAct agent — replaced rigid intent routing with dynamic reasoning
- **V3** — Replanning — agent retries intelligently when searches fail
- **V4 (current)** — Orchestrator + Planning — agent classifies tasks, plans multi-step execution, stores proactive goals

---

## 🧠 Architecture

```
User Query
    ↓
app.py
    ↓
Orchestrator (brain)
    ↓ classifies task type
    ├── SIMPLE      → answers directly, no tools
    ├── SINGLE_TOOL → ReAct agent (1 tool call)
    ├── MULTI_STEP  → generates plan → executes step by step → synthesizes
    └── PROACTIVE   → stores in goal memory for future execution
    ↓
ReAct Agent (executor)
    ↓ for each step
    ├── THOUGHT → reasons about what to do
    ├── ACTION  → calls tool
    ├── OBSERVATION → gets result
    ├── REPLAN  → if empty result, tries simpler query (max 2 replans)
    └── FINAL ANSWER → grounded, honest answer
    ↓
Memory System
    ├── Short-term → current conversation (sliding window)
    ├── Working    → active plan + step progress
    ├── Goal       → proactive tasks for future execution
    └── Long-term  → purchase history (injected only when relevant)
```

---

## 🔄 Evolution — V1 → V4

| Component | V1 | V4 |
|---|---|---|
| Intent Classifier | Manual LLM call | Removed — orchestrator classifies |
| Query Rewriter | Separate LLM call | Removed — agent formulates naturally |
| Generator | Separate LLM call | Removed — agent generates directly |
| Routing | Fixed if/else | Dynamic — orchestrator decides |
| Planning | None | LLM generates plan for complex tasks |
| Replanning | None | Smart retry with simpler query |
| Memory | Flat JSON | Working + Goal + Long-term |
| LLM calls per query | 4–5 | 2–3 (simple) / per step (complex) |

---

## 🗂️ Task Classification — Real Examples

```
"what does RAG mean?"
→ SIMPLE → answered directly, no tools

"do you have leather Samsung case?"
→ SINGLE_TOOL → ReAct agent → search_products → answer

"find rugged case under $30 AND check return policy"
→ MULTI_STEP → plan generated:
   Step 1: Search for rugged cases
   Step 2: Check return policy
→ each step executed → results synthesized → one clean answer

"remind me to reorder stock next Monday"
→ PROACTIVE → stored in goal memory → confirmed to user
```

---

## 🔁 Replanning — Smart Retry Logic

```
search("Apple iPhone 13 Pro Max ultra slim transparent case")
→ nothing found → query was 8 words → REPLAN
→ search("Apple clear case")
→ found Apple Clear Case $25 ✅

search("Nokia cases")
→ nothing found → query was 2 words → no replan
→ honest: "we don't carry Nokia cases" ✅
```

---

## 💾 Memory Design

```
sessions/{user_id}.json
├── conversation.messages    ← sliding window, last 10
├── session.order_state      ← for order tools (future)
├── working_memory           ← active plan + step progress
│   ├── current_goal
│   ├── plan: [step1, step2]
│   ├── current_step
│   └── results: [...]
└── goal_memory              ← proactive tasks
    └── active_goals: [...]

profiles/{user_id}.json
└── purchases: [...]         ← injected only when query is about past orders
```

**Smart context injection:**
```
"do you have leather cases?"    → no purchase context injected
"I want to return my order"     → purchase history injected
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| LLM | Groq — llama-3.3-70b-versatile |
| Vector Store | FAISS |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| Reranker | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| Backend | Python / FastAPI |
| Memory | JSON session store |

---

## 📂 Project Structure

```
agentic/
├── orchestrator.py              # Brain — classify, plan, route
├── agent/
│   ├── agent.py                 # Executor — ReAct loop + replanning
│   ├── tool_registry.py         # Tool definitions
│   └── tools/
│       ├── search_products.py   # RAG pipeline tool
│       └── search_policies.py   # Policy lookup tool
├── rag_pipeline/
│   ├── retriever.py             # FAISS vector search
│   ├── reranker.py              # Cross-encoder (top 9 → top 3)
│   ├── compressor.py            # Context compression
│   └── vector_store.py
├── memory_store.py              # Short-term + working + goal memory
├── profile_store.py             # Long-term purchase history
├── llm_wrapper.py               # Groq API wrapper
├── app.py                       # Entry point
└── requirements.txt
```

---

## ⚙️ Getting Started

```bash
git clone https://github.com/yourusername/agentic.git
cd agentic
pip install -r requirements.txt
```

```env
GROQ_API_KEY=your_groq_api_key
```

```bash
python app.py
```

---

## 🔮 Roadmap

- [x] Advanced RAG — retriever + reranker + compressor
- [x] ReAct agent loop — dynamic tool selection
- [x] Hallucination fix — grounded answers only
- [x] Smart replanning — retries specific queries
- [x] Memory — working memory + goal memory + purchase history
- [x] Orchestrator — task classification + planning + routing
- [ ] Sub-agents — product agent, order agent, policy agent
- [ ] Order tools — collect, confirm, place, track
- [ ] Semantic memory — who the person/company is
- [ ] Proactive scheduler — execute goal memory automatically
- [ ] WhatsApp / Instagram API integration
- [ ] Autonomous 24/7 operation

---

## 👤 Author

Built with 🔥 by [Amal Gafoor P K](https://github.com/amal-gafoor)

---

## 📄 License

MIT License
