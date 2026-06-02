# Ascendic Workflow 🤖

An **agentic customer support system** for merchants — built to answer customer queries about products and store policies with near-zero hallucinations.

---

## 💡 Why I Built This

My first RAG chatbot kept hallucinating — giving wrong product names, made-up prices, and fake policies. Instead of accepting it, I analysed the root cause and rebuilt the entire system from scratch with two major upgrades:

1. A **multi-stage RAG pipeline** to eliminate retrieval noise
2. A **ReAct agent loop** to replace rigid intent-based routing with dynamic reasoning

---

## 🧠 Architecture

### Version 1 — Advanced RAG Pipeline
```
User Query
    ↓
Intent Analysis         ← understands what user really means
    ↓
Query Rewriting         ← rewrites based on conversation history
    ↓
Retrieval               ← fetches top chunks from FAISS vector store
    ↓
Cross-Encoder Reranking ← reranks top 9 → extracts best 3
    ↓
Context Compression     ← removes noise, keeps only relevant context
    ↓
Response Generation     ← clean, grounded answer
```

### Version 2 — ReAct Agentic System (current)
```
User Query
    ↓
ReAct Agent             ← LLM reasons and decides what to do
    ↓
Tool Selection          ← agent picks the right tool dynamically
    ↓
search_products         ← Retriever → Reranker → Compressor
search_policies         ← direct JSON lookup
    ↓
Observation             ← tool result fed back to agent
    ↓
Agent reasons again     ← loops until it has enough information
    ↓
Final Answer            ← grounded, honest, hallucination-free
```

---

## 🔄 What Changed From V1 to V2

| Component | V1 | V2 |
|---|---|---|
| Intent Classifier | Manual LLM call | Removed — agent decides |
| Query Rewriter | Separate LLM call | Removed — agent formulates input |
| Generator | Separate LLM call | Removed — agent generates answer |
| Routing | Fixed if/else branches | Dynamic tool selection |
| Multi-topic queries | One intent only | Agent calls multiple tools |
| LLM calls per query | 4-5 | 2-3 |

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| LLM | [Groq](https://groq.com) — llama-3.3-70b-versatile |
| Vector Store | FAISS (Facebook AI Similarity Search) |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| Reranker | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| Backend | Python / FastAPI |
| Memory | JSON session store with sliding window |

---

## 🔍 How the ReAct Loop Works

```
Iteration 1:
  THOUGHT: Customer wants leather iPhone case, need to search
  ACTION: search_products
  INPUT: leather iPhone case
  OBSERVATION: Apple Leather Finish Case — $45, premium texture...

Iteration 2:
  THOUGHT: Found the product. Customer also asked about stock.
  ACTION: search_products
  INPUT: Apple Leather Finish Case stock level
  OBSERVATION: No stock information available.

Iteration 3:
  THOUGHT: I have enough information to answer honestly.
  FINAL ANSWER: We have the Apple Leather Finish Case at $45.
                Stock info unavailable — please contact us to confirm.
```

No hallucination. No made-up product names. No fake stock numbers.

---

## 💼 Use Case

Built for **merchants** running Instagram or WhatsApp stores who need an intelligent assistant that can:
- Answer customer questions about products accurately
- Handle vague, multi-topic, or conversational queries
- Search both product catalog and store policies in one response
- Reason across multiple steps before answering
- Say "I don't know" honestly when data is unavailable

---

## ⚙️ Getting Started

### Prerequisites
- Python 3.9+
- Groq API key

### Installation

```bash
git clone https://github.com/yourusername/ascendic-workflow.git
cd ascendic-workflow
pip install -r requirements.txt
```

### Environment Variables

```env
GROQ_API_KEY=your_groq_api_key
```

### Run

```bash
python app.py
```

---

## 📂 Project Structure

```
ascendic-workflow/
├── agent/
│   ├── agent.py              # ReAct loop
│   ├── tool_registry.py      # Tool definitions
│   ├── tools/
│   │   ├── search_products.py  # RAG pipeline tool
│   │   └── search_policies.py  # Policy lookup tool
│   └── data/
│       └── policies.json       # Store policies
├── rag_pipeline/
│   ├── retriever.py          # FAISS vector search
│   ├── reranker.py           # Cross-encoder reranking
│   ├── compressor.py         # Context compression
│   └── vector_store.py       # Index builder
├── llm_wrapper.py            # Groq API wrapper
├── memory_store.py           # Session memory
├── app.py                    # Entry point
└── requirements.txt
```

---

## 🔮 Roadmap

- [ ] `get_stock` tool — real-time stock lookup
- [ ] `place_order` tool — agentic order collection
- [ ] `check_order_status` tool — order tracking
- [ ] WhatsApp / Instagram API integration
- [ ] Multi-merchant support
- [ ] Evaluation metrics (faithfulness, answer relevancy)

---

## 👤 Author

Built with 🔥 by [Your Name](https://github.com/yourusername)

---

## 📄 License

MIT License
