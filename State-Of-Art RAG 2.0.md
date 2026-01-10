**State-of-the-Art RAG v2.0 Architecture**

This architecture is managed via a LangChain ChainOfThought, ensuring that every step of the process is recorded in a shared, persistent state using      
TypedDict and operator.add reducers to prevent context overwriting.

### The Architectural Flow

```mermaid
graph TD
    A[User Query] --> B{StateGraph Start}
    B --> C[Query Re-Writer Agent]
    C --> D[Multi-Query Retrieval]
    D --> E[Deduplication & RRF Node]
    E --> F[SLM Re-Ranking Node]
    F --> G[Synthesis Agent]
    G --> H[Final Response]
    H --> I{StateGraph End}

    subgraph "Advanced Retrieval Layer"
        C
        D
        E
    end

    subgraph "Refinement & Optimization"
        F
    end

    subgraph "State Management (Memory)"
        B
        I
    end
```
