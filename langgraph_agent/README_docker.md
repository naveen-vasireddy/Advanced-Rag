# Docker Build Documentation (Day 45)

## 1. Build Context
**Critical:** The build command MUST be run from the root directory (`ADVANCED-RAG`), not inside `langgraph_agent`.
- **Command:** `docker build -t rag-agent-v2 .`
- **Reason:** To allow the Dockerfile to `COPY` the `langgraph_agent` directory structure correctly so Python imports work.

## 2. Dockerfile Fixes
- **Timeout:** Added `--default-timeout=1000` to `pip install`.
    - *Reason:* Large ML libraries (numpy, chroma) were timing out on the default 15s limit.
- **Directory Structure:** Added `COPY . /app/langgraph_agent` and `ENV PYTHONPATH`.
    - *Reason:* Manually restores the package structure inside the container so `from langgraph_agent.app...` works.

## 3. Connectivity
- **Ollama:** The container uses `host.docker.internal` to talk to the host machine's Llama 3.2 model.