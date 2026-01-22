# Use python 3.11 slim for a smaller image size
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# 1. Copy requirements first (optimizes Docker cache)
COPY requirements.txt .

# 2. Install dependencies
RUN pip install --default-timeout=1000 -r requirements.txt
# 3. Copy the entire project code
COPY . .

# 4. Ensure the container can import the package by adding /app
# `/app` contains the `langgraph_agent` package after COPY . .
ENV PYTHONPATH=/app:/app/app
ENV PYTHONUNBUFFERED=1

# 5. Expose the port FastAPI runs on
EXPOSE 8000

# 6. Run the server using the path to your file
# Note the dot notation matching your folder structure
CMD ["uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "8000"]