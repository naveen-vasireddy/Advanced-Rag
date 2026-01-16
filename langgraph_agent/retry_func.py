import time
import random
from langchain_core.runnables import RunnableConfig

def node_with_retry(func, state, node_name, max_retries=3):
    """Utility to handle exponential backoff logic for any node function."""
    delay = 2  # Initial delay in seconds
    for attempt in range(max_retries):
        try:
            return func(state)
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"❌ {node_name} failed after {max_retries} attempts: {e}")
                raise e
            
            # Exponential backoff: 2s, 4s, 8s... plus jitter
            sleep_time = delay * (2 ** attempt) + random.uniform(0, 1)
            print(f"⚠️ {node_name} error. Retrying in {sleep_time:.2f}s... (Attempt {attempt + 1})")
            time.sleep(sleep_time)
