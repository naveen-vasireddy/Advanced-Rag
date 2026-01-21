from functools import wraps
from pydantic import ValidationError
import json
from pydantic import BaseModel

def validate_llm_output(schema: BaseModel):
    """
    Wraps an LLM function. It parses the JSON string response 
    and validates it against the provided Pydantic schema.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 1. Get the raw string output from the LLM chain
            raw_response = func(*args, **kwargs)
            
            try:
                # 2. Extract JSON (Llama 3.2 often wraps code in ```json ... ```)
                clean_json = raw_response.strip()
                if "```json" in clean_json:
                    clean_json = clean_json.split("```json")[3].split("```")
                elif "```" in clean_json:
                    clean_json = clean_json.split("```")[3].split("```")
                
                # 3. Parse and Validate
                data_dict = json.loads(clean_json)
                validated_object = schema(**data_dict)
                
                return validated_object
                
            except (json.JSONDecodeError, ValidationError) as e:
                print(f"❌ Validation Failed: {e}")
                # Raising this triggers the Day 37-39 Retry Logic you built
                raise e 
                
        return wrapper
    return decorator