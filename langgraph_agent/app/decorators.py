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
            # 1. Get the raw output from the wrapped function
            raw_response = func(*args, **kwargs)

            try:
                # If the wrapped function already returned a dict or BaseModel, use it directly
                if isinstance(raw_response, dict):
                    data_dict = raw_response
                elif isinstance(raw_response, BaseModel):
                    return raw_response
                else:
                    # Defensive string extraction
                    clean_json = str(raw_response).strip()

                    # If the LLM wrapped the JSON in a ```json block, extract the inner content
                    if "```json" in clean_json:
                        clean_json = clean_json.split("```json", 1)[1].split("```", 1)[0].strip()
                    elif "```" in clean_json:
                        clean_json = clean_json.split("```", 1)[1].split("```", 1)[0].strip()

                    # Parse the JSON text to a Python dict
                    data_dict = json.loads(clean_json)

                # Instantiate/validate using the provided Pydantic schema
                try:
                    validated_object = schema(**data_dict)
                except Exception:
                    # Pydantic v2 compatibility fallback
                    validated_object = schema.model_validate(data_dict)

                return validated_object

            except (json.JSONDecodeError, ValidationError, IndexError, AttributeError, Exception) as e:
                print(f"❌ Validation Failed: {e}")
                print(f"   Raw Output was: {raw_response}")
                raise e
        return wrapper
    return decorator