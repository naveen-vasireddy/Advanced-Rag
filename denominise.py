import uuid
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

# Global "Memory" for the session to store the mapping
# In production, this would be Redis with a short TTL (Time-To-Live)
pii_mapping = {}

def mock_llm_response(prompt):
    """
    Simulates an LLM that ONLY sees the redacted text.
    Notice it echoes back the tags like <EMAIL_ADDRESS>.
    """
    print(f"\n🤖 [LLM Side] Processing prompt: '{prompt}'")
    # The LLM logic: "If user gives email, say we sent a code to it."
    return f"I have sent a verification code to {prompt.split()[-1]}."

def main():
    print("🛡️ Starting Day 56: PII Round-Trip (Deanonymization)...")
    
    analyzer = AnalyzerEngine()
    anonymizer = AnonymizerEngine()

    # --- Step 1: User Input ---
    user_input = "Please verify my email: john.doe@example.com"
    print(f"\n👤 User Input: '{user_input}'")

    # --- Step 2: Redaction (Day 55 Logic) ---
    results = analyzer.analyze(text=user_input, entities=["EMAIL_ADDRESS"], language='en')
    
    # We use a custom operator to replace with a UNIQUE placeholder
    # instead of generic <EMAIL>, we use <EMAIL_1>, <EMAIL_2> to map them back later.
    
    # NOTE: Presidio's default Deanonymize is complex. 
    # For this learning task, we will do a manual "Dictionary Swap" approach
    # which is common in lightweight agents.
    
    redacted_text = user_input
    current_map = {}
    
    # Sort results reverse to replace without messing up indices
    for res in sorted(results, key=lambda x: x.start, reverse=True):
        original_value = user_input[res.start:res.end]
        placeholder = f"<{res.entity_type}_{str(uuid.uuid4())[:4]}>"
        
        # Replace in text
        redacted_text = redacted_text[:res.start] + placeholder + redacted_text[res.end:]
        
        # Store in Map
        current_map[placeholder] = original_value

    print(f"\n🔒 Redacted (Sent to LLM): '{redacted_text}'")
    print(f"   [System Memory]: {current_map}")

    # --- Step 3: LLM Processing ---
    # The LLM sees: "Please verify my email: <EMAIL_ADDRESS_a1b2>"
    llm_response = mock_llm_response(redacted_text)
    print(f"\n🤖 Raw LLM Response: '{llm_response}'")

    # --- Step 4: Deanonymization (The New Day 56 Task) ---
    # We swap the placeholders back to real values for the user
    final_response = llm_response
    for placeholder, original_value in current_map.items():
        final_response = final_response.replace(placeholder, original_value)

    print(f"\n✅ Final Response to User: '{final_response}'")

if __name__ == "__main__":
    main()