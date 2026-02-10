import sys
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

def main():
    print("🛡️ Initializing PII Redaction Agent (Day 55)...")
    
    # 1. Initialize Engines
    # Analyzer: Finds the PII (The "Eyes")
    # Anonymizer: Replaces the PII (The "Hands")
    analyzer = AnalyzerEngine()
    anonymizer = AnonymizerEngine()

    # 2. Simulate User Input containing sensitive data
    user_query = (
        "Hi, I'm John Doe. My email is john.doe@example.com "
        "and my phone number is 555-0199. Please reset my password."
    )
    
    print(f"\n📥 Raw Input:\n'{user_query}'")

    # 3. ANALYZE: Detect PII entities
    # We ask it to look for specific sensitive types
    results = analyzer.analyze(
        text=user_query,
        entities=["PERSON", "PHONE_NUMBER", "EMAIL_ADDRESS"],
        language='en'
    )

    # 4. ANONYMIZE: Replace detected entities
    # We configure it to replace data with clear tags like <EMAIL_ADDRESS>
    anonymized_result = anonymizer.anonymize(
        text=user_query,
        analyzer_results=results,
        operators={
            "DEFAULT": OperatorConfig("replace", {"new_value": "<REDACTED>"}),
            "PHONE_NUMBER": OperatorConfig("replace", {"new_value": "<PHONE_NUMBER>"}),
            "EMAIL_ADDRESS": OperatorConfig("replace", {"new_value": "<EMAIL_ADDRESS>"}),
            "PERSON": OperatorConfig("replace", {"new_value": "<PERSON_NAME>"}),
        }
    )

    # 5. Output the Safe String
    # This is what you would actually send to OpenAI/GPT-4
    safe_text = anonymized_result.text
    print(f"\n📤 Sanitized Output (Ready for LLM):\n'{safe_text}'")
    
    # 6. Verification (Optional)
    print("\n🔍 Detection Details:")
    for res in results:
        print(f"   - Found {res.entity_type} with confidence {res.score:.2f}")

if __name__ == "__main__":
    main()