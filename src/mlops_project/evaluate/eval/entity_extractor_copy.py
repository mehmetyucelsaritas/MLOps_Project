import json
from langchain_core.prompts import PromptTemplate
import os
import sys
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
# Initialize LLM and embedding
config = {
    "model": "gemini-1.5-pro",
    "temperature": 0.4,
    "max_tokens": None,
    "top_p": 0.8,
}

llm = ChatGoogleGenerativeAI(
    model=config["model"],
    temperature=config["temperature"],
    max_tokens=config["max_tokens"],
    top_p=config["top_p"],
    google_api_key=api_key,
)

# Build the prompt
entity_prompt = PromptTemplate.from_template("""
You are an expert information extractor working on firefighting scenario documents.

Extract the following three categories from the input text:
1. waypoints — locations like ridges, zones, hills, flanks, etc.
2. actions — things that can be done (e.g., deploy resources, suppress fire)
3. resources — things like drones, trucks, equipment, crews

Format your response as JSON with three fields: "waypoints", "actions", "resources".
Each should be a list of strings.

Text:
{text}
""")

# Chain it with your Gemini LLM
llm_entity_extractor = entity_prompt | llm  # llm = your ChatGoogleGenerativeAI object


def extract_entities(text: str) -> dict:
    try:
        # Call Gemini LLM through LangChain
        response = llm_entity_extractor.invoke({"text": text})
        time.sleep(0.05)  # Add 500ms delay to avoid flooding API

        # Extract content from LangChain's AIMessage
        if hasattr(response, "content"):
            response_text = response.content.strip()
        else:
            response_text = str(response).strip()

        # Remove Markdown code block formatting (```json ... ```)
        if response_text.startswith("```json"):
            response_text = response_text.removeprefix("```json").strip()
        if response_text.startswith("```"):
            response_text = response_text.removeprefix("```").strip()
        if response_text.endswith("```"):
            response_text = response_text.removesuffix("```").strip()

        # Empty response check
        if not response_text:
            print("[LLM Error] Empty response for input:")
            print(text[:300])
            return {"waypoints": [], "actions": [], "resources": []}

        # Attempt to parse cleaned response
        try:
            entities = json.loads(response_text)
        except json.JSONDecodeError:
            print(f"[LLM JSON Error] Could not parse:\n{response_text}")
            return {"waypoints": [], "actions": [], "resources": []}

        # Structure check
        if not isinstance(entities, dict):
            print(f"[LLM Warning] Unexpected structure:\n{entities}")
            return {"waypoints": [], "actions": [], "resources": []}

    except Exception as e:
        print(f"[Entity Extraction Error] {e}")
        return {"waypoints": [], "actions": [], "resources": []}

    # Cleanup: deduplicate, trim, and filter long/invalid strings
    def clean(entity_list):
        return list(set([e.strip() for e in entity_list if isinstance(e, str) and 0 < len(e.strip()) <= 40]))

    return {
        "waypoints": clean(entities.get("waypoints", [])),
        "actions": clean(entities.get("actions", [])),
        "resources": clean(entities.get("resources", [])),
    }
