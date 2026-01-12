import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
from dotenv import load_dotenv

from langchain.docstore.document import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

from ragas.testset.graph import KnowledgeGraph, Node, NodeType
from ragas.testset.persona import Persona
from ragas.testset.synthesizers.single_hop.specific import (
    SingleHopSpecificQuerySynthesizer,
)
from ragas.testset.transforms import apply_transforms, KeyphrasesExtractor
from ragas.testset import TestsetGenerator

from src.chunking import Chunking

# from entity_extractor import extract_entities

# Load environment variables from .env file

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

# Parameters
text_dir = "D:/AMI_Project/ami_group1/Dataset/Test/text"
max_words = 300
min_words = 100
testset_size = 20
output_path = "D:/AMI_Project/ami_group1/testset/testset6.json"

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

generator_llm = LangchainLLMWrapper(llm)

embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
generator_embeddings = LangchainEmbeddingsWrapper(embedding_model)


class MissionAwareSingleHopQuerySynthesizer(SingleHopSpecificQuerySynthesizer):
    def _format_query_prompt(self, node, persona):
        keyphrase = node.properties.get("keyphrases", [""])[0]
        return f"""
You are acting as a {persona.name} managing a wildfire response.

Using the following information extracted from a wildfire scenario:
"{keyphrase}"

Generate one **specific, realistic, and scenario-driven question** that:
- Reflects **tactical decision-making** in response to conditions like terrain, wind, slope, resources, fireline geometry, or visibility
- Is **not tied to any specific historical event, person, or place** (e.g., no “South Canyon” or “Ted Putnam”)
- Focuses on **real-time planning**, **response**, or **safety**, not learning history or biographies

Avoid:
- Biographical or historical summary questions
- Mentioning real names or locations
- General quiz-like or reading-comprehension questions

Good examples:
- “Given limited visibility and steep terrain on the east flank, should we continue direct attack or switch to indirect?”
- “What are the best drone deployment strategies under shifting wind conditions near the ridgeline?”

Only output the question. Do not include explanations or commentary.
        """.strip()


# 1. Chunking all text files
all_docs = []
for fname in os.listdir(text_dir):
    if fname.endswith(".txt"):
        filepath = os.path.join(text_dir, fname)
        chunker = Chunking(filepath, max_words, min_words)
        all_docs.extend(Document(page_content=chunk, metadata={"source": fname}) for chunk in chunker.chunks)

print(f"✅ Total chunks created: {len(all_docs)}")

# 2. Create knowledge graph
kg = KnowledgeGraph()

# Build graph
for doc in all_docs:
    text = doc.page_content
    meta = doc.metadata
    # extracted = extract_entities(text)

    # Add CHUNK node
    chunk_node = Node(
        type=NodeType.CHUNK,
        properties={
            "page_content": text,
            # "entities": extracted,
            "document_metadata": meta,
        },
    )
    kg.nodes.append(chunk_node)

print(f"✅ Total nodes in KG before transforms: {len(kg.nodes)}")

# 3. Setup Transforms
# headline_extractor = HeadlinesExtractor(llm=generator_llm, max_num=20)
# headline_splitter = HeadlineSplitter(max_tokens=1500)
keyphrase_extractor = KeyphrasesExtractor(llm=generator_llm)

transforms = [
    # headline_extractor,
    # headline_splitter,
    keyphrase_extractor
]

apply_transforms(kg, transforms=transforms)
print(f"✅ Total nodes in KG after transforms: {len(kg.nodes)}")
keyphrase_nodes = [n for n in kg.nodes if "keyphrases" in n.properties]
print(f"🧠 Keyphrase nodes found: {len(keyphrase_nodes)}")

# Example Personas
persona_list = [
    Persona(
        name="Junior Commander",
        role_description="Has various resources at your disposal — including drones, fire trucks, and other firefighting equipment — to extinguish a fire.",
    ),
]

query_distibution = [
    (
        MissionAwareSingleHopQuerySynthesizer(llm=generator_llm, property_name="keyphrases"),
        1,
    ),
]

# 4. Generate testset
generator = TestsetGenerator(
    llm=generator_llm,
    embedding_model=generator_embeddings,
    knowledge_graph=kg,
    persona_list=persona_list,
)

testset = generator.generate(testset_size=10, query_distribution=query_distibution, with_debugging_logs=True)

# Save to file
testset.to_pandas()
df = testset.to_pandas()
# Save as nicely formatted JSON
df.to_json(
    "D:/AMI_Project/ami_group1/testset/testset6.json",
    orient="records",
    indent=2,
    force_ascii=False,
)
