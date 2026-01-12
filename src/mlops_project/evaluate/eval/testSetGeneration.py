import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import asyncio
from dotenv import load_dotenv
from datetime import datetime

from langchain.docstore.document import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

from ragas.testset.graph import KnowledgeGraph, Node, NodeType
from ragas.testset.persona import Persona
from ragas.testset.synthesizers.multi_hop.specific import (
    MultiHopSpecificQuerySynthesizer,
)
from ragas.testset import TestsetGenerator
from ragas.testset.transforms.relationship_builders.traditional import (
    JaccardSimilarityBuilder,
)

from src.chunking import Chunking
from entity_extractor_copy import extract_entities

# from collections import Counter

# Load environment variables from .env file

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

# Parameters
text_dir = "D:/AMI_Project/ami_group1/Dataset/Test/text"
max_words = 300
min_words = 100
testset_size = 2
output_path = "D:/AMI_Project/ami_group1/testset/testset.json"

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

# 1. Chunking all text files
all_docs = []
for fname in os.listdir(text_dir):
    if fname.endswith(".txt"):
        filepath = os.path.join(text_dir, fname)
        chunker = Chunking(filepath, max_words, min_words)
        all_docs.extend(Document(page_content=chunk, metadata={"source": fname}) for chunk in chunker.chunks)

# 2. Create knowledge graph
kg = KnowledgeGraph()


# Avoid duplicate concept nodes
def is_duplicate(node: Node, existing_nodes: list[Node]) -> bool:
    return any(n.properties.get("name") == node.properties.get("name") and n.properties.get("category") == node.properties.get("category") for n in existing_nodes)


# Build graph
for doc in all_docs:
    text = doc.page_content
    meta = doc.metadata
    extracted = extract_entities(text)

    # Skip chunk if no entities
    if not any(extracted.get(k) for k in ["waypoints", "actions", "resources"]):
        continue

    # Add CHUNK node
    chunk_node = Node(
        type=NodeType.CHUNK,
        properties={
            "text": text,
            "entities": extracted,
            "document_metadata": meta,
        },
    )
    kg.nodes.append(chunk_node)

    # Uncomment when better data avaliable
    # Add WAYPOINT nodes
    for wp in extracted.get("waypoints", []):
        node = Node(type=NodeType.DOCUMENT, properties={"name": wp, "category": "waypoint"})
        if not is_duplicate(node, kg.nodes):
            kg.nodes.append(node)

    # Add ACTION nodes
    for act in extracted.get("actions", []):
        node = Node(type=NodeType.DOCUMENT, properties={"name": act, "category": "action"})
        if not is_duplicate(node, kg.nodes):
            kg.nodes.append(node)

    # Add RESOURCE nodes
    for res in extracted.get("resources", []):
        node = Node(type=NodeType.DOCUMENT, properties={"name": res, "category": "resource"})
        if not is_duplicate(node, kg.nodes):
            kg.nodes.append(node)

# type_counter = Counter(node.properties.get("category", "chunk") for node in kg.nodes)
# print("Node category breakdown:", type_counter)


# 3. Build relationships with JaccardSimilarity
# for node in kg.nodes:
#     if node.properties.get("category") in ["waypoint", "action", "resource"]:
#         print(f"{node.properties['category']}: {node.properties['name']}")


async def build_relationships(graph: KnowledgeGraph):
    print("Building relationships...")

    def has_entities(node: Node) -> bool:
        if node.type != NodeType.CHUNK:
            return False
        entities = node.properties.get("entities", {})
        return any(entities.get(key) for key in ["waypoints", "actions", "resources"])

    kg.nodes = [node for node in kg.nodes if node.type != NodeType.CHUNK or has_entities(node)]

    rel_builder = JaccardSimilarityBuilder(filter_nodes=has_entities)

    relationships = await rel_builder.transform(graph)
    graph.relationships.extend(relationships)

    print(f"Total relationships added: {len(graph.relationships)}")
    for rel in kg.relationships[:5]:
        print(f"Source: {rel.source_id}, Target: {rel.target_id}, Type: {rel.type}, Properties: {rel.properties}")


asyncio.run(build_relationships(kg))


# 4. Generate testset
async def generate_testset():
    await build_relationships(kg)

    # Example Personas
    persona_list = [
        Persona(
            name="Junior Commander",
            role_description="Has various resources at your disposal — including drones, fire trucks, and other firefighting equipment — to extinguish a fire.",
        ),
    ]

    # Scenario Synthesizer
    synthesizer = MultiHopSpecificQuerySynthesizer(llm=generator_llm)
    synthesizer.relation_type = "jaccard_similarity"
    scenarios = await synthesizer._generate_scenarios(
        n=testset_size,
        knowledge_graph=kg,
        persona_list=persona_list,
        callbacks=[],
    )

    # Testset Generator
    testset_generator = TestsetGenerator(
        llm=generator_llm,
        embedding=generator_embeddings,
    )

    testset = await testset_generator.generate(scenarios)
    print(f"Testset generated with {len(testset)} items.")

    # Save to file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(testset, f, indent=2, ensure_ascii=False)
    print(f"Testset saved to {output_path}")


if __name__ == "__main__":
    asyncio.run(generate_testset())
