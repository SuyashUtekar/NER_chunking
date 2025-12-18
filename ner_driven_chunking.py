
# ner_driven_chunking.py
# End-to-end NER-driven chunking implementation
# NER: Rule-based + Dictionary-based + GLiNER (Transformer)
# Coreference: fastcoref
# Chunking: Entity-centric (sentence can belong to multiple entities)

import re
import nltk
from nltk.tokenize import sent_tokenize
from gliner import GLiNER
from fastcoref import FCoref

# -----------------------
# Setup
# -----------------------
nltk.download("punkt")

# -----------------------
# Input Text
# -----------------------
TEXT = """
State Bank of India announced a strategic partnership with Microsoft to modernize its digital banking platform.
SBI said it will use cloud infrastructure to improve customer experience.
The bank believes this collaboration will help it scale services across India.

Microsoft confirmed that it will provide Azure-based solutions and AI tools.
The tech giant emphasized that security and scalability are priorities.

HDFC Bank reported strong quarterly profits.
The bank expects growth in retail lending.

Infosys announced that it will support the integration process.
The company has previously worked with SBI.
"""

# -----------------------
# Sentence Segmentation
# -----------------------
sentences = sent_tokenize(TEXT)

# -----------------------
# Rule-based NER
# -----------------------
def rule_based_ner(text):
    patterns = {
        "ORG": [
            r"\bState Bank of India\b",
            r"\bHDFC Bank\b"
        ]
    }
    entities = []
    for label, regs in patterns.items():
        for r in regs:
            for m in re.finditer(r, text):
                entities.append({"text": m.group(), "label": label})
    return entities

# -----------------------
# Dictionary-based NER
# -----------------------
NER_DICTIONARY = {
    "SBI": "State Bank of India",
    "Microsoft": "Microsoft",
    "Infosys": "Infosys",
    "Azure": "Azure"
}

def dictionary_ner(text):
    entities = []
    for k in NER_DICTIONARY:
        if k in text:
            entities.append({"text": k, "label": "ORG"})
    return entities

# -----------------------
# Transformer-based NER (GLiNER)
# -----------------------
gliner = GLiNER.from_pretrained("urchade/gliner_base")

def gliner_ner(text):
    labels = ["organization", "location", "product", "person"]
    return gliner.predict_entities(text, labels)

# -----------------------
# Combine NER Mentions
# -----------------------
ner_mentions = []
ner_mentions.extend(rule_based_ner(TEXT))
ner_mentions.extend(dictionary_ner(TEXT))
ner_mentions.extend(gliner_ner(TEXT))

# -----------------------
# Normalize Entity Names
# -----------------------
def normalize(name):
    name = name.lower()
    name = re.sub(r"^(the|a|an)\s+", "", name)
    return name.strip()

canonical_map = {}
for ent in ner_mentions:
    norm = normalize(ent["text"])
    canonical_map.setdefault(norm, ent["text"])

canonical_entities = list(set(canonical_map.values()))

# -----------------------
# Coreference Resolution (fastcoref)
# -----------------------
coref_model = FCoref(device="cpu")
preds = coref_model.predict(texts=[TEXT])
coref_clusters = preds[0].get_clusters()

# -----------------------
# Build Final Entity Set
# -----------------------
final_entities = {}

def add_mention(entity, mention):
    final_entities.setdefault(entity, set()).add(mention)

# Add canonical entities
for ent in canonical_entities:
    add_mention(ent, ent)

# Attach coreference mentions (NER-anchored)
for cluster in coref_clusters:
    anchor = None
    for m in cluster:
        if normalize(m) in canonical_map:
            anchor = canonical_map[normalize(m)]
            break
    if anchor:
        for m in cluster:
            add_mention(anchor, m)

# -----------------------
# Sentence-to-Entity Mapping (Multi-entity allowed)
# -----------------------
sentence_entity_map = {i: set() for i in range(len(sentences))}

for i, sent in enumerate(sentences):
    sent_lower = sent.lower()
    for entity, mentions in final_entities.items():
        for m in mentions:
            if m.lower() in sent_lower:
                sentence_entity_map[i].add(entity)

# -----------------------
# Entity-centric Chunking
# -----------------------
entity_chunks = {}

for idx, entities in sentence_entity_map.items():
    for e in entities:
        entity_chunks.setdefault(e, []).append(sentences[idx])

# -----------------------
# Output
# -----------------------
print("\n===== ENTITY-CENTRIC CHUNKS =====")
for entity, chunks in entity_chunks.items():
    print(f"\nENTITY: {entity}")
    for c in chunks:
        print("-", c)
