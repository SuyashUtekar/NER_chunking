import torch
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("biu-nlp/f-coref")
model = AutoModel.from_pretrained("biu-nlp/f-coref")
model.eval()

def run_coref(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    with torch.no_grad():
        outputs = model(**inputs)

    return outputs

def extract_coref_clusters(text, outputs):
    input_ids = tokenizer(text)["input_ids"]
    clusters = []

    if "clusters" not in outputs:
        return clusters

    for cluster in outputs["clusters"]:
        mentions = []
        for start, end in cluster:
            mention = tokenizer.decode(
                input_ids[start:end+1],
                skip_special_tokens=True
            )
            mentions.append(mention)
        clusters.append(mentions)

    return clusters

text = """
State Bank of India reported profits.
SBI said it expects growth.
The bank believes this will help it expand.
"""

outputs = run_coref(text)
coref_clusters = extract_coref_clusters(text, outputs)
coref_clusters



def filter_coref_clusters(coref_clusters, ner_entities):
    valid_clusters = []

    for cluster in coref_clusters:
        if any(m in ner_entities for m in cluster):
            valid_clusters.append(cluster)

    return valid_clusters

final_entities = {}

def add_mention(entity, mention):
    if entity not in final_entities:
        final_entities[entity] = set()
    final_entities[entity].add(mention)

# Step 1: add NER entities
for ent in ner_entities:
    add_mention(ent, ent)

# Step 2: attach coref mentions
filtered_clusters = filter_coref_clusters(coref_clusters, ner_entities)

for cluster in filtered_clusters:
    canonical = next(m for m in cluster if m in ner_entities)
    for mention in cluster:
        add_mention(canonical, mention)

for entity, mentions in final_entities.items():
    print(entity, "→", list(mentions))





