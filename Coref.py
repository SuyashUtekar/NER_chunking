from fastcoref import FCoref
import spacy

# Load the coref model and spacy for sentence splitting
model = FCoref(device='cpu') 
nlp = spacy.load("en_core_web_sm")

text = (
    "Satya Nadella took over as CEO in 2014. Under his leadership, the company shifted "
    "toward cloud computing. He is often credited with transforming the culture. "
    "Meanwhile, Sundar Pichai led Google through various AI transitions. "
    "The Microsoft leader has written several books on empathy in business."
)

# 1. Predict Clusters
preds = model.predict(texts=[text])
clusters_strings = preds[0].get_clusters(string=True)
clusters_indices = preds[0].get_clusters()

# 2. Identify the target cluster for "Satya Nadella"
target_name = "Satya Nadella"
selected_cluster_indices = []

for i, cluster in enumerate(clusters_strings):
    # Check if our target name appears in this cluster
    if any(target_name.lower() in mention.lower() for mention in cluster):
        selected_cluster_indices = clusters_indices[i]
        break

# 3. Filter sentences based on the cluster indices
doc = nlp(text)
extracted_sentences = []

for sent in doc.sents:
    # Check if any mention in the cluster falls within this sentence's character range
    for m_start, m_end in selected_cluster_indices:
        if sent.start_char <= m_start < sent.end_char:
            extracted_sentences.append(sent.text)
            break

# Output the results
print(f"--- Entity Chunks for {target_name} ---")
for i, sentence in enumerate(extracted_sentences, 1):
    print(f"{i}. {sentence}")
