# ============================================================
# Generalized Entity (NER + Coref) Driven Chunking Service
# GLiNER + fastcoref
# ============================================================

# ---------------------------
# Standard Library
# ---------------------------
import re
import unicodedata
from typing import List, Dict
from collections import defaultdict, Counter

# ---------------------------
# PDF Processing
# ---------------------------
import pdfplumber

# ---------------------------
# NLP
# ---------------------------
import spacy
from gliner import GLiNER
from fastcoref import FCoref
import torch

# ---------------------------
# Fuzzy Matching
# ---------------------------
from rapidfuzz import fuzz


# ============================================================
# STEP 1 — PDF TEXT EXTRACTION
# ============================================================

def extract_text_from_page(page):
    tables = page.find_tables()
    table_bboxes = [t.bbox for t in tables]

    def inside_table(char):
        for x0, top, x1, bottom in table_bboxes:
            if x0 <= char["x0"] <= x1 and top <= char["top"] <= bottom:
                return True
        return False

    chars = [c for c in page.chars if not inside_table(c)]
    if not chars:
        return []

    text = pdfplumber.utils.extract_text(chars)
    if not text:
        return []

    return [line.strip() for line in text.split("\n") if line.strip()]


def detect_repeated_lines(pages_lines, threshold=0.7):
    counter = Counter()
    total_pages = len(pages_lines)

    for lines in pages_lines:
        counter.update(set(lines))

    return {
        line for line, count in counter.items()
        if count / total_pages >= threshold
    }


def extract_clean_text_from_pdf(pdf_path: str) -> str:
    pages_lines = []

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            pages_lines.append(extract_text_from_page(page))

    repeated = detect_repeated_lines(pages_lines)

    cleaned_pages = []
    for lines in pages_lines:
        cleaned_pages.append(" ".join(l for l in lines if l not in repeated))

    return " ".join(cleaned_pages)


# ============================================================
# STEP 2 — CLEANING & PREPROCESSING (NO \n)
# ============================================================

def normalize_unicode(text):
    return unicodedata.normalize("NFKC", text)


def fix_hyphenated_line_breaks(text):
    return re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)


def clean_text(text: str) -> str:
    text = normalize_unicode(text)
    text = fix_hyphenated_line_breaks(text)
    text = "".join(ch if unicodedata.category(ch)[0] != "C" else " " for ch in text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ============================================================
# STEP 3 — SENTENCE SEGMENTATION
# ============================================================

def load_sentence_segmenter():
    nlp = spacy.load("en_core_web_sm", disable=["ner", "parser", "tagger", "lemmatizer"])
    nlp.add_pipe("sentencizer")
    return nlp


def segment_sentences(text: str, nlp) -> List[Dict]:
    doc = nlp(text)
    return [
        {
            "index": i,
            "sentence_text": sent.text,
            "start_char": sent.start_char,
            "end_char": sent.end_char
        }
        for i, sent in enumerate(doc.sents)
    ]


# ============================================================
# STEP 4 — GLiNER NER
# ============================================================

def load_gliner():
    return GLiNER.from_pretrained(
        "gliner-community/gliner_medium-v2.5",
        load_tokenizer=True
    )


def extract_entities_with_gliner(sentences, model, labels, threshold=0.5):
    entities = []
    for s in sentences:
        preds = model.predict_entities(s["sentence_text"], labels, threshold=threshold)
        for p in preds:
            entities.append({
                "entity_text": p["text"],
                "label": p["label"],
                "confidence": p["score"],
                "sentence_index": s["index"],
                "start_char": s["start_char"] + p["start"],
                "end_char": s["start_char"] + p["end"]
            })
    return entities


# ============================================================
# STEP 5 — NORMALIZATION & DEDUP
# ============================================================

ARTICLE_RE = re.compile(r"^(a|an|the)\s+", re.I)


def normalize_entity_text(text: str) -> str:
    text = text.lower().strip()
    text = ARTICLE_RE.sub("", text)
    text = re.sub(r"^[^\w]+|[^\w]+$", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def normalize_entities(entities):
    for e in entities:
        e["normalized_text"] = normalize_entity_text(e["entity_text"])
    return entities


def build_canonical_entity_map(entities):
    cmap = defaultdict(lambda: {
        "canonical_text": None,
        "labels": set(),
        "surface_forms": set(),
        "mentions": []
    })

    for e in entities:
        key = e["normalized_text"]
        entry = cmap[key]
        if entry["canonical_text"] is None or len(e["entity_text"]) > len(entry["canonical_text"]):
            entry["canonical_text"] = e["entity_text"]
        entry["labels"].add(e["label"])
        entry["surface_forms"].add(e["entity_text"])
        entry["mentions"].append(e)

    return cmap


# ============================================================
# STEP 6 — ACRONYM RESOLUTION
# ============================================================

ACRONYM_RE = re.compile(r"([A-Za-z][A-Za-z\s]+)\s*\(([A-Z]{2,})\)|([A-Z]{2,})\s*\(([A-Za-z][A-Za-z\s]+)\)")


def resolve_acronyms(text, cmap):
    acr_map = {}
    for m in ACRONYM_RE.finditer(text):
        full = m.group(1) or m.group(4)
        acr = m.group(2) or m.group(3)
        n_full = normalize_entity_text(full)
        n_acr = normalize_entity_text(acr)
        if n_full in cmap:
            acr_map[n_acr] = n_full
            cmap[n_full]["surface_forms"].add(acr)
    return acr_map


# ============================================================
# STEP 7 — ALIAS RESOLUTION
# ============================================================

def resolve_aliases(cmap, entities, threshold=92):
    alias_map = {}
    keys = list(cmap.keys())

    for e in entities:
        alias = e["normalized_text"]
        if alias in cmap:
            continue
        matches = []
        for k in keys:
            if fuzz.ratio(alias, k) >= threshold:
                matches.append(k)
        if len(matches) == 1:
            alias_map[alias] = matches[0]
            cmap[matches[0]]["surface_forms"].add(e["entity_text"])
    return alias_map


# ============================================================
# STEP 8 — FINAL CANONICAL REGISTRY
# ============================================================

def build_final_registry(cmap, acr_map, alias_map):
    registry = {}
    surface_to_canonical = {}

    for i, (k, v) in enumerate(cmap.items()):
        cid = f"ENT_{i:04d}"
        registry[cid] = {
            "canonical_text": v["canonical_text"],
            "normalized_key": k,
            "surface_forms": set(v["surface_forms"]),
            "mentions": v["mentions"]
        }
        for sf in v["surface_forms"]:
            surface_to_canonical[normalize_entity_text(sf)] = cid

    for a, k in {**acr_map, **alias_map}.items():
        for cid, ent in registry.items():
            if ent["normalized_key"] == k:
                surface_to_canonical[a] = cid

    return registry, surface_to_canonical


# ============================================================
# STEP 9 — COREFERENCE
# ============================================================

def load_coref():
    return FCoref(
        model_name="biu-nlp/f-coref",
        device="cuda" if torch.cuda.is_available() else "cpu"
    )


def run_coref(text, model):
    pred = model.predict(texts=[text])[0]
    clusters = []
    for i, cl in enumerate(pred["clusters"]):
        mentions = []
        for s, e in cl:
            mentions.append({
                "text": text[s:e],
                "start_char": s,
                "end_char": e,
                "normalized_text": normalize_entity_text(text[s:e])
            })
        clusters.append({"cluster_id": i, "mentions": mentions})
    return clusters


# ============================================================
# STEP 10–15 — ENTITY-CENTRIC CHUNKING
# ============================================================

def build_final_chunks(
    sentences,
    entities,
    registry,
    surface_to_canonical,
    coref_clusters
):
    sentence_map = {s["index"]: s["sentence_text"] for s in sentences}
    entity_chunks = defaultdict(lambda: {
        "mentions": set(),
        "sentence_indexes": set(),
        "sentences": []
    })

    # Coref attachment
    for cl in coref_clusters:
        for m in cl["mentions"]:
            norm = m["normalized_text"]
            if norm in surface_to_canonical:
                cid = surface_to_canonical[norm]
                sent_idx = next(
                    s["index"] for s in sentences
                    if s["start_char"] <= m["start_char"] < s["end_char"]
                )
                entity_chunks[cid]["mentions"].add(m["text"])
                entity_chunks[cid]["sentence_indexes"].add(sent_idx)

    # NER-only attachment
    for e in entities:
        cid = surface_to_canonical.get(e["normalized_text"])
        if cid:
            entity_chunks[cid]["mentions"].add(e["entity_text"])
            entity_chunks[cid]["sentence_indexes"].add(e["sentence_index"])

    # Final formatting
    final = {}
    for cid, data in entity_chunks.items():
        idxs = sorted(data["sentence_indexes"])
        final[registry[cid]["canonical_text"]] = {
            "total_mentions": list(data["mentions"]),
            "sentences_text": [sentence_map[i] for i in idxs],
            "sentences_indexes": idxs
        }

    return final


# ============================================================
# PUBLIC SERVICE ENTRYPOINT
# ============================================================

def run_entity_driven_chunking(pdf_path: str, gliner_labels: List[str]) -> Dict:
    raw_text = extract_clean_text_from_pdf(pdf_path)
    cleaned = clean_text(raw_text)

    nlp = load_sentence_segmenter()
    sentences = segment_sentences(cleaned, nlp)

    gliner = load_gliner()
    ner_entities = extract_entities_with_gliner(sentences, gliner, gliner_labels)
    ner_entities = normalize_entities(ner_entities)

    cmap = build_canonical_entity_map(ner_entities)
    acr_map = resolve_acronyms(cleaned, cmap)
    alias_map = resolve_aliases(cmap, ner_entities)

    registry, surface_map = build_final_registry(cmap, acr_map, alias_map)

    coref_model = load_coref()
    coref_clusters = run_coref(cleaned, coref_model)

    return build_final_chunks(
        sentences,
        ner_entities,
        registry,
        surface_map,
        coref_clusters
    )
