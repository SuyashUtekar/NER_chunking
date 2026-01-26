from entity_driven_chunking_service import run_entity_driven_chunking

labels = ["person", "organization", "location", "date", "event", "award"]

chunks = run_entity_driven_chunking("/content/sample.pdf", labels)

print(chunks)
