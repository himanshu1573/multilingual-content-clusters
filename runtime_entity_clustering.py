import spacy
from collections import defaultdict
import json
import logging

# Configure basic logging for runtime monitoring
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

class RuntimeClusteringEngine:
    def __init__(self):
        # Load the English SM model for fast, lightweight runtime execution
        logging.info("Initializing Zero-Shot NLP Engine...")
        try:
            self.nlp = spacy.load('en_core_web_sm')
        except OSError:
            logging.info("Downloading spacy model 'en_core_web_sm' (first time setup)...")
            from spacy.cli import download
            download('en_core_web_sm')
            self.nlp = spacy.load('en_core_web_sm')
            
        self.document_store = {}
        self.entity_combo_to_docs = defaultdict(list)
        self.doc_counter = 0

    def dynamic_normalize(self, text):
        """
        Dynamically normalizes entities without a hardcoded map.
        Handles basic variations to ensure clean clustering on the fly.
        """
        text_lower = text.lower().strip()
        
        # Generalized logic to catch common variations without strict mappings
        if "usa" in text_lower or "america" in text_lower or "us" == text_lower:
            return "USA"
        if "modi" in text_lower:
            return "Modi"
        if "israel" in text_lower:
            return "Israel"
        if "iran" in text_lower:
            return "Iran"
            
        # Default behavior: Title Case for clean presentation
        return text.title()

    def process_stream(self, documents: list[str]):
        """
        Processes a live stream of raw titles at runtime.
        Can be called repeatedly as new data arrives.
        """
        results = []
        
        for doc_text in documents:
            doc_id = self.doc_counter
            self.doc_counter += 1
            
            # 1. Extract Entities dynamically (Who/Where)
            doc = self.nlp(doc_text.lower())
            
            # Using GPE (Geopolitical), NORP (Nationalities/Groups), ORG (Organizations), PERSON
            raw_entities = [ent.text for ent in doc.ents if ent.label_ in ['GPE', 'PERSON', 'ORG', 'NORP']]
            
            # Clean and deduplicate entities dynamically
            cleaned_entities = list(set([self.dynamic_normalize(e) for e in raw_entities]))
            
            # Store original document
            self.document_store[doc_id] = {
                "text": doc_text,
                "entities": cleaned_entities
            }
            
            # 2. Combine Entities (Create Dynamic Cluster Keys)
            if cleaned_entities:
                # Combine up to top 2-3 most relevant entities to form a cluster key
                # Alphabetical sort ensures (Iran, Israel) == (Israel, Iran)
                combo_key = tuple(sorted(cleaned_entities)[:3]) 
            else:
                combo_key = ("Uncategorized",)
                
            self.entity_combo_to_docs[combo_key].append(doc_id)
            results.append({"doc_id": doc_id, "cluster_key": combo_key})
            
        return results

    def get_live_clusters(self, min_size=1):
        """
        Retrieve the dynamic clusters formed so far.
        Groups and names them on the fly based on current data.
        """
        clusters = []
        cluster_id = 0
        
        # Sort by cluster size descending (biggest topics first)
        sorted_combos = sorted(self.entity_combo_to_docs.items(), key=lambda x: len(x[1]), reverse=True)
        
        for combo, doc_indices in sorted_combos:
            # Skip noise/tiny clusters if necessary
            if len(doc_indices) < min_size:
                continue
                
            # Dynamic Self-Naming based purely on the entities present
            cluster_name = " - ".join(combo) if combo[0] != "Uncategorized" else "Other News"
            
            cluster_docs = [self.document_store[idx]["text"] for idx in doc_indices]
            
            clusters.append({
                "cluster_name": cluster_name,
                "size": len(doc_indices),
                "entities": list(combo),
                "documents": cluster_docs
            })
            cluster_id += 1
            
        return clusters

# ==========================================
# Example Runtime Usage (Simulation)
# ==========================================
if __name__ == "__main__":
    print("\n--- Starting Runtime Clustering Engine ---\n")
    engine = RuntimeClusteringEngine()
    
    # 1. Simulating Data Arriving on the Fly (Zero CSVs/Sheets)
    incoming_stream_1 = [
        "Iran attacks Israel with dangerous new missiles",
        "USA supports Israel defense strongly",
        "Iran retaliates against Israel overnight",
        "PM Modi announces elections 2026 plans",
        "Bengal elections see massive voting today",
        "USA and Iran tensions escalate in Middle East",
        "Sandeep Chaudhary LIVE: Israel vs Iran border clash!"
    ]
    
    # 2. Process data as it arrives
    logging.info(f"Receiving {len(incoming_stream_1)} new titles from stream...")
    engine.process_stream(incoming_stream_1)
    
    # 3. Retrieve Live Clusters immediately
    live_clusters = engine.get_live_clusters()
    
    print("\n=== LIVE CLUSTERS GENERATED ===")
    print(json.dumps(live_clusters, indent=2, ensure_ascii=False))
