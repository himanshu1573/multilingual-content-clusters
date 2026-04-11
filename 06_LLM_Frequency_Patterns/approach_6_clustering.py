import pandas as pd
import collections
import itertools
import json
import os
import re
import requests
from pathlib import Path
from tqdm import tqdm

class Approach6Clustering:
    def __init__(self, ner_csv_path, output_dir="outputs/approach_6"):
        self.ner_csv_path = Path(ner_csv_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.df = None
        self.top_combinations = []
        self.cluster_definitions = {}

    def load_data(self):
        print(f"Loading NER data from {self.ner_csv_path}...")
        self.df = pd.read_csv(self.ner_csv_path)
        # Fill NaN values
        self.df[['Persons', 'Organizations', 'Locations']] = self.df[['Persons', 'Organizations', 'Locations']].fillna("")
        return self.df

    def get_entities_from_row(self, row):
        """Extracts all entities into a flat list."""
        entities = []
        for col in ['Persons', 'Organizations', 'Locations']:
            if row[col]:
                # Split by comma and strip
                parts = [p.strip() for p in str(row[col]).split(',') if p.strip()]
                entities.extend(parts)
        return list(set(entities))

    def find_frequent_combinations(self, top_n=100, min_freq=3):
        print("Analyzing entity combinations...")
        combination_counts = collections.Counter()
        
        for _, row in self.df.iterrows():
            entities = self.get_entities_from_row(row)
            if not entities:
                continue
            
            # Generate 1, 2, and 3-entity combinations
            for r in range(1, 4):
                if len(entities) >= r:
                    for combo in itertools.combinations(sorted(entities), r):
                        combination_counts[combo] += 1
        
        # Filter and sort
        sorted_combos = [
            {"entities": combo, "count": count}
            for combo, count in combination_counts.items()
            if count >= min_freq
        ]
        sorted_combos.sort(key=lambda x: x['count'], reverse=True)
        
        self.top_combinations = sorted_combos[:top_n]
        print(f"Found {len(self.top_combinations)} frequent combinations.")
        return self.top_combinations

    def generate_llm_definitions(self, gemini_api_key=None):
        """
        In a real scenario, this would call Gemini. 
        For this script, we provide a placeholder or a way to inject results.
        """
        if not gemini_api_key:
            print("Warning: No Gemini API key provided. Skipping LLM naming.")
            return

        import google.generativeai as genai
        genai.configure(api_key=gemini_api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')

        print("Generating cluster names via LLM...")
        for combo_data in tqdm(self.top_combinations):
            combo = combo_data['entities']
            count = combo_data['count']
            
            prompt = f"""
            Identify a professional news cluster name for these entities: {', '.join(combo)}
            This pattern appears in {count} articles.
            
            Return ONLY a valid JSON object:
            {{
                "topic_name": "Short, clear cluster name",
                "related_entities": ["list", "of", "5", "related", "entities"],
                "context": "1-sentence context"
            }}
            """
            
            try:
                response = model.generate_content(prompt)
                # Extract JSON from potential markdown blocks
                text = response.text
                match = re.search(r'\{.*\}', text, re.DOTALL)
                if match:
                    definition = json.loads(match.group())
                    self.cluster_definitions[tuple(combo)] = definition
                else:
                    self.cluster_definitions[tuple(combo)] = {
                        "topic_name": " / ".join(combo),
                        "related_entities": [],
                        "context": "Auto-generated from entities"
                    }
            except Exception as e:
                print(f"Error calling LLM for {combo}: {e}")
                self.cluster_definitions[tuple(combo)] = {
                    "topic_name": " / ".join(combo),
                    "related_entities": [],
                    "context": "Error during LLM call"
                }

    def generate_ollama_definitions(self, model_name="qwen2.5:3b", host="http://localhost:11434"):
        """
        Uses local Ollama instance for cluster naming and validation.
        """
        print(f"Generating cluster names via Ollama ({model_name})...")
        
        for combo_data in tqdm(self.top_combinations):
            combo = combo_data['entities']
            count = combo_data['count']
            
            prompt = f"""
            Identify a professional news cluster name for these entities: {', '.join(combo)}
            This pattern appears in {count} articles.
            
            Return ONLY a valid JSON object:
            {{
                "topic_name": "Short, clear cluster name",
                "related_entities": ["list", "of", "5", "related", "entities"],
                "context": "1-sentence context"
            }}
            """
            
            try:
                response = requests.post(
                    f"{host}/api/generate",
                    json={
                        "model": model_name,
                        "prompt": prompt,
                        "stream": False,
                        "format": "json"
                    },
                    timeout=120
                )
                response.raise_for_status()
                result = response.json()
                definition = json.loads(result['response'])
                self.cluster_definitions[tuple(combo)] = definition
            except Exception as e:
                print(f"Error calling Ollama for {combo}: {e}")
                self.cluster_definitions[tuple(combo)] = {
                    "topic_name": " / ".join(combo),
                    "related_entities": [],
                    "context": "Auto-generated from entities (Ollama Fallback)"
                }

    def tag_documents(self):
        print("Tagging documents...")
        results = []
        
        for _, row in self.df.iterrows():
            doc_entities = set(self.get_entities_from_row(row))
            best_match = None
            max_score = 0
            
            for combo, definition in self.cluster_definitions.items():
                combo_set = set(combo)
                # Check if the document contains ALL entities in the combination
                if combo_set.issubset(doc_entities):
                    # Score is based on combo size + related entity overlap
                    related_entities = set(definition.get('related_entities', []))
                    overlap = len(doc_entities & related_entities)
                    score = len(combo_set) * 10 + overlap
                    
                    if score > max_score:
                        max_score = score
                        best_match = definition['topic_name']
            
            results.append(best_match if best_match else "Other")
        
        self.df['cluster_approach_6'] = results
        return self.df

    def save_results(self):
        output_path = self.output_dir / "approach_6_clustered_results.csv"
        self.df.to_csv(output_path, index=False)
        
        # Save cluster definitions for reference
        definitions_path = self.output_dir / "cluster_definitions.json"
        with open(definitions_path, 'w') as f:
            # Convert tuple keys to strings for JSON
            json_defs = {", ".join(k): v for k, v in self.cluster_definitions.items()}
            json.dump(json_defs, f, indent=4)
            
        print(f"Results saved to {self.output_dir}")

if __name__ == "__main__":
    # 1. Setup paths
    base_dir = Path(__file__).parent.parent
    ner_data = base_dir / "02_Pure_NER" / "Video_Titles_NER.csv"
    output_dir = base_dir / "outputs" / "approach_6"
    
    if not ner_data.exists():
        print(f"Error: Could not find NER data at {ner_data}")
        # Fallback to local file if it exists
        if Path("Video_Titles_NER.csv").exists():
            ner_data = Path("Video_Titles_NER.csv")
            print(f"Using local file instead: {ner_data}")
        else:
            exit(1)

    # 2. Initialize clusterer
    clusterer = Approach6Clustering(ner_data, output_dir=output_dir)
    clusterer.load_data()
    
    # 3. Find patterns
    clusterer.find_frequent_combinations(top_n=50, min_freq=3)
    
    # 4. Generate Definitions (Ollama preferred as local fallback)
    api_key = os.getenv("GEMINI_API_KEY")
    if api_key:
        print("Using Gemini API...")
        clusterer.generate_llm_definitions(gemini_api_key=api_key)
    else:
        print("Gemini API key not found. Switching to local Ollama...")
        clusterer.generate_ollama_definitions()
    
    # 5. Tag and Save
    clusterer.tag_documents()
    clusterer.save_results()
    
    print("\n--- Pipeline Execution Complete ---")
    print(f"Final clustered data: {output_dir / 'approach_6_clustered_results.csv'}")
    print(f"Cluster definitions: {output_dir / 'cluster_definitions.json'}")
