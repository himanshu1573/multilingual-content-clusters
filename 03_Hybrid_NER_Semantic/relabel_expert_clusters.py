import json
import requests
from pathlib import Path

def generate_label(keywords, docs):
    prompt = f"""You are a professional news analyst. Create a clear, specific, and concise label for a news topic.
Format your answer according to this formula: Entity/Main Concept + Action/Detail (Context in parentheses)
Examples:
- USA Military Rescue Operation (Iran, 2026)
- Expert Commentary - Sandeep Chaudhary (Iran-Israel)
- Trump Ceasefire Negotiations (Iran-Israel)
- West Bengal Election Campaign 2026

Do not use quotes or introductory phrases. Return ONLY the label.

Keywords: {', '.join(keywords)}
Sample documents:
1. {docs[0]}
2. {docs[1] if len(docs) > 1 else ''}
3. {docs[2] if len(docs) > 2 else ''}

Label:"""
    
    try:
        response = requests.post(
            "http://127.0.0.1:11434/api/generate",
            json={
                "model": "qwen2.5:3b",
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "num_predict": 40
                }
            },
            timeout=60
        )
        if response.status_code == 200:
            result = response.json().get("response", "").strip()
            if result.startswith("Label:"):
                result = result[6:].strip()
            return result
    except Exception as e:
        print(f"Error querying LLM: {e}")
        return None

def main():
    json_path = Path("outputs/expert_clustering/expert_hierarchy.json")
    if not json_path.exists():
        print("Could not find expert_hierarchy.json")
        return
        
    data = json.loads(json_path.read_text())
    print(f"Starting LLM Relabeling for {len(data['clusters'])} Expert Clusters...")
    
    total = len(data["clusters"])
    # Batch processing for efficiency (not literally batches of API calls, but progress tracking)
    for idx, cluster in enumerate(data["clusters"]):
        print(f"Processing {idx+1}/{total}: {cluster['name']}")
        new_label = generate_label(cluster["keywords"], cluster.get("sample_documents", []))
        if new_label:
            print(f"  -> New Label: {new_label}")
            cluster["name"] = new_label
        else:
            print("  -> Failed to generate label.")
            
        # Optional: Save every 10 to prevent loss if it crashes
        if (idx + 1) % 10 == 0:
            json_path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
            print(f"...Checkpoint saved at {idx+1}...")
            
    # Final save
    json_path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    print("Relabeling complete and saved.")

if __name__ == "__main__":
    main()
