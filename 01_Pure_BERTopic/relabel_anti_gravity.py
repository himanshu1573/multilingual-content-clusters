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
                    "num_predict": 30
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
    json_path = Path("outputs/anti_gravity/anti_gravity_hierarchy.json")
    if not json_path.exists():
        print("Could not find anti_gravity_hierarchy.json")
        return
        
    data = json.loads(json_path.read_text())
    print("Starting LLM Relabeling for Anti-Gravity Clusters...")
    
    total = len(data["clusters"])
    for idx, cluster in enumerate(data["clusters"]):
        print(f"Processing {idx+1}/{total}: {cluster['name']}")
        new_label = generate_label(cluster["keywords"], cluster.get("sample_documents", []))
        if new_label:
            print(f"  -> New Label: {new_label}")
            cluster["name"] = new_label
        else:
            print("  -> Failed to generate label.")
            
    # Save back
    json_path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    print("Relabeling complete and saved.")

if __name__ == "__main__":
    main()
