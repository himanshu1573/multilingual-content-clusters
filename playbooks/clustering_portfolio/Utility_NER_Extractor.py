import pandas as pd
import torch
from transformers import pipeline
import os
from pathlib import Path
from tqdm import tqdm
import re

def clean_for_ner(text):
    """
    Cleans video titles to remove noise that confuses NER models.
    """
    if not isinstance(text, str):
        return ""
    
    # 1. Remove hashtags
    text = re.sub(r'#\w+', '', text)
    
    # 2. Remove common news channel suffixes and pipe noise
    # Patterns for common Hindi/English news outlets
    channels = [
        r'\|?\s*ABP\s*NEWS', r'\|?\s*ZEE\s*NEWS', r'\|?\s*INDIA\s*TODAY', 
        r'\|?\s*NEWS18', r'\|?\s*AAJ\s*TAK', r'\|?\s*NEWS24', 
        r'\|?\s*NDTV', r'\|?\s*REPUBLIC', r'\|?\s*TIMES\s*NOW',
        r'\|?\s*HINDUSTAN\s*TIMES', r'\|?\s*SAKSHI\s*POST',
        r'\|?\s*विशेश बुलेटिन', r'\|?\s*Janhit', r'\|?\s*Breaking',
        r'\|?\s*Latest News', r'\|?\s*Big News', r'\|?\s*LIVE',
        r'\|?\s*Analysis', r'\|?\s*Report'
    ]
    for pattern in channels:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    
    # 3. Remove common prefixes
    prefixes = [
        r'^Breaking News[:\s]*', r'^Top News[:\s]*', r'^Latest News[:\s]*', 
        r'^LIVE[:\s]*', r'^Big News[:\s]*', r'^News Updates[:\s]*'
    ]
    for prefix in prefixes:
        text = re.sub(prefix, '', text, flags=re.IGNORECASE)
    
    # 4. Remove emojis and specific decorative characters
    # This keeps alphanumeric, punctuation, and Hindi characters
    text = re.sub(r'[^\w\s\d,.\-\!\?\u0900-\u097F]', '', text)
    
    # 5. Clean up surrounding noise characters
    text = text.strip(' |:-')
    
    # 6. Normalize extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def deduplicate_entities(entities):
    """
    Deduplicates entities, keeping the longest string when one is a substring of another.
    Processes each entity group separately.
    """
    import collections
    grouped = collections.defaultdict(list)
    for ent in entities:
        # ent is {'entity_group': 'PER', 'word': '...', 'score': ...}
        word = ent['word'].strip()
        # Remove artifacts often seen in RoBERTa/multilingual models
        word = word.replace(' ', ' ').strip()
        if word:
            grouped[ent['entity_group']].append(word)
    
    deduped = {}
    for group, words in grouped.items():
        unique_words = sorted(list(set(words)), key=len, reverse=True)
        final_words = []
        for w in unique_words:
            # If current word is already covered by a longer word, skip it
            # e.g., if 'Donald Trump' is there, skip 'Trump'
            # Also handle cases with different scripts separately (best effort)
            is_covered = False
            for existing in final_words:
                if w.lower() in existing.lower() and w != existing:
                    is_covered = True
                    break
            if not is_covered:
                final_words.append(w)
        deduped[group] = ", ".join(final_words)
    
    return deduped

def main():
    input_file = Path("Video Titles.txt")
    output_file = Path("Video_Titles_NER.csv")
    
    if not input_file.exists():
        print(f"Error: {input_file} not found.")
        return

    print(f"Loading {input_file}...")
    # Using the robust loader logic for video titles
    try:
        df = pd.read_csv(input_file)
    except Exception:
        df = pd.read_csv(input_file, sep="\v", engine="python")
    
    if 'title' not in df.columns:
        # Fallback to first column
        df.columns = ['title'] + list(df.columns[1:])

    titles_raw = df['title'].fillna("").astype(str).tolist()
    print("Cleaning titles for NER...")
    titles_clean = [clean_for_ner(t) for t in titles_raw]
    
    # Determine device
    device = "cpu"
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = 0
    print(f"Using device: {device}")

    # Using high-speed, accurate multilingual model
    model_name = "Babelscape/wikineural-multilingual-ner"
    print(f"Initializing NER Pipeline with {model_name}...")
    ner_pipe = pipeline(
        "ner", 
        model=model_name, 
        aggregation_strategy="simple", # Automatically merges BPE tokens
        device=device
    )
    print("Model loaded successfully.")

    results = []
    print(f"Extracting entities from {len(titles_raw)} titles...")
    
    batch_size = 32
    for i in tqdm(range(0, len(titles_clean), batch_size)):
        batch_titles_clean = titles_clean[i : i + batch_size]
        batch_titles_raw = titles_raw[i : i + batch_size]
        
        try:
            # Filter out empty strings to avoid errors
            # We must keep track of which indices were valid to map results back correctly
            valid_indices = [idx for idx, t in enumerate(batch_titles_clean) if t.strip()]
            
            if not valid_indices:
                for r in batch_titles_raw:
                    results.append({"Original Title": r, "Persons": "", "Organizations": "", "Locations": ""})
                continue
            
            clean_inputs = [batch_titles_clean[idx] for idx in valid_indices]
            batch_results = ner_pipe(clean_inputs)
            
            # Map results back to index
            results_idx = 0
            for j in range(len(batch_titles_raw)):
                if j in valid_indices:
                    entities_raw = batch_results[results_idx]
                    
                    if isinstance(entities_raw, dict): # Single result case
                        entities_raw = [entities_raw]
                    
                    deduped = deduplicate_entities(entities_raw)
                    results.append({
                        "Original Title": batch_titles_raw[j],
                        "Persons": deduped.get("PER", ""),
                        "Organizations": deduped.get("ORG", ""),
                        "Locations": deduped.get("LOC", "")
                    })
                    results_idx += 1
                else:
                    results.append({
                        "Original Title": batch_titles_raw[j],
                        "Persons": "", "Organizations": "", "Locations": ""
                    })
        except Exception as e:
            print(f"Error in batch {i}: {e}")
            for r in batch_titles_raw:
                results.append({"Original Title": r, "Persons": "", "Organizations": "", "Locations": ""})

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_file, index=False)
    print(f"\nNER Extraction Complete! Saved to {output_file}")

if __name__ == "__main__":
    main()
