import json
from collections import Counter
import os

def main():
    filepath = "data/counter_speech_kb/filtered_candidates.jsonl"
    if not os.path.exists(filepath):
        print("Filtered candidates not found.")
        return

    category_counts = Counter()
    source_counts = Counter()
    target_counts = Counter()
    language_counts = Counter()
    provenance_counts = Counter()
    strategy_counts = Counter()

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            c = json.loads(line)
            category_counts[c.get('category')] += 1
            source_counts[c.get('source')] += 1
            
            target = c.get('target')
            if target: target_counts[target] += 1
            
            language_counts[c.get('language')] += 1
            provenance_counts[c.get('provenance')] += 1
            
            strategy = c.get('strategy')
            if strategy: strategy_counts[strategy] += 1

    report = {
        "category": dict(category_counts),
        "source": dict(source_counts),
        "target": dict(target_counts),
        "language": dict(language_counts),
        "provenance": dict(provenance_counts),
        "strategy": dict(strategy_counts)
    }

    os.makedirs("reports", exist_ok=True)
    with open("reports/m5_3_gap_analysis.json", 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
        
    print("Gap analysis written to reports/m5_3_gap_analysis.json")

if __name__ == "__main__":
    main()
