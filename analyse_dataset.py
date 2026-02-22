import pandas as pd
from collections import Counter
import re

df = pd.read_csv("datasets/dpl_full.csv")

# 1. Duplicate analysis
total_dupes = df.duplicated(subset="description").sum()
print(f"Total duplicate descriptions: {total_dupes} ({100*total_dupes/len(df):.1f}%)")

# Cross-tag duplicates (same description, different tag) — worst case for a classifier
cross = df[df.duplicated(subset="description", keep=False)]
cross_conflict = cross.groupby("description")["dpl_tag"].nunique()
conflicted = cross_conflict[cross_conflict > 1]
print(f"Descriptions with CONFLICTING tags (same text, different label): {len(conflicted)}")
if len(conflicted) > 0:
    print("Examples:")
    for desc in list(conflicted.index)[:5]:
        tags = df[df["description"] == desc]["dpl_tag"].tolist()
        print(f'  "{desc}" -> {tags}')

# 2. Template repetition — unique descriptions per tag
print()
print("Unique descriptions per tag (lowest 15 = most repetition):")
uniq = df.groupby("dpl_tag")["description"].nunique().sort_values()
print(uniq.head(15).to_string())

# 3. Overall variety
print()
print("Template variety (unique / 300 per tag):")
variety = (df.groupby("dpl_tag")["description"].nunique() / 300)
print(variety.describe())
print("Tags with variety < 0.20 (severely repetitive):")
print(variety[variety < 0.20].sort_values().to_string())

# 4. Shared vocabulary — words that appear in many tags reduce discriminability
tag_words = {}
for tag, grp in df.groupby("dpl_tag"):
    words = re.findall(r"[a-zA-Z]{4,}", " ".join(grp["description"].str.lower()))
    tag_words[tag] = Counter(words)

word_tag_count = Counter()
for tag, wc in tag_words.items():
    for w in wc:
        word_tag_count[w] += 1

shared_10plus = {w: c for w, c in word_tag_count.items() if c >= 10}
print(f"\nWords in 10+ different tags (low discriminative power): {len(shared_10plus)}")
print("Top 20:", sorted(shared_10plus, key=lambda x: -shared_10plus[x])[:20])

# 5. Identify the most confusable tag pairs by lexical overlap
print()
print("Most lexically similar tag pairs (Jaccard similarity on top-50 words):")
similarities = []
tags = list(tag_words.keys())
for i in range(len(tags)):
    for j in range(i+1, len(tags)):
        a = set(w for w, c in tag_words[tags[i]].most_common(50))
        b = set(w for w, c in tag_words[tags[j]].most_common(50))
        jaccard = len(a & b) / len(a | b) if a | b else 0
        similarities.append((tags[i], tags[j], jaccard))

similarities.sort(key=lambda x: -x[2])
print(f"{'Tag A':<10} {'Tag B':<10} {'Jaccard':>8}  Shared words")
for a, b, j in similarities[:20]:
    shared = set(w for w, c in tag_words[a].most_common(50)) & set(w for w, c in tag_words[b].most_common(50))
    print(f"{a:<10} {b:<10} {j:>8.3f}  {sorted(shared)[:8]}")

# 6. Tags with fewest unique templates (hardest to learn from)
print()
print("Tags at risk — very few unique descriptions:")
at_risk = uniq[uniq < 20]
if len(at_risk):
    print(at_risk.to_string())
else:
    print("None below 20 unique descriptions.")
