import numpy as np
import pandas as pd

# ----------------------------
# Load concepts
# ----------------------------
path = '../../Datasets/Things-EEG2/image_metadata.npy'
meta = np.load(path, allow_pickle=True).item()

full_set = meta['train_img_concepts_THINGS'] + meta['test_img_concepts_THINGS']

df = pd.DataFrame({'concept': full_set}).drop_duplicates(subset=['concept'])

# ----------------------------
# Split into first_digits + last_str
# ----------------------------
# Extract leading digits (as nullable Int so missing stays <NA> instead of crashing)
df['first_digits'] = (
    df['concept']
    .str.extract(r'^(\d+)', expand=False)
    .astype('Int64')
)

# Extract trailing text part (letters/underscores), then remove ONLY leading underscores
df['last_str'] = (
    df['concept']
    .str.replace(r'^\d+_+', '', regex=True)
    .str.lstrip('_')
)
print(df)

# Sort by numeric id (keep NAs last)
df = df.sort_values(['first_digits', 'concept'], na_position='last').reset_index(drop=True)

# ----------------------------
# Map to categories via merge (fast + safe)
# ----------------------------
# Use long format - it has category, uniqueID, Word columns
category_long_df = pd.read_csv(
    '../../Datasets/Things-EEG2/category53_long-format.tsv',
    sep='\t'
)

# Group categories per uniqueID (some concepts belong to multiple categories)
category_grouped = (
    category_long_df
    .groupby('uniqueID')['category']
    .apply(list)
    .reset_index()
)
category_grouped.columns = ['uniqueID', 'categories']

# Merge on last_str (concept name without numeric prefix)
df = df.merge(
    category_grouped,
    left_on='last_str',
    right_on='uniqueID',
    how='left'
)

# Optional: rename outputs the way you intended
df = df.rename(columns={'uniqueID': 'map_key'})

# Debug: show which ones didn't map
not_found = df[df['categories'].isna()][['concept', 'last_str']]
if len(not_found) > 0:
    print(f"Not found: {len(not_found)}")
    print(not_found.head(30).to_string(index=False))

# Save if needed
df.to_csv('../../Datasets/Things-EEG2/things_concepts_split.csv', index=False)
print(f"\nSaved {len(df)} concepts with category mappings")
print(df[['concept', 'first_digits', 'last_str', 'categories']].head(20).to_string())
