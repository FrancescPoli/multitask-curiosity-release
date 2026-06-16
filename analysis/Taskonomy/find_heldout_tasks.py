
import pandas as pd

# Load the CSV
df = pd.read_csv('analysis/Taskonomy/Data/modcog_fundamental_struct2.csv')

def calculate_complexity(row):
    score = 0
    score += row['integration'] 
    score += row['sequence'] 
    score += row['anti'] 
    score += row['has_delay'] 
    score += row['stim_in_decision'] 
    score += row['has_stim2'] 
    score += row['has_cohs'] 
    score += row['has_match_rule'] 
    score += row['has_matchgo'] 
    return score

df['complexity'] = df.apply(calculate_complexity, axis=1)

# Sort by complexity (ascending), then by name
df_sorted = df.sort_values(by=['complexity', 'name'])

# Get tasks 20 to 25 (The next 5)
held_out = df_sorted.iloc[20:25]['name'].tolist()

print("\nHeld-out Tasks (Next 5 simplest):")
for t in held_out:
    print(f"khona.{t}")
