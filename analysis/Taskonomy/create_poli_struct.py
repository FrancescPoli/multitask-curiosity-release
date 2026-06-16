
import pandas as pd
import numpy as np

# Define columns expected by plot_taskonomy_separate.py
columns = [
    "name", "family", 
    "w_mod_type", "match_rule_type", "matchgo_type",
    "has_cohs", "has_stim2", "has_match_rule", "has_matchgo",
    "integration", "sequence", "anti", "has_delay", "stim_in_decision"
]

data = []

def add_task(name, family, **kwargs):
    row = {c: 0 for c in columns} # Default 0s
    row["name"] = name
    row["family"] = family
    row["w_mod_type"] = "none"
    row["match_rule_type"] = "none"
    row["matchgo_type"] = "match"
    
    # Update with kwargs
    for k, v in kwargs.items():
        if k in row:
            row[k] = v
    
    # Logic for derived fields (to make manually setting them easier)
    if kwargs.get('ctx', False):
        row['w_mod_type'] = 'context_mod1'
    if kwargs.get('multi', False):
        row['w_mod_type'] = 'multi_sum'
    if kwargs.get('cat', False):
        row['match_rule_type'] = 'category'
    if kwargs.get('nms', False):
        row['matchgo_type'] = 'non_match'
    
    # Primitives from kwargs
    if kwargs.get('anti', False):
        row['anti'] = 1
    if kwargs.get('dly', False):
        row['has_delay'] = 1
    if kwargs.get('rt', False):
        row['stim_in_decision'] = 1
        row['has_delay'] = 0 # RT implies no delay
    
    # Family specifics
    if family == "_DMFamily":
        row['has_cohs'] = 1
    if family == "_DelayMatch1DResponse":
        row['has_stim2'] = 1
        row['has_match_rule'] = 1
        row['has_matchgo'] = 1
        row['has_delay'] = 1 # Match always has delay (dlyms)
        
    data.append(row)

# --- 1. Reach Family ---
# Base: go
add_task('poli.go', '_Reach') # Simple (No modifiers)

# L1
add_task('poli.dlygo', '_Reach', dly=True)
add_task('poli.rtgo', '_Reach', rt=True)
add_task('poli.antigo', '_Reach', anti=True)
add_task('poli.ctxgo', '_Reach', ctx=True)

# L2
add_task('poli.dlyantigo', '_Reach', dly=True, anti=True)
add_task('poli.rtantigo', '_Reach', rt=True, anti=True)
add_task('poli.dlyctxgo', '_Reach', dly=True, ctx=True)
add_task('poli.antictxgo', '_Reach', anti=True, ctx=True)
add_task('poli.rtctxgo', '_Reach', rt=True, ctx=True)

# L3
add_task('poli.dlyantictxgo', '_Reach', dly=True, anti=True, ctx=True)
add_task('poli.rtantictxgo', '_Reach', rt=True, anti=True, ctx=True)


# --- 2. Decision Family ---
# Need to handle Modality Splits (dm1, dm2). The script treats names as unique identifiers but groups them by state.
# I'll add both variants if they exist, or just one representative? 
# "Total Registered Tasks: 48". I should list them all.

def add_dm_pair(base_name, **kwargs):
    # Add dm1 and dm2 (or just base_name if multi)
    if kwargs.get('multi', False):
        add_task(f'poli.{base_name}', '_DMFamily', **kwargs)
    else:
        add_task(f'poli.{base_name}1', '_DMFamily', **kwargs)
        if base_name != 'dm': # dm1/dm2 special case? No, consistent.
             pass # Actually, for struct, we usually list specific tasks.
        add_task(f'poli.{base_name}2', '_DMFamily', **kwargs)

# Base
add_dm_pair('dm') # dm1, dm2

# L1
add_dm_pair('dlydm', dly=True)
add_dm_pair('multidm', multi=True) # Single task
add_dm_pair('ctxdm', ctx=True)
add_dm_pair('antidm', anti=True)

# L2
add_dm_pair('ctxdlydm', dly=True, ctx=True)
add_dm_pair('multidlydm', dly=True, multi=True) # Single
add_dm_pair('antidlydm', dly=True, anti=True)
add_dm_pair('antictxdm', ctx=True, anti=True)
add_dm_pair('antimultidm', multi=True, anti=True) # Single

# L3
add_dm_pair('antictxdlydm', dly=True, ctx=True, anti=True)
add_dm_pair('antimultidlydm', dly=True, multi=True, anti=True) # Single


# --- 3. Match Family ---
# Base
add_task('poli.dlyms', '_DelayMatch1DResponse') # Intrinsic delay

# L1
add_task('poli.antidlyms', '_DelayMatch1DResponse', anti=True)
add_task('poli.ctxdlyms', '_DelayMatch1DResponse', ctx=True)
add_task('poli.catdlyms', '_DelayMatch1DResponse', cat=True)
add_task('poli.dlynms', '_DelayMatch1DResponse', nms=True)

# L2
add_task('poli.antictxdlyms', '_DelayMatch1DResponse', anti=True, ctx=True)
add_task('poli.anticatdlyms', '_DelayMatch1DResponse', anti=True, cat=True)
add_task('poli.ctxcatdlyms', '_DelayMatch1DResponse', ctx=True, cat=True)
add_task('poli.catdlynms', '_DelayMatch1DResponse', cat=True, nms=True)
add_task('poli.antidlynms', '_DelayMatch1DResponse', anti=True, nms=True)
add_task('poli.ctxdlynms', '_DelayMatch1DResponse', ctx=True, nms=True)

# L3
add_task('poli.antictxcatdlyms', '_DelayMatch1DResponse', anti=True, ctx=True, cat=True)
add_task('poli.antictxdlynms', '_DelayMatch1DResponse', anti=True, ctx=True, nms=True)
add_task('poli.anticatdlynms', '_DelayMatch1DResponse', anti=True, cat=True, nms=True)
add_task('poli.ctxcatdlynms', '_DelayMatch1DResponse', ctx=True, cat=True, nms=True)

# L4
add_task('poli.antictxcatdlynms', '_DelayMatch1DResponse', anti=True, ctx=True, cat=True, nms=True)


# Convert to DF and Save
df = pd.DataFrame(data)
df.to_csv("analysis/Taskonomy/Data/poli_struct.csv", index=False)
print("Created poli_struct.csv with", len(df), "tasks.")
print(df.groupby('family')['name'].count())
