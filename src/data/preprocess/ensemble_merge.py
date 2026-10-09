import pandas as pd
import numpy as np
import argparse
import os

TARGETS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
    "Medial OA", "Lateral OA", "PF OA",
    "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"
]

def merge_models(csv1, csv2, out_csv):
    df1 = pd.read_csv(csv1)
    df2 = pd.read_csv(csv2)
    
    # Align by StudyInstanceUID
    df1 = df1.set_index('StudyInstanceUID').sort_index()
    df2 = df2.set_index('StudyInstanceUID').sort_index()
    
    common_idx = df1.index.intersection(df2.index)
    df1 = df1.loc[common_idx]
    df2 = df2.loc[common_idx]
    
    out_df = pd.DataFrame(index=common_idx)
    
    for t in TARGETS:
        # State values
        y1, y2 = df1[t].values, df2[t].values
        # Weights
        w1, w2 = df1[f'{t}_weight'].values, df2[f'{t}_weight'].values
        
        # Agreement logic
        # Both output valid predictions
        valid = (y1 >= 0.0) & (y2 >= 0.0)
        
        # Agreement threshold (e.g. diff <= 0.1)
        agree = valid & (np.abs(y1 - y2) <= 0.1)
        disagree = valid & (np.abs(y1 - y2) > 0.1)
        
        final_y = np.full_like(y1, -1.0) # default not stated
        final_w = np.full_like(w1, 0.1)  # default not stated weight
        
        # Where they agree, take average prediction and max weight
        final_y[agree] = (y1[agree] + y2[agree]) / 2.0
        final_w[agree] = np.maximum(w1[agree], w2[agree])
        
        # Where they disagree, completely mask the loss (weight=0) to prevent noisy gradients
        final_w[disagree] = 0.0
        
        # Where one is valid and the other is -1.0 (not stated), trust the valid one but lower weight
        only_1 = (y1 >= 0.0) & (y2 < 0.0)
        final_y[only_1] = y1[only_1]
        final_w[only_1] = w1[only_1] * 0.5
        
        only_2 = (y2 >= 0.0) & (y1 < 0.0)
        final_y[only_2] = y2[only_2]
        final_w[only_2] = w2[only_2] * 0.5
        
        out_df[t] = final_y
        out_df[f'{t}_weight'] = final_w
        
    out_df = out_df.reset_index()
    out_df.to_csv(out_csv, index=False)
    print(f"[SUCCESS] Ensembled {len(out_df)} studies -> {out_csv}")
    print("Disagreements masked to weight=0.0")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv1', required=True)
    parser.add_argument('--csv2', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    merge_models(args.csv1, args.csv2, args.out)
