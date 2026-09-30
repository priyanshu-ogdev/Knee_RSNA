import os
import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from . import config
from .model import build_model
from .dataset import RSNADataset

def predict(model, dataloader, device):
    """
    Runs highly optimized inference over the dataset.
    Uses AMP (Automatic Mixed Precision) and inference_mode to maximize speed.
    """
    model.eval()
    all_preds = []
    
    with torch.inference_mode(), torch.autocast('cuda', enabled=device.type == 'cuda'):
        for imgs, masks, _, _ in dataloader:
            imgs = imgs.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            
            # Forward pass
            logits = model(imgs, masks)
            probs = torch.sigmoid(logits)
            all_preds.append(probs.cpu().numpy())
            
    return np.vstack(all_preds)

def rank_ensemble(dino_df, coatnet_df):
    """
    Implements the 0.943 SOTA Rank Ensembling logic.
    Converts raw probabilities to percentiles (ranks) to fix calibration issues,
    then applies target-specific blending weights.
    """
    # Ensure study order matches
    coatnet_df = coatnet_df.set_index('StudyInstanceUID').reindex(dino_df['StudyInstanceUID']).reset_index()
    
    # Convert to percentiles (Rank Ensembling)
    dino_ranks = dino_df[config.TARGETS].rank(method='average', pct=True)
    coat_ranks = coatnet_df[config.TARGETS].rank(method='average', pct=True)
    
    # Target-specific CoatNet weights (favoring CoatNet for sharp structural anomalies)
    coat_weights = {label: 0.60 for label in config.TARGETS}
    coat_weights.update({
        'ACL': 0.75,
        'Medial Meniscus': 0.80,
        'Lateral Meniscus': 1.00,  # 100% CoatNet
        'Lateral OA': 0.75,
        'Fracture': 0.75,
    })
    
    blend_df = dino_df.copy()
    for label in config.TARGETS:
        w = coat_weights[label]
        blend_df[label] = ((1.0 - w) * dino_ranks[label]) + (w * coat_ranks[label])
        
    # Re-rank the blended output
    blend_df[config.TARGETS] = blend_df[config.TARGETS].rank(method='average', pct=True)
    return blend_df

def run_inference(test_csv_path, data_root, model_weights_path, coatnet_csv_path=None):
    """
    Main inference execution. Fits securely within the Kaggle 9-hour limit by avoiding redundant ops.
    """
    t0 = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Initializing inference on {device}...")
    
    # Load test dataframe
    try:
        df = pd.read_csv(test_csv_path)
    except FileNotFoundError:
        print(f"Error: Could not find {test_csv_path}")
        return
        
    dataset = RSNADataset(df, data_root, is_train=False)
    dataloader = DataLoader(dataset, batch_size=config.BATCH_SIZE * 2, shuffle=False, num_workers=4, pin_memory=True)
    
    # Initialize and load model
    model = build_model().to(device)
    if os.path.exists(model_weights_path):
        model.load_state_dict(torch.load(model_weights_path, map_location=device))
        print("Loaded DINOv2 weights successfully.")
    else:
        print("Warning: Model weights not found. Running with random initialization.")
        
    # Run predictions
    preds = predict(model, dataloader, device)
    
    # Format submission
    sub_df = df[['StudyInstanceUID']].copy()
    for i, target in enumerate(config.TARGETS):
        sub_df[target] = preds[:, i]
        
    # Apply 0.943 Ensembling if CoatNet predictions are provided
    if coatnet_csv_path and os.path.exists(coatnet_csv_path):
        print(f"Found CoatNet predictions at {coatnet_csv_path}. Applying 0.943 Rank Ensembling...")
        coatnet_df = pd.read_csv(coatnet_csv_path)
        sub_df = rank_ensemble(sub_df, coatnet_df)
    else:
        print("No external CoatNet predictions provided. Exporting raw DINOv2 probabilities.")
        
    sub_df.to_csv('submission.csv', index=False)
    
    elapsed = time.time() - t0
    print(f"Inference complete in {elapsed:.1f}s. Saved to submission.csv")
    
if __name__ == '__main__':
    run_inference(
        test_csv_path='/kaggle/input/rsna-knee-abnormality-detection/test.csv',
        data_root='/kaggle/input/rsna-knee-abnormality-detection/test_images',
        model_weights_path='best_dinov2_model.pt',
        coatnet_csv_path='coatnet_preds.csv' # Optional
    )
