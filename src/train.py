import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import time
from . import config
from .model import build_model
from .dataset import RSNADataset

def train_epoch(model, dataloader, optimizer, scaler, scheduler, device):
    model.train()
    total_loss = 0.0
    
    for batch_idx, (imgs, masks, targets, weights) in enumerate(dataloader):
        imgs = imgs.to(device)
        masks = masks.to(device)
        targets = targets.to(device)
        weights = weights.to(device)

        with torch.autocast('cuda', enabled=device.type == 'cuda'):
            preds = model(imgs, masks)
            loss = (F.binary_cross_entropy_with_logits(preds, targets, reduction='none') * weights).mean()

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        
        total_loss += loss.item()
        
    return total_loss / max(len(dataloader), 1)

def run_training(train_csv_path, data_root):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Building Model on {device}...")
    model = build_model().to(device)
    
    # Load data
    print("Loading Dataset...")
    try:
        df = pd.read_csv(train_csv_path)
    except FileNotFoundError:
        print(f"Warning: {train_csv_path} not found. Using a dummy dataframe for testing.")
        df = pd.DataFrame(columns=['StudyInstanceUID', 'laterality'] + config.TARGETS + [f"{t}_weight" for t in config.TARGETS])
    
    dataset = RSNADataset(df, data_root, is_train=True)
    dataloader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    
    # Dual Learning Rate setup
    opt = torch.optim.AdamW([
        {'params': [p for p in model.backbone.parameters() if p.requires_grad], 'lr': config.LR_BACKBONE},
        {'params': model.head.parameters(), 'lr': config.LR_HEAD}
    ], weight_decay=config.WEIGHT_DECAY)
    
    num_training_steps = config.EPOCHS * max(len(dataloader), 1)
    
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, 
        max_lr=[config.LR_BACKBONE, config.LR_HEAD], 
        total_steps=num_training_steps, 
        pct_start=0.15
    )
    
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')
    
    print("Starting Training Loop...")
    t0 = time.time()
    for ep in range(config.EPOCHS):
        loss = train_epoch(model, dataloader, opt, scaler, sched, device)
        print(f"Epoch {ep+1}/{config.EPOCHS} | Loss: {loss:.4f}")
        
        if time.time() - t0 > config.TIME_BUDGET_HOURS * 3600:
            print("Time budget reached! Stopping early to prevent timeouts.")
            break
            
if __name__ == '__main__':
    run_training('train.csv', '/kaggle/input/rsna-knee-abnormality-detection/train_images')
