import torch
import torch.nn.functional as F
import numpy as np
import time
import config
from model import build_model

def train_epoch(model, dataloader, optimizer, scaler, scheduler, device):
    model.train()
    total_loss = 0.0
    
    for imgs, masks, targets, weights in dataloader:
        imgs = imgs.to(device)
        masks = masks.to(device)
        targets = targets.to(device)
        weights = weights.to(device)

        with torch.autocast('cuda', enabled=device.type == 'cuda'):
            preds = model(imgs, masks)
            # Custom weighted BCE Loss
            loss = (F.binary_cross_entropy_with_logits(preds, targets, reduction='none') * weights).mean()

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        
        total_loss += loss.item()
        
    return total_loss / max(len(dataloader), 1)

def run_training():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model().to(device)
    
    # Dual Learning Rate setup
    opt = torch.optim.AdamW([
        {'params': [p for p in model.backbone.parameters() if p.requires_grad], 'lr': config.LR_BACKBONE},
        {'params': model.head.parameters(), 'lr': config.LR_HEAD}
    ], weight_decay=config.WEIGHT_DECAY)
    
    # Example placeholder values
    num_training_steps = config.EPOCHS * 100 # Adjust 100 to actual steps/epoch based on dataset
    
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, 
        max_lr=[config.LR_BACKBONE, config.LR_HEAD], 
        total_steps=num_training_steps, 
        pct_start=0.15
    )
    
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')
    
    t0 = time.time()
    for ep in range(config.EPOCHS):
        # loss = train_epoch(model, dataloader, opt, scaler, sched, device)
        # print(f"Epoch {ep+1}/{config.EPOCHS} | Loss: {loss:.4f}")
        
        # Hard time budget check
        if time.time() - t0 > config.TIME_BUDGET_HOURS * 3600:
            print("Time budget reached! Stopping early.")
            break
            
if __name__ == '__main__':
    print("Training script ready. Requires valid dataloader implementation to run.")
