import torch
import torch.optim as optim
import warmup_scheduler
from test_model import evaluate

def fit(epochs, lr, momentum, weight_decay, milestones, gamma, model, train_loader, val_loader):
    history = []

    #criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999), weight_decay=5e-5)
    base_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    scheduler = warmup_scheduler.GradualWarmupScheduler(optimizer, multiplier=1., total_epoch=5, after_scheduler=base_scheduler)

    for epoch in range(epochs):
        # Training Phase 
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            loss = model.training_step(batch)
            loss.backward()
            optimizer.step()
            scheduler.step()
        # Validation phase
        model.eval()
        result = evaluate(model, val_loader)
        model.epoch_end(epoch, result)
        history.append(result)
    return history
