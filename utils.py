import torch 
from tqdm import tqdm 

def train_one_epoch(
  model,
  loader, 
  optimizer,
  criterion,
  scaler,
  device = 'cpu'
):
  model.train()

  total_loss = 0.0
  correct = 0
  total = 0

  pbar = tqdm(loader)

  for images, labels in pbar:
    images, labels = images.to(device), labels.to(device)

    optimizer.zero_grad()
    with torch.amp.autocast(device_type = device, enabled = (device == 'cuda')):

      logits = model(images)

      loss = criterion(logits, labels)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()



    total_loss += loss.item()

    _, preds = logits.max(1)

    total += labels.size(0)

    correct += preds.eq(labels).sum().item()
    accuracy = correct / total 

    pbar.set_description(
            f"Loss: {loss.item():.4f} "
            f"Acc: {accuracy:.4f}"
        )

  epoch_accuracy = correct / total 

  return total_loss / len(loader), epoch_accuracy 

@torch.no_grad
def validate(
  model,
  loader,
  criterion,
  device = 'cpu'
):
  model.eval()
  
  total_loss = 0.0
  correct = 0
  total = 0 

  for images, labels in loader:
    images, labels = images.to(device), labels.to(device)

    logits = model(images)
    loss = criterion(logits, labels)
    
    total_loss += loss.item()

    _, preds = logits.max(1)

    total+= labels.size(0)

    correct += preds.eq(labels).sum().item()
  accuracy = correct / total 

  return total_loss / len(loader), accuracy 
