import torch 
import torch.nn as nn 

from config import * 
from dataset import get_loaders
from model import SignEfficientNet
from utils import train_one_epoch, validate 


def main():
  train_loader, val_loader, test_loader = get_loaders()

  model = SignEfficientNet(num_classes = NUM_CLASSES).to(DEVICE) 
  criterion = nn.CrossEntropyLoss()

  optimizer = torch.optim.AdamW(
    filter(
            lambda p: p.requires_grad,
            model.parameters()
    ),
    lr = LR,
    weight_decay=WEIGHT_DECAY
  )

  scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode = 'max',
    factor = 0.5,
    patience = 2
  )

  scaler = torch.amp.GradScaler(
    "cuda",
    enabled=(DEVICE == "cuda")
  )


  best_acc = 0

  for epoch in range(EPOCHS):
    print("\n" + "=" * 50)
    print(f"Epoch [{epoch+1}/{EPOCHS}]")
    print("=" * 50)
    train_loss, train_acc = train_one_epoch(
        model=model,
        loader=train_loader,
        optimizer=optimizer,
        criterion=criterion,
        scaler=scaler,
        device = DEVICE
    )


    val_loss, val_acc = validate(
          model=model,
          loader=val_loader,
          criterion=criterion,
          device = DEVICE
    )

    scheduler.step(val_acc)

    print(f"Train Loss : {train_loss:.4f}")
    print(f"Train Acc  : {train_acc:.4f}")

    print(f"Val Loss   : {val_loss:.4f}")
    print(f"Val Acc    : {val_acc:.4f}") 


    if val_acc > best_acc:
      best_acc = val_acc 

      torch.save(model.state_dict(), SAVE_PATH)

      print("Best model saved!")


    # Fine-tuning 
    
    if epoch == 4:
      print('\nUnfreezing backbone!')
      model.unfreeze_backbone()

      optimizer = torch.optim.AdamW(
        model.parameters(), lr = 1e-5, weight_decay= WEIGHT_DECAY
      )
  print("\nTraining finished!")

  print("\nEvaluating on the test set!")

  test_loss, test_acc = validate(
      model = model,
      loader= test_loader,
      criterion = criterion,
      device = DEVICE 
    )

  print("Test accuracy:", test_acc)


if __name__ == '__main__':
  main()


