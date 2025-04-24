import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import time
import copy

def train_model(model, train_loader, criterion, optimizer, device, scheduler=None):
    """Performs one training epoch."""
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    for inputs, labels in tqdm(train_loader, desc="Training", leave=False):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total_samples += labels.size(0)
        correct_predictions += (predicted == labels).sum().item()

    epoch_loss = running_loss / total_samples
    epoch_acc = correct_predictions / total_samples

    if scheduler:
        scheduler.step() # Or scheduler.step(epoch_loss) for ReduceLROnPlateau

    return epoch_loss, epoch_acc

def evaluate_model(model, test_loader, criterion, device):
    """Evaluates the model on the test set."""
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Evaluating", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / total_samples
    epoch_acc = correct_predictions / total_samples

    return epoch_loss, epoch_acc, all_labels, all_preds


def run_training_loop(model, train_loader, test_loader, num_epochs, learning_rate, device, model_name="model"):
    """Main training loop with evaluation."""
    criterion = nn.CrossEntropyLoss()
    # Use AdamW for potentially better generalization
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
    # Optional: Learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1) # Example scheduler

    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    print(f"\nStarting training for {num_epochs} epochs on {device}...")
    start_time = time.time()

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print('-' * 10)

        train_loss, train_acc = train_model(model, train_loader, criterion, optimizer, device, scheduler)
        val_loss, val_acc, _, _ = evaluate_model(model, test_loader, criterion, device) # Get metrics only during training loop

        print(f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}   Acc: {val_acc:.4f}")

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        # Save best model weights based on validation accuracy
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            print(f"New best validation accuracy: {best_acc:.4f} (Saving model weights)")
            # NOTE: We don't save the model to disk here to comply with submission rules
            # but you would typically save it during development:
            # torch.save(model.state_dict(), f'{model_name}_best.pth')


    time_elapsed = time.time() - start_time
    print(f'\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # Load best model weights for final evaluation
    model.load_state_dict(best_model_wts)

    # Final evaluation on the test set with the best model
    print("\nFinal evaluation using best model weights:")
    final_loss, final_acc, final_labels, final_preds = evaluate_model(model, test_loader, criterion, device)
    print(f"Final Test Loss: {final_loss:.4f}")
    print(f"Final Test Acc: {final_acc:.4f}")

    return model, history, final_labels, final_preds