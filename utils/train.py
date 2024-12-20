import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

class Trainer:
    def __init__(self, model, device, train_loader, test_loader, optimizer, scheduler=None):
        self.model = model
        self.device = device
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
        
    def train_epoch(self):
        self.model.train()
        pbar = tqdm(self.train_loader)
        train_loss = 0
        correct = 0
        processed = 0

        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad(set_to_none=True)

            if self.scaler is not None:
                with torch.cuda.amp.autocast():
                    pred = self.model(data)
                    loss = F.cross_entropy(pred, target)
                
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                pred = self.model(data)
                loss = F.cross_entropy(pred, target)
                loss.backward()
                self.optimizer.step()

            if self.scheduler:
                self.scheduler.step()

            train_loss += loss.item()
            pred = pred.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            processed += len(data)
            
            pbar.set_description(
                desc=f'Loss={loss.item():0.4f} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}'
            )

        return train_loss/len(self.train_loader), 100*correct/processed

    def test(self):
        self.model.eval()
        test_loss = 0
        correct = 0
        
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += F.cross_entropy(output, target, reduction='sum').item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(self.test_loader.dataset)
        accuracy = 100. * correct / len(self.test_loader.dataset)
        
        print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(self.test_loader.dataset)} ({accuracy:.2f}%)\n')
        
        return test_loss, accuracy 