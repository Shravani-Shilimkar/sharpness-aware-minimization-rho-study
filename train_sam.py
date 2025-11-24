# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torch.nn.functional as F
# import torchvision
# import torchvision.transforms as transforms
# import time
# import os
# import argparse
# from sam_optimizer import SAM
# from torch.optim.lr_scheduler import CosineAnnealingLR

# # --- Configuration ---
# # Match settings from Section 4.2 of the project proposal
# NUM_EPOCHS = 100
# BATCH_SIZE = 128
# LEARNING_RATE = 0.1
# MOMENTUM = 0.9
# WEIGHT_DECAY = 5e-4
# RHO = 0.05 # SAM parameter for this run (ρ = 0.05)
# DEVICE = 'mps' if torch.backends.mps.is_available() else 'cpu'

# # --- Model Definition (ResNet-18) ---

# def conv3x3(in_planes, out_planes, stride=1):
#     return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

# class BasicBlock(nn.Module):
#     expansion = 1

#     def __init__(self, in_planes, planes, stride=1):
#         super(BasicBlock, self).__init__()
#         self.conv1 = conv3x3(in_planes, planes, stride)
#         self.bn1 = nn.BatchNorm2d(planes)
#         self.conv2 = conv3x3(planes, planes)
#         self.bn2 = nn.BatchNorm2d(planes)

#         self.shortcut = nn.Sequential()
#         if stride != 1 or in_planes != self.expansion * planes:
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(self.expansion * planes)
#             )

#     def forward(self, x):
#         out = F.relu(self.bn1(self.conv1(x)))
#         out = self.bn2(self.conv2(out))
#         out += self.shortcut(x)
#         out = F.relu(out)
#         return out

# class ResNet(nn.Module):
#     def __init__(self, block, num_blocks, num_classes=100):
#         super(ResNet, self).__init__()
#         self.in_planes = 64

#         self.conv1 = conv3x3(3, 64)
#         self.bn1 = nn.BatchNorm2d(64)
#         self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
#         self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
#         self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
#         self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
#         self.linear = nn.Linear(512 * block.expansion, num_classes)

#     def _make_layer(self, block, planes, num_blocks, stride):
#         strides = [stride] + [1] * (num_blocks - 1)
#         layers = []
#         for stride in strides:
#             layers.append(block(self.in_planes, planes, stride))
#             self.in_planes = planes * block.expansion
#         return nn.Sequential(*layers)

#     def forward(self, x):
#         out = F.relu(self.bn1(self.conv1(x)))
#         out = self.layer1(out)
#         out = self.layer2(out)
#         out = self.layer3(out)
#         out = self.layer4(out)
#         out = F.avg_pool2d(out, 4)
#         out = out.view(out.size(0), -1)
#         out = self.linear(out)
#         return out

# def ResNet18():
#     return ResNet(BasicBlock, [2, 2, 2, 2])


# # --- Data Setup ---
# def get_data_loaders():
#     print('==> Preparing data..')
#     transform_train = transforms.Compose([
#         transforms.RandomCrop(32, padding=4),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
#     ])

#     transform_test = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
#     ])

#     trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
#     trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

#     testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
#     testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
    
#     return trainloader, testloader

# # --- Training and Testing Functions ---

# def train(epoch, model, optimizer, scheduler, criterion, trainloader):
#     """
#     SAM training step: requires a closure for the second forward/backward pass.
#     """
#     model.train()
#     train_loss = 0
#     correct = 0
#     total = 0
#     start_time = time.time()

#     for batch_idx, (inputs, targets) in enumerate(trainloader):
#         inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        
#         # 1. First forward/backward pass (computes gradient at w)
#         # This is implicitly needed before calling optimizer.step() 
#         # but the SAM implementation uses a closure for the second pass.
        
#         def closure():
#             # Reset gradients for the current pass
#             optimizer.zero_grad() 
#             outputs = model(inputs)
#             loss = criterion(outputs, targets)
#             loss.backward()
#             return loss

#         # Run the first closure manually to get the initial gradient at w
#         # The SAM step will handle the second pass internally
#         loss = closure()
        
#         # SAM's custom step handles perturbation, second backward pass, and reset/update
#         # The loss returned here is the loss at w + epsilon
#         loss = optimizer.step(closure)
        
#         train_loss += loss.item()
        
#         # For tracking accuracy, we use the output from the last forward pass 
#         # which happened during the second closure inside optimizer.step()
#         # This is a slight approximation but standard practice for SAM.
        
#         # Re-run forward pass for final accuracy tracking after update
#         outputs = model(inputs)
        
#         _, predicted = outputs.max(1)
#         total += targets.size(0)
#         correct += predicted.eq(targets).sum().item()

#     scheduler.step()
#     end_time = time.time()
    
#     avg_loss = train_loss / len(trainloader)
#     accuracy = 100. * correct / total
    
#     print(f'Epoch: {epoch} | Time: {end_time - start_time:.2f}s | Train Loss: {avg_loss:.4f} | Acc: {accuracy:.2f}% ({correct}/{total})')
    
#     return avg_loss, accuracy, (end_time - start_time)

# def test(epoch, model, criterion, testloader, best_acc, checkpoint_dir):
#     model.eval()
#     test_loss = 0
#     correct = 0
#     total = 0
#     with torch.no_grad():
#         for inputs, targets in testloader:
#             inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
#             outputs = model(inputs)
#             loss = criterion(outputs, targets)

#             test_loss += loss.item()
#             _, predicted = outputs.max(1)
#             total += targets.size(0)
#             correct += predicted.eq(targets).sum().item()

#     avg_loss = test_loss / len(testloader)
#     accuracy = 100. * correct / total
    
#     print(f'Test Epoch: {epoch} | Test Loss: {avg_loss:.4f} | Acc: {accuracy:.2f}% ({correct}/{total})')

#     # Save checkpoint if it's the best accuracy so far
#     if accuracy > best_acc:
#         print('==> Saving Best Model..')
#         state = {
#             'net': model.state_dict(),
#             'acc': accuracy,
#             'epoch': epoch,
#         }
#         if not os.path.isdir(checkpoint_dir):
#             os.mkdir(checkpoint_dir)
#         torch.save(state, os.path.join(checkpoint_dir, 'sam_rho005_best_ckpt.pth'))
#         best_acc = accuracy
    
#     return avg_loss, accuracy, best_acc

# # --- Main Execution ---
# def main():
#     print(f'==> Starting SAM Training with Rho={RHO} on {DEVICE}...')
    
#     # Setup directories and logging
#     log_dir = './logs'
#     checkpoint_dir = './checkpoint'
#     os.makedirs(log_dir, exist_ok=True)
    
#     trainloader, testloader = get_data_loaders()

#     # Model and initialization
#     net = ResNet18().to(DEVICE)
#     criterion = nn.CrossEntropyLoss()
    
#     # Base Optimizer (SGD) - Note: weight decay is handled by the base optimizer
#     base_optimizer = optim.SGD
    
#     # SAM Optimizer setup
#     optimizer = SAM(
#         net.parameters(), 
#         base_optimizer, 
#         rho=RHO, 
#         lr=LEARNING_RATE, 
#         momentum=MOMENTUM, 
#         weight_decay=WEIGHT_DECAY
#     )

#     # Cosine Annealing Learning Rate Scheduler
#     scheduler = CosineAnnealingLR(optimizer.base_optimizer, T_max=NUM_EPOCHS)
    
#     # Storage for plotting later
#     train_loss_history = []
#     test_loss_history = []
#     train_acc_history = []
#     test_acc_history = []
#     runtime_per_epoch = []
#     best_acc = 0.0

#     print(f"Total Epochs: {NUM_EPOCHS}")
#     for epoch in range(1, NUM_EPOCHS + 1):
#         # TRAIN
#         train_loss, train_acc, runtime = train(epoch, net, optimizer, scheduler, criterion, trainloader)
        
#         # TEST
#         test_loss, test_acc, best_acc = test(epoch, net, criterion, testloader, best_acc, checkpoint_dir)
        
#         # Log data
#         train_loss_history.append(train_loss)
#         test_loss_history.append(test_loss)
#         train_acc_history.append(train_acc)
#         test_acc_history.append(test_acc)
#         runtime_per_epoch.append(runtime)

#     print(f"\nFinal Best Test Accuracy (rho={RHO}): {best_acc:.2f}%")
    
#     # Save training history for later plotting and analysis
#     history_data = {
#         'train_loss': train_loss_history,
#         'test_loss': test_loss_history,
#         'train_acc': train_acc_history,
#         'test_acc': test_acc_history,
#         'runtime_per_epoch': runtime_per_epoch,
#         'best_acc': best_acc,
#         'rho': RHO
#     }
#     torch.save(history_data, os.path.join(log_dir, 'sam_rho005_history.pth'))

# if __name__ == '__main__':
#     main()



# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torch.nn.functional as F
# import torchvision
# import torchvision.transforms as transforms
# import time
# import os
# import argparse
# from sam_optimizer import SAM
# from torch.optim.lr_scheduler import CosineAnnealingLR

# # --- Configuration ---
# # Match settings from Section 4.2 of the project proposal
# NUM_EPOCHS = 100
# BATCH_SIZE = 128
# LEARNING_RATE = 0.1
# MOMENTUM = 0.9
# WEIGHT_DECAY = 5e-4
# RHO = 0.05 # SAM parameter for this run (ρ = 0.05)
# DEVICE = 'mps' if torch.backends.mps.is_available() else 'cpu'

# # --- Model Definition (ResNet-18) ---

# def conv3x3(in_planes, out_planes, stride=1):
#     return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

# class BasicBlock(nn.Module):
#     expansion = 1

#     def __init__(self, in_planes, planes, stride=1):
#         super(BasicBlock, self).__init__()
#         self.conv1 = conv3x3(in_planes, planes, stride)
#         self.bn1 = nn.BatchNorm2d(planes)
#         self.conv2 = conv3x3(planes, planes)
#         self.bn2 = nn.BatchNorm2d(planes)

#         self.shortcut = nn.Sequential()
#         if stride != 1 or in_planes != self.expansion * planes:
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(self.expansion * planes)
#             )

#     def forward(self, x):
#         out = F.relu(self.bn1(self.conv1(x)))
#         out = self.bn2(self.conv2(out))
#         out += self.shortcut(x)
#         out = F.relu(out)
#         return out

# class ResNet(nn.Module):
#     def __init__(self, block, num_blocks, num_classes=100):
#         super(ResNet, self).__init__()
#         self.in_planes = 64

#         self.conv1 = conv3x3(3, 64)
#         self.bn1 = nn.BatchNorm2d(64)
#         self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
#         self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
#         self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
#         self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
#         self.linear = nn.Linear(512 * block.expansion, num_classes)

#     def _make_layer(self, block, planes, num_blocks, stride):
#         strides = [stride] + [1] * (num_blocks - 1)
#         layers = []
#         for stride in strides:
#             layers.append(block(self.in_planes, planes, stride))
#             self.in_planes = planes * block.expansion
#         return nn.Sequential(*layers)

#     def forward(self, x):
#         out = F.relu(self.bn1(self.conv1(x)))
#         out = self.layer1(out)
#         out = self.layer2(out)
#         out = self.layer3(out)
#         out = self.layer4(out)
#         out = F.avg_pool2d(out, 4)
#         out = out.view(out.size(0), -1)
#         out = self.linear(out)
#         return out

# def ResNet18():
#     return ResNet(BasicBlock, [2, 2, 2, 2])


# # --- Data Setup ---
# def get_data_loaders():
#     print('==> Preparing data..')
#     transform_train = transforms.Compose([
#         transforms.RandomCrop(32, padding=4),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
#     ])

#     transform_test = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
#     ])

#     trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
#     trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

#     testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
#     testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
    
#     return trainloader, testloader

# # --- Training and Testing Functions ---

# def train(epoch, model, optimizer, scheduler, criterion, trainloader):
#     """
#     SAM training step: requires a closure for the second forward/backward pass.
#     """
#     model.train()
#     train_loss = 0
#     correct = 0
#     total = 0
#     start_time = time.time()

#     for batch_idx, (inputs, targets) in enumerate(trainloader):
#         inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        
#         # --- SAM STEP 1: Calculate gradient at w (first forward/backward pass) ---
#         # 1. Forward pass at w
#         outputs = model(inputs)
#         loss = criterion(outputs, targets)
        
#         # 2. Backward pass at w
#         loss.backward()

#         # --- SAM STEP 2: Perturb and calculate gradient at w + epsilon ---
#         # Define the closure for the second pass (at w + epsilon)
#         def closure():
#             # The perturb_weights function in SAM will call optimizer.zero_grad() 
#             # and then call this closure to get the gradient at w + epsilon.
#             # We must ensure the graph is reset for the second backward pass.
            
#             # Note: We do NOT call optimizer.zero_grad() here; it's handled internally by SAM
#             outputs = model(inputs)
#             loss_closure = criterion(outputs, targets)
#             loss_closure.backward() # Second backward pass
#             return loss_closure

#         # 3. Call SAM's custom step
#         # This function handles the perturbation, calls closure(), calculates grad_w+eps, 
#         # resets weights to w, overwrites grad_w with grad_w+eps, and finally calls SGD.step().
#         loss_at_w_plus_eps = optimizer.step(closure)
        
#         # We use the loss at w + epsilon for logging as it represents the sharper minimum
#         train_loss += loss_at_w_plus_eps.item()
        
#         # After the optimizer.step() call, the model weights have been updated (w <- w - lr * grad_w+eps).
        
#         # We track accuracy using the initial output from the forward pass at w 
#         # (or re-run forward pass for most accurate tracking, but using the initial loss output is fine for training history)
#         # Using the output at w+eps is also common, but for simplicity, we use the model's current state:
        
#         # Re-run forward pass for final accuracy tracking after update
#         # (Note: This is an extra forward pass outside the SAM mechanism, but useful for accurate batch-level stats)
#         outputs = model(inputs)
        
#         _, predicted = outputs.max(1)
#         total += targets.size(0)
#         correct += predicted.eq(targets).sum().item()

#     scheduler.step()
#     end_time = time.time()
    
#     # We log the average loss as the sum of loss_at_w_plus_eps divided by the number of batches
#     avg_loss = train_loss / len(trainloader)
#     accuracy = 100. * correct / total
    
#     print(f'Epoch: {epoch} | Time: {end_time - start_time:.2f}s | Train Loss: {avg_loss:.4f} | Acc: {accuracy:.2f}% ({correct}/{total})')
    
#     return avg_loss, accuracy, (end_time - start_time)

# def test(epoch, model, criterion, testloader, best_acc, checkpoint_dir):
#     model.eval()
#     test_loss = 0
#     correct = 0
#     total = 0
#     with torch.no_grad():
#         for inputs, targets in testloader:
#             inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
#             outputs = model(inputs)
#             loss = criterion(outputs, targets)

#             test_loss += loss.item()
#             _, predicted = outputs.max(1)
#             total += targets.size(0)
#             correct += predicted.eq(targets).sum().item()

#     avg_loss = test_loss / len(testloader)
#     accuracy = 100. * correct / total
    
#     print(f'Test Epoch: {epoch} | Test Loss: {avg_loss:.4f} | Acc: {accuracy:.2f}% ({correct}/{total})')

#     # Save checkpoint if it's the best accuracy so far
#     if accuracy > best_acc:
#         print('==> Saving Best Model..')
#         state = {
#             'net': model.state_dict(),
#             'acc': accuracy,
#             'epoch': epoch,
#         }
#         if not os.path.isdir(checkpoint_dir):
#             os.mkdir(checkpoint_dir)
#         torch.save(state, os.path.join(checkpoint_dir, 'sam_rho005_best_ckpt.pth'))
#         best_acc = accuracy
    
#     return avg_loss, accuracy, best_acc

# # --- Main Execution ---
# def main():
#     print(f'==> Starting SAM Training with Rho={RHO} on {DEVICE}...')
    
#     # Setup directories and logging
#     log_dir = './logs'
#     checkpoint_dir = './checkpoint'
#     os.makedirs(log_dir, exist_ok=True)
#     os.makedirs(checkpoint_dir, exist_ok=True) # Ensure checkpoint directory exists
    
#     trainloader, testloader = get_data_loaders()

#     # Model and initialization
#     net = ResNet18().to(DEVICE)
#     criterion = nn.CrossEntropyLoss()
    
#     # Base Optimizer (SGD) - Note: weight decay is handled by the base optimizer
#     base_optimizer = optim.SGD
    
#     # SAM Optimizer setup
#     optimizer = SAM(
#         net.parameters(), 
#         base_optimizer, 
#         rho=RHO, 
#         lr=LEARNING_RATE, 
#         momentum=MOMENTUM, 
#         weight_decay=WEIGHT_DECAY
#     )

#     # Cosine Annealing Learning Rate Scheduler
#     scheduler = CosineAnnealingLR(optimizer.base_optimizer, T_max=NUM_EPOCHS)
    
#     # Storage for plotting later
#     train_loss_history = []
#     test_loss_history = []
#     train_acc_history = []
#     test_acc_history = []
#     runtime_per_epoch = []
#     best_acc = 0.0

#     print(f"Total Epochs: {NUM_EPOCHS}")
#     for epoch in range(1, NUM_EPOCHS + 1):
#         # TRAIN
#         train_loss, train_acc, runtime = train(epoch, net, optimizer, scheduler, criterion, trainloader)
        
#         # TEST
#         test_loss, test_acc, best_acc = test(epoch, net, criterion, testloader, best_acc, checkpoint_dir)
        
#         # Log data
#         train_loss_history.append(train_loss)
#         test_loss_history.append(test_loss)
#         train_acc_history.append(train_acc)
#         test_acc_history.append(test_acc)
#         runtime_per_epoch.append(runtime)

#     print(f"\nFinal Best Test Accuracy (rho={RHO}): {best_acc:.2f}%")
    
#     # Save training history for later plotting and analysis
#     history_data = {
#         'train_loss': train_loss_history,
#         'test_loss': test_loss_history,
#         'train_acc': train_acc_history,
#         'test_acc': test_acc_history,
#         'runtime_per_epoch': runtime_per_epoch,
#         'best_acc': best_acc,
#         'rho': RHO
#     }
#     torch.save(history_data, os.path.join(log_dir, 'sam_rho005_history.pth'))

# if __name__ == '__main__':
#     main()




import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import time
import os
import argparse
from sam_optimizer import SAM
from torch.optim.lr_scheduler import CosineAnnealingLR

# --- Configuration ---
# Match settings from Section 4.2 of the project proposal
NUM_EPOCHS = 100 # Adjusted back to 100 for faster debugging/testing
BATCH_SIZE = 128
LEARNING_RATE = 0.1
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4
RHO = 0.05 # SAM parameter for this run (ρ = 0.05)
DEVICE = 'mps' if torch.backends.mps.is_available() else 'cpu'

# --- Model Definition (ResNet-18) ---

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=100):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = conv3x3(3, 64)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])


# --- Data Setup ---
def get_data_loaders():
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
    
    return trainloader, testloader

# --- Training and Testing Functions ---

def train(epoch, model, optimizer, scheduler, criterion, trainloader):
    """
    SAM training step: requires a closure for the second forward/backward pass.
    """
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    start_time = time.time()

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        
        # --- SAM STEP 1: Calculate gradient at w (first forward/backward pass) ---
        # 1. Zero the previous gradients
        optimizer.zero_grad() 
        
        # 2. Forward pass at w
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # 3. Backward pass at w
        # CRITICAL FIX: retain_graph=True is necessary here so the graph is not destroyed
        # and can be used for the second backward pass inside the closure.
        loss.backward(retain_graph=True)

        # --- SAM STEP 2: Perturb and calculate gradient at w + epsilon ---
        # Define the closure for the second pass (at w + epsilon)
        def closure():
            # 1. Zero gradients before the second backward pass (important for momentum)
            optimizer.zero_grad() 

            # 2. Forward pass at w + epsilon (this is run after weights are perturbed by optimizer.step)
            outputs = model(inputs)
            loss_closure = criterion(outputs, targets)
            
            # 3. Backward pass at w + epsilon
            loss_closure.backward() 
            return loss_closure

        # 4. Call SAM's custom step
        # This function handles the perturbation, calls closure(), resets weights to w, 
        # overwrites grad_w with grad_w+eps, and finally calls SGD.step().
        loss_at_w_plus_eps = optimizer.step(closure)
        
        # We use the loss at w + epsilon for logging as it represents the sharper minimum
        train_loss += loss_at_w_plus_eps.item()
        
        # Track accuracy using the output at the *final* position w after the update
        outputs = model(inputs)
        
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    scheduler.step()
    end_time = time.time()
    
    # We log the average loss as the sum of loss_at_w_plus_eps divided by the number of batches
    avg_loss = train_loss / len(trainloader)
    accuracy = 100. * correct / total
    
    print(f'Epoch: {epoch} | Time: {end_time - start_time:.2f}s | Train Loss: {avg_loss:.4f} | Acc: {accuracy:.2f}% ({correct}/{total})')
    
    return avg_loss, accuracy, (end_time - start_time)

def test(epoch, model, criterion, testloader, best_acc, checkpoint_dir):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).sum().item()

    avg_loss = test_loss / len(testloader)
    accuracy = 100. * correct / total
    
    print(f'Test Epoch: {epoch} | Test Loss: {avg_loss:.4f} | Acc: {accuracy:.2f}% ({correct}/{total})')

    # Save checkpoint if it's the best accuracy so far
    if accuracy > best_acc:
        print('==> Saving Best Model..')
        state = {
            'net': model.state_dict(),
            'acc': accuracy,
            'epoch': epoch,
        }
        if not os.path.isdir(checkpoint_dir):
            os.mkdir(checkpoint_dir)
        torch.save(state, os.path.join(checkpoint_dir, 'sam_rho005_best_ckpt.pth'))
        best_acc = accuracy
    
    return avg_loss, accuracy, best_acc

# --- Main Execution ---
def main():
    print(f'==> Starting SAM Training with Rho={RHO} on {DEVICE}...')
    
    # Setup directories and logging
    log_dir = './logs'
    checkpoint_dir = './checkpoint'
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True) # Ensure checkpoint directory exists
    
    trainloader, testloader = get_data_loaders()

    # Model and initialization
    net = ResNet18().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    
    # Base Optimizer (SGD) - Note: weight decay is handled by the base optimizer
    base_optimizer = optim.SGD
    
    # SAM Optimizer setup
    optimizer = SAM(
        net.parameters(), 
        base_optimizer, 
        rho=RHO, 
        lr=LEARNING_RATE, 
        momentum=MOMENTUM, 
        weight_decay=WEIGHT_DECAY
    )

    # Cosine Annealing Learning Rate Scheduler
    scheduler = CosineAnnealingLR(optimizer.base_optimizer, T_max=NUM_EPOCHS)
    
    # Storage for plotting later
    train_loss_history = []
    test_loss_history = []
    train_acc_history = []
    test_acc_history = []
    runtime_per_epoch = []
    best_acc = 0.0

    print(f"Total Epochs: {NUM_EPOCHS}")
    for epoch in range(1, NUM_EPOCHS + 1):
        # TRAIN
        train_loss, train_acc, runtime = train(epoch, net, optimizer, scheduler, criterion, trainloader)
        
        # TEST
        test_loss, test_acc, best_acc = test(epoch, net, criterion, testloader, best_acc, checkpoint_dir)
        
        # Log data
        train_loss_history.append(train_loss)
        test_loss_history.append(test_loss)
        train_acc_history.append(train_acc)
        test_acc_history.append(test_acc)
        runtime_per_epoch.append(runtime)

    print(f"\nFinal Best Test Accuracy (rho={RHO}): {best_acc:.2f}%")
    
    # Save training history for later plotting and analysis
    history_data = {
        'train_loss': train_loss_history,
        'test_loss': test_loss_history,
        'train_acc': train_acc_history,
        'test_acc': test_acc_history,
        'runtime_per_epoch': runtime_per_epoch,
        'best_acc': best_acc,
        'rho': RHO
    }
    torch.save(history_data, os.path.join(log_dir, 'sam_rho005_history.pth'))

if __name__ == '__main__':
    main()