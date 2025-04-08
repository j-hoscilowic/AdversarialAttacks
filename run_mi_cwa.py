# run_isolated_mi_commonweakness.py

import torch
import torch.nn as nn
import torchvision.models as tv_models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import scipy.stats as st
from math import ceil
from abc import abstractmethod
from typing import Callable, List, Iterable
import random
import os
import csv
import time # For placeholder evaluation timing

print("--- Loading Isolated MI_CommonWeakness Script ---")

# --- Configuration ---
USE_FAKE_DATA = True # Set to False to use the real NIPS17 dataset
NIPS17_IMAGE_PATH = './resources/NIPS17/images/' # Adjust if your path is different
NIPS17_LABEL_PATH = './resources/NIPS17/images.csv' # Adjust if your path is different
BATCH_SIZE = 8 # Reduced for potentially slower CPU execution / testing
EPSILON = 16 / 255
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Using device: {DEVICE}")
if USE_FAKE_DATA:
    print("Using fake data for quick testing.")
else:
    print(f"Attempting to load NIPS17 data from: {NIPS17_IMAGE_PATH}")
    print("Ensure the dataset exists and paths are correct.")
    print("Also ensure 'Pillow' is installed (`pip install Pillow`)")

# --- Necessary External Libraries ---
# Ensure these are installed:
# pip install torch torchvision numpy scipy Pillow opencv-python
try:
    import cv2 # Check if opencv is installed (needed by original ImageHandling)
except ImportError:
    print("Warning: 'opencv-python' not found. Some original utility functions might fail if used.")
    # Define dummy cv2 functions if needed by code we keep
    class dummy_cv2:
        def imwrite(self, path, img):
            print(f"Dummy cv2.imwrite called for {path}. Requires 'opencv-python'.")
    cv2 = dummy_cv2()


# --- Utility Functions ---

# Clamp tensor values to [0, 1]
def clamp01(x: torch.Tensor) -> torch.Tensor:
    return torch.clamp(x, 0, 1)

# In-place clamp tensor values to [0, 1]
def inplace_clamp01(x: torch.Tensor) -> torch.Tensor:
    return x.clamp_(0, 1)

# Placeholder for cosine_similarity (originally from attacks/AdversarialInput/utils.py)
@torch.no_grad()
def cosine_similarity(grads: List[torch.Tensor]) -> float:
    """Placeholder for cosine similarity calculation."""
    if not grads or len(grads) < 2:
        return 1.0 # Or 0.0, depending on desired behavior for single/no grad

    # Simple average pairwise cosine similarity (example)
    sim_sum = 0.0
    count = 0
    flat_grads = [g.reshape(-1) for g in grads]
    for i in range(len(flat_grads)):
        for j in range(i + 1, len(flat_grads)):
            # Ensure tensors are on the same device for cosine_similarity
            g1 = flat_grads[i].to(DEVICE)
            g2 = flat_grads[j].to(DEVICE)
            sim = torch.nn.functional.cosine_similarity(g1, g2, dim=0)
            sim_sum += sim.item()
            count += 1
    # print(f"Placeholder cosine_similarity: {sim_sum / count if count > 0 else 1.0:.4f}")
    return sim_sum / count if count > 0 else 1.0


# --- Data Loading ---
class NIPS17(Dataset):
    def __init__(self, images_path=NIPS17_IMAGE_PATH,
                 label_path=NIPS17_LABEL_PATH):
        self.labels = {}
        try:
            with open(label_path) as f:
                reader = csv.reader(f)
                header = next(reader) # Skip header
                for line in reader:
                    # Check if line has enough elements
                    if len(line) >= 7:
                         name, label_str = line[0], line[6]
                         try:
                             label = int(label_str) - 1 # Adjust index (1-based to 0-based)
                             self.labels[name + '.png'] = label
                         except ValueError:
                             print(f"Warning: Could not parse label '{label_str}' for image '{name}'. Skipping.")
                    else:
                         print(f"Warning: Skipping malformed line in CSV: {line}")

            self.images = os.listdir(images_path)
            self.images = [img for img in self.images if img in self.labels] # Only keep images with labels
            self.images.sort()
            self.images_path = images_path
            if not self.images:
                 raise FileNotFoundError(f"No valid image files found or matched with labels in {images_path}")

        except FileNotFoundError:
            print(f"Error: NIPS17 image path '{images_path}' or label path '{label_path}' not found.")
            print("Please ensure the dataset is downloaded and paths are correct.")
            print("You can also set USE_FAKE_DATA = True for testing.")
            raise
        except Exception as e:
            print(f"An error occurred during NIPS17 dataset initialization: {e}")
            raise

        # Use standard ImageNet normalization for models pretrained on it
        self.transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224), # Most models expect 224x224
            transforms.ToTensor(),
            # No normalization here, BaseNormModel will handle it
        ])
        print(f"NIPS17 Dataset initialized. Found {len(self.images)} images with labels.")


    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        name = self.images[item]
        try:
            x = Image.open(os.path.join(self.images_path, name)).convert('RGB')
            y = self.labels[name]
            return self.transforms(x), y
        except Exception as e:
            print(f"Error loading image {name}: {e}")
            # Return dummy data or skip
            dummy_img = torch.zeros((3, 224, 224))
            dummy_label = 0
            return dummy_img, dummy_label


class FakeNIPS17Dataset(Dataset):
    """Creates fake data mimicking the NIPS17 dataset structure."""
    def __init__(self, num_samples=64, image_size=(3, 224, 224), num_classes=1000):
        self.num_samples = num_samples
        self.image_size = image_size
        self.num_classes = num_classes
        print(f"--- Using Placeholder: FakeNIPS17Dataset with {num_samples} samples ---")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        image = torch.rand(self.image_size)
        label = torch.randint(0, self.num_classes, (1,)).item()
        return image, label

def get_NIPS17_loader(batch_size=BATCH_SIZE, use_fake_data=USE_FAKE_DATA, **kwargs):
    """Gets a DataLoader for NIPS17 or fake data."""
    if use_fake_data:
        # Use a small number of samples for quick testing
        dataset = FakeNIPS17Dataset(num_samples=batch_size * 4)
        print(f"--- Using DataLoader with FAKE data (Batch Size: {batch_size}) ---")
    else:
        try:
            dataset = NIPS17(**kwargs)
            print(f"--- Using DataLoader with REAL NIPS17 data (Batch Size: {batch_size}) ---")
        except Exception as e:
            print(f"\n*** Failed to load REAL NIPS17 dataset: {e} ***")
            print("*** Switching to FAKE data for execution. ***\n")
            dataset = FakeNIPS17Dataset(num_samples=batch_size * 4)
            print(f"--- Using DataLoader with FAKE data (Batch Size: {batch_size}) ---")

    # Reduce num_workers if causing issues, especially on Windows or limited systems
    num_workers = kwargs.get('num_workers', 2)
    pin_memory = kwargs.get('pin_memory', True) if DEVICE == 'cuda' else False
    shuffle = kwargs.get('shuffle', False) # Usually False for evaluation

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                        num_workers=num_workers, pin_memory=pin_memory)
    return loader


# --- Model Definitions (Placeholders and Wrappers) ---

class Identity(nn.Module):
    """Identity wrapper."""
    def __init__(self, model=None):
        super().__init__()
        self.model = model if model is not None else nn.Identity()
        # print(f"--- Using Identity wrapper for {type(self.model).__name__} ---")

    def forward(self, x):
        return self.model(x)

class BaseNormModel(nn.Module):
    """Applies ImageNet normalization before feeding data to the model."""
    def __init__(self, model):
        super().__init__()
        self.model = model
        # ImageNet normalization
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        # print(f"--- Using BaseNormModel wrapper for {type(self.model).__name__} ---")

    def forward(self, x):
        # Ensure mean and std are on the same device as input
        mean = self.mean.to(x.device, non_blocking=True)
        std = self.std.to(x.device, non_blocking=True)
        return self.model((x - mean) / std)

# --- Define placeholder functions for actual models ---
# Use torchvision models where possible for realistic structure
def resnet18(pretrained=True): print("--- Loading torchvision: resnet18 ---"); return tv_models.resnet18(weights='IMAGENET1K_V1' if pretrained else None)
def resnet34(pretrained=True): print("--- Loading torchvision: resnet34 ---"); return tv_models.resnet34(weights='IMAGENET1K_V1' if pretrained else None)
def resnet50(pretrained=True): print("--- Loading torchvision: resnet50 ---"); return tv_models.resnet50(weights='IMAGENET1K_V1' if pretrained else None)
def resnet101(pretrained=True): print("--- Loading torchvision: resnet101 ---"); return tv_models.resnet101(weights='IMAGENET1K_V1' if pretrained else None)
def resnet152(pretrained=True): print("--- Loading torchvision: resnet152 ---"); return tv_models.resnet152(weights='IMAGENET1K_V1' if pretrained else None)
def alexnet(pretrained=True): print("--- Loading torchvision: alexnet ---"); return tv_models.alexnet(weights='IMAGENET1K_V1' if pretrained else None)
def inception_v3(pretrained=True): print("--- Loading torchvision: inception_v3 ---"); return tv_models.inception_v3(weights='IMAGENET1K_V1' if pretrained else None)
# Add others from main.py if needed...

# --- Placeholder for models not in torchvision or needing specific implementation ---
class SimpleConvNet(nn.Module):
    """A very simple CNN to act as a placeholder for complex/unknown models."""
    def __init__(self, num_classes=1000):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d((1, 1)) # Makes it robust to input size
        self.fc = nn.Linear(8, num_classes)
        print(f"--- Using Placeholder: SimpleConvNet ---")

    def forward(self, x):
        # Handle potential InceptionV3 aux logits during training mode (force eval)
        if isinstance(x, tuple): # pragma: no cover
            x = x[0]
        x = self.pool(self.relu(self.conv1(x)))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# Map unknown model names to the placeholder
def Salman2020Do_R50(pretrained=True): print("--- Using Placeholder SimpleConvNet for Salman2020Do_R50 ---"); return SimpleConvNet()
def Debenedetti2022Light_XCiT_S12(pretrained=True): print("--- Using Placeholder SimpleConvNet for Debenedetti2022Light_XCiT_S12 ---"); return SimpleConvNet()
def adv_inception_v3(pretrained=True): print("--- Using torchvision inception_v3 for adv_inception_v3 ---"); return inception_v3(pretrained=pretrained) # Reuse standard one
def ens_adv_inception_resnet_v2(pretrained=True): print("--- Using Placeholder SimpleConvNet for ens_adv_inception_resnet_v2 ---"); return SimpleConvNet()
def Wong2020Fast(pretrained=True): print("--- Using Placeholder SimpleConvNet for Wong2020Fast ---"); return SimpleConvNet()
def Engstrom2019Robustness(pretrained=True): print("--- Using Placeholder SimpleConvNet for Engstrom2019Robustness ---"); return SimpleConvNet()
# Add others from main.py if needed...


# --- Attack Code ---

# Base Class (from attacks/AdversarialInput/AdversarialInputBase.py)
class AdversarialInputAttacker():
    def __init__(self, model: List[torch.nn.Module],
                 epsilon=EPSILON,
                 norm='Linf'):
        assert norm in ['Linf', 'L2']
        self.norm = norm
        self.epsilon = epsilon
        self.models = model # Expects a list
        if not isinstance(self.models, list):
             self.models = [self.models]

        self.device = DEVICE # Use global device setting
        self.n = len(self.models)
        self.init()
        # Simplified model distribution: move all to the main device
        self.to(self.device)
        print(f"Initialized AdversarialInputAttacker with {self.n} models on {self.device}.")
        print(f"  Epsilon: {self.epsilon:.4f}, Norm: {self.norm}")


    @abstractmethod
    def attack(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        # Ensure models are on the correct device before attack
        # self.to(self.device) # Might be redundant if done at init and models aren't moved
        return self.attack(*args, **kwargs)

    # Simplified distribution
    def model_distribute(self):
        print(f"  Moving {len(self.models)} models to device: {self.device}")
        for i, model in enumerate(self.models):
            model.to(self.device)
            # Store device on model object if needed by attack logic (as in original)
            model.device = self.device

    def init(self):
        # set the model parameters requires_grad is False
        for model in self.models:
            model.requires_grad_(False)
            model.eval()

    def to(self, device: torch.device):
        for model in self.models:
            model.to(device)
            model.device = device # Keep track of the device
        self.device = device

    # Renamed from clamp to avoid conflict with global clamp01
    @torch.no_grad()
    def apply_norm_constraint(self, x: torch.Tensor, ori_x: torch.Tensor) -> torch.Tensor:
        """Applies L-inf or L-2 constraint based on self.norm and self.epsilon."""
        B = x.shape[0]
        constrained_x = x.clone() # Avoid modifying input tensor directly

        if self.norm == 'Linf':
            constrained_x = torch.clamp(constrained_x, min=ori_x - self.epsilon, max=ori_x + self.epsilon)
        elif self.norm == 'L2':
            difference = constrained_x - ori_x
            distance = torch.norm(difference.view(B, -1), p=2, dim=1)
            mask = distance > self.epsilon
            if torch.sum(mask) > 0:
                # Project difference onto the L2 ball surface
                scale = self.epsilon / distance[mask]
                # Apply scaling only to elements exceeding the norm
                difference[mask] *= scale.view(-1, 1, 1, 1)
                constrained_x = ori_x + difference # Reconstruct clamped tensor

        # Final clamp to valid image range [0, 1]
        constrained_x = torch.clamp(constrained_x, min=0, max=1)
        return constrained_x

# MI_CommonWeakness Attack Class (from attacks/AdversarialInput/CommonWeakness.py)
class MI_CommonWeakness(AdversarialInputAttacker):
    def __init__(self,
                 model: List[nn.Module],
                 total_step: int = 10,
                 random_start: bool = False,
                 step_size: float = EPSILON / 5, # Default from file: 16/255/5
                 criterion: Callable = nn.CrossEntropyLoss(),
                 targeted_attack=False,
                 mu=1.0, # Momentum decay factor
                 outer_optimizer=None, # Not used in the isolated script by default
                 reverse_step_size= EPSILON / 15, # Default from file: 16/255/15
                 inner_step_size: float = EPSILON * 1.5, # Adjusted based on common PGD practices, original 250 seemed very large
                 DI=False, # Diversity Input
                 TI=False, # Translation Invariance
                 *args,
                 **kwargs
                 ):
        # Pass epsilon, norm etc. to base class constructor
        super(MI_CommonWeakness, self).__init__(model, *args, **kwargs)

        self.random_start = random_start
        self.total_step = total_step
        self.step_size = step_size # Step size for the outer loop update (ksi in original code)
        self.criterion = criterion
        self.targerted_attack = targeted_attack
        self.mu = mu
        self.outer_optimizer = outer_optimizer # Typically None unless integrating complex optimizers
        self.reverse_step_size = reverse_step_size # Step size for the 'reverse' gradient step
        self.inner_step_size = inner_step_size # Step size for the main attack gradient step

        self.DI = DI
        self.TI = TI

        print(f"  MI_CommonWeakness Params: steps={total_step}, random_start={random_start}")
        print(f"    step_size(ksi)={self.step_size:.4f}, reverse_step={self.reverse_step_size:.4f}, inner_step={self.inner_step_size:.4f}")
        print(f"    targeted={targeted_attack}, mu={mu}, DI={DI}, TI={TI}")


        if DI:
            # Input image size (assuming 224x224 after NIPS17 transforms)
            img_size = 224
            crop_size = int(img_size * 0.9)
            padding = img_size - crop_size
            self.aug_policy = transforms.Compose([
                transforms.RandomCrop((crop_size, crop_size), padding=padding),
                transforms.Resize((img_size, img_size)) # Resize back to original size
            ])
            print("    DI enabled.")
        else:
            self.aug_policy = nn.Identity()

        if TI:
            self.ti_conv = self.gkern().to(self.device)
            self.ti_conv.requires_grad_(False)
            print("    TI enabled.")
        else:
             self.ti_conv = None # Use None check later

    def perturb(self, x):
        """Applies random start."""
        x_perturbed = x + (torch.rand_like(x) - 0.5) * 2 * self.epsilon
        x_perturbed = clamp01(x_perturbed) # Clamp to [0, 1]
        return x_perturbed

    def attack(self, x, y):
        x = x.to(self.device)
        y = y.to(self.device)
        N = x.shape[0]
        original_x = x.clone()
        adv_x = x.clone() # This will be modified

        # Momentums
        inner_momentum = torch.zeros_like(x, device=self.device)
        outer_momentum = torch.zeros_like(x, device=self.device) # Renamed from self.outer_momentum

        if self.random_start:
            adv_x = self.perturb(adv_x)

        for step in range(self.total_step):
            # --- Store original point for outer loop update ---
            current_original_for_outer = adv_x.clone().detach()

            # --- First step (Reverse Gradient) ---
            adv_x.requires_grad = True
            logit_sum_rev = 0
            for model in self.models:
                # Ensure model and input are on the same device
                logit_sum_rev += model(adv_x.to(model.device)) # Output should be on model.device

            # Ensure loss calculation happens on the primary device
            loss_rev = self.criterion(logit_sum_rev.to(self.device), y)
            loss_rev.backward()
            grad_rev = adv_x.grad.detach() # Detach grad

            # Apply TI if enabled
            if self.ti_conv is not None:
                grad_rev = self.ti_conv(grad_rev)

            adv_x.requires_grad = False # Detach from graph

            # Update based on reverse gradient
            if self.targerted_attack:
                # Move towards target: add gradient of loss wrt target
                # This implementation seems non-targeted, moving away from correct label
                 adv_x += self.reverse_step_size * grad_rev.sign() # Sign step
            else:
                # Move away from correct label: subtract gradient
                 adv_x -= self.reverse_step_size * grad_rev.sign() # Sign step

            # Apply constraints: epsilon ball and [0, 1] range
            adv_x = self.apply_norm_constraint(adv_x, original_x) # Epsilon constraint wrt original
            adv_x = clamp01(adv_x) # Range constraint [0, 1]

            # --- Second step (Forward Gradient - Main Attack) ---
            adv_x_for_inner = adv_x.clone().detach() # Use result from reverse step
            grad_inner_sum = torch.zeros_like(adv_x_for_inner, device=self.device)
            self.grad_record = [] # Reset grad record for this step

            for model in self.models:
                adv_x_for_inner.requires_grad = True

                # Apply DI if enabled
                aug_x = self.aug_policy(adv_x_for_inner)

                # Ensure model, input, and label are on the same device
                loss_inner = self.criterion(model(aug_x.to(model.device)), y.to(model.device))
                loss_inner.backward()

                grad_inner = adv_x_for_inner.grad.detach() # Detach grad
                self.grad_record.append(grad_inner.clone()) # Store grad for cosine sim later

                # Apply TI if enabled
                if self.ti_conv is not None:
                    grad_inner = self.ti_conv(grad_inner)

                # Accumulate gradients (or handle momentum per model if needed)
                grad_inner_sum += grad_inner # Simple sum of gradients

                # Reset grad for next model
                adv_x_for_inner.grad = None
                adv_x_for_inner.requires_grad = False


            # Normalize the summed gradient for momentum update (using L1 norm like original MI-FGSM)
            grad_norm = torch.norm(grad_inner_sum.reshape(N, -1), p=1, dim=1).view(N, 1, 1, 1) + 1e-12 # Avoid division by zero
            normalized_grad = grad_inner_sum / grad_norm

            # Update inner momentum
            inner_momentum = self.mu * inner_momentum + normalized_grad

            # Update adv_x based on inner momentum (using inner_step_size)
            if self.targerted_attack:
                 # This implementation seems non-targeted
                 adv_x -= self.inner_step_size * inner_momentum.sign() # Sign step
            else:
                 adv_x += self.inner_step_size * inner_momentum.sign() # Sign step

            # Apply constraints
            adv_x = self.apply_norm_constraint(adv_x, original_x)
            adv_x = clamp01(adv_x)

            # --- Outer Loop Update (End Attack Logic) ---
            # This simulates the 'end_attack' logic using momentum
            # ksi = self.step_size (renamed for clarity)
            # fake_grad = (adv_x - current_original_for_outer) # Difference created in this step

            # Normalize fake_grad (using L1 norm like original)
            # fake_grad_norm = torch.norm(fake_grad.reshape(N, -1), p=1, dim=1).view(N, 1, 1, 1) + 1e-12
            # normalized_fake_grad = fake_grad / fake_grad_norm
            # outer_momentum = self.mu * outer_momentum + normalized_fake_grad

            # Simplified outer update: Use sign of difference directly with outer step size (ksi)
            # This matches the original code's structure more closely if outer_optimizer is None
            diff = adv_x - current_original_for_outer
            outer_momentum = self.mu * outer_momentum + diff / (torch.norm(diff.reshape(N, -1), p=1, dim=1).view(N, 1, 1, 1) + 1e-12)


            # Update adv_x using outer momentum and step_size (ksi)
            adv_x = current_original_for_outer + self.step_size * outer_momentum.sign()

            # Final constraints for the step
            adv_x = self.apply_norm_constraint(adv_x, original_x)
            adv_x = clamp01(adv_x)

            # Optional: Calculate cosine similarity for monitoring (doesn't affect attack)
            # sim = cosine_similarity(self.grad_record)
            # if (step + 1) % 5 == 0: # Print every 5 steps
            #      print(f"  Step {step+1}/{self.total_step}, Grad Cos Sim (placeholder): {sim:.4f}")


        # Clean up grads just in case
        del self.grad_record
        # Return final adversarial example
        return adv_x.detach()


    @staticmethod
    def gkern(kernlen=15, nsig=3):
        """Returns a 2D Gaussian kernel array for TI."""
        x = np.linspace(-nsig, nsig, kernlen)
        kern1d = st.norm.pdf(x)
        kernel_raw = np.outer(kern1d, kern1d)
        kernel = kernel_raw / kernel_raw.sum()
        kernel = torch.tensor(kernel, dtype=torch.float32) # Ensure float32
        # Create a Conv2d layer to apply the kernel
        conv = nn.Conv2d(3, 3, kernel_size=kernlen, stride=1, padding=kernlen // 2, bias=False, groups=3)
        # Set the weights of the Conv2d layer
        kernel_weights = kernel.repeat(3, 1, 1, 1) # Shape: (out_channels, in_channels/groups, H, W) -> (3, 1, k, k)
        conv.weight = nn.Parameter(kernel_weights, requires_grad=False)
        return conv


# --- Tester Placeholder ---
def test_transfer_attack_acc(attacker, loader, test_models):
    """Placeholder function to simulate evaluating the attack."""
    print(f"\n--- Running Placeholder Evaluation: test_transfer_attack_acc ---")
    print(f"Attacker: {attacker.__class__.__name__}")
    print(f"Testing against {len(test_models)} target models.")
    start_time = time.time()

    # Use the device from the attacker
    device = attacker.device

    total_correct = {i: 0 for i in range(len(test_models))}
    total_samples = 0
    max_batches_to_test = 4 # Limit batches for quick placeholder testing

    attacker.init() # Ensure attacker models are in eval mode

    for batch_idx, (images, labels) in enumerate(loader):
        if batch_idx >= max_batches_to_test:
            print(f"    (Stopping after {max_batches_to_test} batches for placeholder test)")
            break
        print(f"    Processing batch {batch_idx + 1}/{min(len(loader), max_batches_to_test)}...")

        images, labels = images.to(device), labels.to(device)
        batch_size = images.size(0)

        # Generate adversarial examples using the attacker
        adv_images = attacker(images, labels) # Attacker is callable

        # Test against each target model
        with torch.no_grad():
            for i, model in enumerate(test_models):
                model.eval() # Ensure model is in eval mode
                try:
                    # Move adv_images to the target model's device if necessary
                    # (Assuming all test models are on the same main DEVICE here)
                    outputs = model(adv_images.to(DEVICE))

                    # Handle InceptionV3 output tuple (logits, aux_logits)
                    if isinstance(outputs, tuple):
                        outputs = outputs[0]

                    _, predicted = torch.max(outputs.data, 1)
                    correct_in_batch = (predicted == labels).sum().item()
                    total_correct[i] += correct_in_batch
                except Exception as e:
                    print(f"      Error evaluating target model {i}: {e}")
                    # Optionally break or continue
            total_samples += batch_size # Increment total samples only if batch processed successfully


    end_time = time.time()
    print("\n--- Placeholder Evaluation Results (Simulated Accuracy on Adversarial Examples) ---")
    if total_samples == 0:
         print("No samples processed.")
    else:
        for i, model in enumerate(test_models):
             model_name = f"TargetModel_{i}"
             try:
                 actual_model = model.module if isinstance(model, nn.DataParallel) else model
                 actual_model = actual_model.model if hasattr(actual_model, 'model') else actual_model
                 model_name = actual_model.__class__.__name__
             except Exception: pass
             accuracy = 100 * total_correct[i] / total_samples
             print(f"  Target Model {i} ({model_name}): {accuracy:.2f}% Acc ({total_correct[i]}/{total_samples})")

    print(f"--- Placeholder Evaluation took {end_time - start_time:.2f} seconds ---")
    print("-" * 60)


# --- Main Script Logic (Adapted from your main.py) ---

if __name__ == "__main__":
    print("\n--- Starting Main Execution Block ---")

    # 1. Setup Loader
    print("Setting up data loader...")
    loader = get_NIPS17_loader(batch_size=BATCH_SIZE, use_fake_data=USE_FAKE_DATA,
                               images_path=NIPS17_IMAGE_PATH, label_path=NIPS17_LABEL_PATH)

    # 2. Prepare Train/Source Models (Used by the attacker)
    print("\nLoading and preparing source models (for attack generation)...")
    # Using fewer models for faster placeholder execution
    origin_train_models_defs = [
        (BaseNormModel, resnet18),
        (BaseNormModel, resnet50),
        #(Identity, Salman2020Do_R50), # Using placeholder SimpleConvNet
    ]
    train_models = []
    for wrapper, model_func in origin_train_models_defs:
        try:
            # Load model (potentially downloading weights)
            model_instance = model_func(pretrained=True)
            # Apply wrapper
            wrapped_model = wrapper(model_instance)
            # Set to eval mode, move to device, disable gradients
            wrapped_model.eval().to(DEVICE)
            wrapped_model.requires_grad_(False)
            train_models.append(wrapped_model)
            print(f"  Loaded source model: {model_func.__name__}")
        except Exception as e:
            print(f"  Failed to load source model {model_func.__name__}: {e}")
            print("  Ensure torchvision is installed and you have internet for pretrained weights.")

    if not train_models:
        print("\nError: No source models could be loaded. Cannot proceed.")
        exit()


    # 3. Prepare Test/Target Models (Used for evaluation)
    print("\nLoading and preparing target models (for evaluation)...")
    test_models = []
    # Using fewer models for faster placeholder execution
    origin_test_models_defs_norm = [
        alexnet,
        resnet152,
        # adv_inception_v3, # InceptionV3 can be tricky with aux logits
    ]
    origin_test_models_defs_id = [
        # Wong2020Fast, # Placeholder SimpleConvNet
        # Engstrom2019Robustness, # Placeholder SimpleConvNet
    ]

    for model_func in origin_test_models_defs_norm:
        try:
            now_model = BaseNormModel(model_func(pretrained=True)).to(DEVICE)
            now_model.eval()
            now_model.requires_grad_(False)
            test_models.append(now_model)
            print(f"  Loaded target model (BaseNorm): {model_func.__name__}")
        except Exception as e:
            print(f"  Failed to load target model {model_func.__name__}: {e}")

    for model_func in origin_test_models_defs_id:
         try:
            now_model = Identity(model_func(pretrained=True)).to(DEVICE)
            now_model.eval()
            now_model.requires_grad_(False)
            test_models.append(now_model)
            print(f"  Loaded target model (Identity): {model_func.__name__}")
         except Exception as e:
            print(f"  Failed to load target model {model_func.__name__}: {e}")

    if not test_models:
        print("\nWarning: No target models could be loaded. Evaluation will be skipped.")

    # 4. Instantiate and Run the MI_CommonWeakness Attack and Evaluation
    print("\nInstantiating MI_CommonWeakness attacker...")
    attacker = MI_CommonWeakness(
        model=train_models, # Pass the source models
        epsilon=EPSILON,
        # Add other MI_CommonWeakness parameters if needed (e.g., total_step, DI, TI)
        total_step=10,
        DI=False,
        TI=False,
    )
    print(f"Attacker Class: {attacker.__class__}")

    # 5. Evaluate the attack
    if test_models and len(loader) > 0:
        test_transfer_attack_acc(attacker, loader, test_models)
    elif not test_models:
         print("Skipping evaluation as no target models were loaded.")
    else:
         print("Skipping evaluation as data loader is empty.")


    print("\n--- Isolated MI_CommonWeakness Script Finished ---")

