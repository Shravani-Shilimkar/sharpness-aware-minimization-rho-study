# import torch
# from torch.optim import Optimizer

# class SAM(Optimizer):
#     """
#     Sharpness-Aware Minimization (SAM) Optimizer.

#     This custom optimizer implements the two-step procedure of SAM:
#     1. Finds the adversarial perturbation (epsilon) that maximizes the loss within 
#        a neighborhood of radius rho (gradient ascent step).
#     2. Performs the standard optimization step (gradient descent) using the 
#        gradient computed at the sharpest point (w + epsilon).
    
#     It wraps a base optimizer (e.g., SGD) which handles the actual weight updates 
#     and momentum/weight decay logic.

#     Based on the paper: Sharpness-Aware Minimization for Efficiently Improving Generalization
#     by Foret et al. (2021).
#     """

#     def __init__(self, params, base_optimizer, rho=0.05, **kwargs):
#         """
#         Initializes the SAM optimizer.

#         Args:
#             params (iterable): Iterable of parameters to optimize or dicts defining
#                                parameter groups.
#             base_optimizer (torch.optim.Optimizer): The base optimizer (e.g., torch.optim.SGD)
#                                                     that will perform the final weight update.
#             rho (float): The neighborhood radius hyperparameter (p in the proposal).
#             **kwargs: Arguments to be passed to the base_optimizer.
#         """
#         # Initialize the base optimizer using the provided parameters and kwargs
#         # self.param_groups is inherited from the base class
#         super(SAM, self).__init__(params, defaults=dict(rho=rho, **kwargs))
        
#         # Instantiate the base optimizer
#         self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        
#         # We need the param_groups list from the base optimizer for the next steps
#         self.param_groups = self.base_optimizer.param_groups 
        
#         # Shared gradient state storage
#         self.grad_after_perturb = []


#     @torch.no_grad()
#     def _grad_norm(self, weights):
#         """
#         Calculates the L2 norm of the gradient vector across all parameters.
#         This is required for normalizing the perturbation (epsilon).
#         """
#         norm = 0.0
#         for p in weights:
#             # We assume the parameter 'p' has a gradient 'p.grad' attached.
#             if p.grad is not None:
#                 norm += p.grad.norm(2).item() ** 2
#         return norm ** 0.5


#     @torch.no_grad()
#     def perturb_weights(self, closure):
#         """
#         STEP 1: Calculates the perturbation epsilon and applies it to the weights.
        
#         Args:
#             closure (callable): A function that re-evaluates the model and 
#                                 returns the loss (required for the second backward pass).
#         """
#         # Calculate the overall gradient norm
#         grad_norm = self._grad_norm(self.param_groups[0]['params'])
        
#         # A small constant for numerical stability (1e-12 as per proposal pseudocode)
#         epsilon = 1e-12 
        
#         for group in self.param_groups:
#             scale = group['rho'] / (grad_norm + epsilon) # Calculate scale factor
            
#             for p in group['params']:
#                 if p.grad is None:
#                     continue
                
#                 # Store original weight and gradient needed for the next step
#                 state = self.state[p]
#                 state['old_w'] = p.data.clone()
                
#                 # Calculate perturbation vector (epsilon)
#                 # epsilon = rho * grad / (grad_norm + 1e-12)
#                 e_w = p.grad * scale 
                
#                 # Add perturbation to weights: w <- w + epsilon
#                 p.data.add_(e_w)

#         # STEP 2a (Start): Calculate the gradient at the sharpest point (w + epsilon)
#         # This requires a new forward and backward pass
#         loss = closure()
        
#         # Store the new gradients (gradient at w + epsilon) before resetting
#         self.grad_after_perturb.clear()
#         for group in self.param_groups:
#             for p in group['params']:
#                 if p.grad is None:
#                     continue
#                 self.grad_after_perturb.append(p.grad.data.clone())

#         # STEP 2b: Reset parameters to their original position (w)
#         for group in self.param_groups:
#             for p in group['params']:
#                 if 'old_w' in self.state[p]:
#                     p.data = self.state[p]['old_w']
#                     del self.state[p]['old_w']

#         return loss

    
#     def step(self, closure=None):
#         """
#         Performs a single optimization step (parameter update).
        
#         Args:
#             closure (callable, optional): A closure that reevaluates the model 
#                                          and returns the loss. 
#                                          This is MANDATORY for SAM.
#         """
#         if closure is None:
#             raise ValueError("SAM requires a closure to perform the two forward/backward passes.")
            
#         # The first backward pass (which computes grad at w) must be done 
#         # BEFORE calling self.perturb_weights(closure).
#         # Aditi's part (baseline SGD) should have included this first pass.
#         # This implementation requires the user to perform: loss.backward() BEFORE step().
        
#         # STEP 1 & 2a/2b: Perturb weights, calculate gradient at w+epsilon, and reset weights.
#         self.perturb_weights(closure)

#         # STEP 2c: Call the base optimizer's step() method.
#         # The base optimizer will use the gradients stored in self.grad_after_perturb 
#         # (which are the gradients at w + epsilon) to update the original parameters w.
        
#         # Manually overwrite the gradient of w with the gradient of w + epsilon
#         # This is critical: SGD must use the SAM-derived gradient.
        
#         grad_idx = 0
#         for group in self.param_groups:
#             for p in group['params']:
#                 if p.grad is None:
#                     continue
#                 p.grad.data.copy_(self.grad_after_perturb[grad_idx])
#                 grad_idx += 1
        
#         # Clear the temporary storage
#         self.grad_after_perturb.clear()

#         # Perform the actual update using the base optimizer (SGD)
#         loss = self.base_optimizer.step()
        
#         return loss

    
#     def zero_grad(self, set_to_none: bool = False):
#         """Zeroes out the gradient buffers for all parameters."""
#         self.base_optimizer.zero_grad(set_to_none=set_to_none)



# import torch
# from torch.optim import Optimizer

# class SAM(Optimizer):
#     """
#     Sharpness-Aware Minimization (SAM) Optimizer.

#     This custom optimizer implements the two-step procedure of SAM:
#     1. Finds the adversarial perturbation (epsilon) that maximizes the loss within 
#        a neighborhood of radius rho (gradient ascent step).
#     2. Performs the standard optimization step (gradient descent) using the 
#        gradient computed at the sharpest point (w + epsilon).
    
#     It wraps a base optimizer (e.g., SGD) which handles the actual weight updates 
#     and momentum/weight decay logic.

#     Based on the paper: Sharpness-Aware Minimization for Efficiently Improving Generalization
#     by Foret et al. (2021).
#     """

#     def __init__(self, params, base_optimizer, rho=0.05, **kwargs):
#         """
#         Initializes the SAM optimizer.

#         Args:
#             params (iterable): Iterable of parameters to optimize or dicts defining
#                                parameter groups.
#             base_optimizer (torch.optim.Optimizer): The base optimizer (e.g., torch.optim.SGD)
#                                                     that will perform the final weight update.
#             rho (float): The neighborhood radius hyperparameter (p in the proposal).
#             **kwargs: Arguments to be passed to the base_optimizer.
#         """
#         # Initialize the base optimizer using the provided parameters and kwargs
#         # self.param_groups is inherited from the base class
#         super(SAM, self).__init__(params, defaults=dict(rho=rho, **kwargs))
        
#         # Instantiate the base optimizer
#         self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        
#         # We need the param_groups list from the base optimizer for the next steps
#         self.param_groups = self.base_optimizer.param_groups 
        
#         # Shared gradient state storage
#         self.grad_after_perturb = []


#     @torch.no_grad()
#     def _grad_norm(self, weights):
#         """
#         Calculates the L2 norm of the gradient vector across all parameters.
#         This is required for normalizing the perturbation (epsilon).
#         """
#         norm = 0.0
#         for p in weights:
#             # We assume the parameter 'p' has a gradient 'p.grad' attached.
#             if p.grad is not None:
#                 norm += p.grad.norm(2).item() ** 2
#         return norm ** 0.5


#     @torch.no_grad()
#     def perturb_weights(self, closure):
#         """
#         STEP 1: Calculates the perturbation epsilon and applies it to the weights.
        
#         Args:
#             closure (callable): A function that re-evaluates the model and 
#                                 returns the loss (required for the second backward pass).
#         """
#         # Calculate the overall gradient norm
#         grad_norm = self._grad_norm(self.param_groups[0]['params'])
        
#         # A small constant for numerical stability (1e-12 as per proposal pseudocode)
#         epsilon = 1e-12 
        
#         for group in self.param_groups:
#             scale = group['rho'] / (grad_norm + epsilon) # Calculate scale factor
            
#             for p in group['params']:
#                 if p.grad is None:
#                     continue
                
#                 # Store original weight and gradient needed for the next step
#                 state = self.state[p]
#                 state['old_w'] = p.data.clone()
                
#                 # Calculate perturbation vector (epsilon)
#                 # epsilon = rho * grad / (grad_norm + 1e-12)
#                 e_w = p.grad * scale 
                
#                 # Add perturbation to weights: w <- w + epsilon
#                 p.data.add_(e_w)

#         # STEP 2a (Start): Calculate the gradient at the sharpest point (w + epsilon)
#         # This requires a new forward and backward pass, handled by the closure
        
#         # Ensure gradients are zeroed out before running the second backward pass
#         self.zero_grad(set_to_none=True)
        
#         loss = closure()
        
#         # Store the new gradients (gradient at w + epsilon) before resetting
#         self.grad_after_perturb.clear()
#         for group in self.param_groups:
#             for p in group['params']:
#                 if p.grad is None:
#                     continue
#                 self.grad_after_perturb.append(p.grad.data.clone())

#         # STEP 2b: Reset parameters to their original position (w)
#         for group in self.param_groups:
#             for p in group['params']:
#                 if 'old_w' in self.state[p]:
#                     p.data = self.state[p]['old_w']
#                     del self.state[p]['old_w']

#         return loss

    
#     def step(self, closure=None):
#         """
#         Performs a single optimization step (parameter update).
        
#         Args:
#             closure (callable, optional): A closure that reevaluates the model 
#                                          and returns the loss. 
#                                          This is MANDATORY for SAM.
#         """
#         if closure is None:
#             raise ValueError("SAM requires a closure to perform the two forward/backward passes.")
            
#         # The gradient at w is assumed to be calculated BEFORE this function is called:
#         # 1. loss = criterion(model(inputs), targets)
#         # 2. loss.backward()
        
#         # STEP 1 & 2a/2b: Perturb weights, calculate gradient at w+epsilon, and reset weights.
#         self.perturb_weights(closure)

#         # STEP 2c: Call the base optimizer's step() method.
#         # Overwrite the gradient of w with the gradient of w + epsilon.
        
#         grad_idx = 0
#         for group in self.param_groups:
#             for p in group['params']:
#                 if p.grad is None:
#                     continue
#                 # Ensure the gradient is available before copying
#                 if grad_idx < len(self.grad_after_perturb):
#                      p.grad.data.copy_(self.grad_after_perturb[grad_idx])
#                 else:
#                     # Should not happen if all steps are followed correctly
#                     print("Warning: Missing gradient data for SAM update.") 
                    
#                 grad_idx += 1
        
#         # Clear the temporary storage
#         self.grad_after_perturb.clear()

#         # Perform the actual update using the base optimizer (SGD)
#         loss = self.base_optimizer.step()
        
#         return loss

    
#     def zero_grad(self, set_to_none: bool = False):
#         """Zeroes out the gradient buffers for all parameters."""
#         self.base_optimizer.zero_grad(set_to_none=set_to_none)



# import torch
# from torch.optim import Optimizer
# import torch.nn.functional as F

# class SAM(Optimizer):
#     """
#     Sharpness-Aware Minimization (SAM) Optimizer.

#     Implements the two-step procedure of SAM:
#     1. Finds the perturbation (epsilon) that maximizes the loss within 
#        a neighborhood of radius rho.
#     2. Performs the final optimization step using the gradient computed at the 
#        sharpest point (w + epsilon).
    
#     It wraps a base optimizer (e.g., SGD) which handles the actual weight updates.

#     Based on the paper: Sharpness-Aware Minimization for Efficiently Improving Generalization
#     by Foret et al. (2021).
#     """

#     def __init__(self, params, base_optimizer, rho=0.05, **kwargs):
#         """
#         Initializes the SAM optimizer.
        
#         The kwargs are passed directly to the base_optimizer (e.g., lr, momentum, weight_decay).
#         """
#         # Initialize the base optimizer using the provided parameters and kwargs
#         super(SAM, self).__init__(params, defaults=dict(rho=rho, **kwargs))
        
#         # Instantiate the base optimizer
#         self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        
#         # We need the param_groups list from the base optimizer for the next steps
#         self.param_groups = self.base_optimizer.param_groups 
        
#         # This is used to temporarily store the calculated gradient at w + epsilon
#         self.grad_after_perturb = []


#     @torch.no_grad()
#     def _grad_norm(self, weights):
#         """
#         Calculates the L2 norm of the gradient vector across all parameters.
#         """
#         norm = 0.0
#         for p in weights:
#             if p.grad is not None:
#                 norm += p.grad.norm(2).item() ** 2
#         # Add a small epsilon for numerical stability 
#         return norm ** 0.5


#     @torch.no_grad()
#     def perturb_weights(self, closure):
#         """
#         STEP 1: Calculates the perturbation epsilon and applies it to the weights.
        
#         Args:
#             closure (callable): A function that re-evaluates the model and 
#                                 returns the loss (required for the second backward pass).
#         """
        
#         # Calculate the overall gradient norm using the gradient calculated at w
#         grad_norm = self._grad_norm(self.param_groups[0]['params'])
#         epsilon = 1e-12 
        
#         for group in self.param_groups:
#             # Scale factor for normalization
#             scale = group['rho'] / (grad_norm + epsilon)
            
#             for p in group['params']:
#                 if p.grad is None:
#                     continue
                
#                 # Store original weight and gradient
#                 state = self.state[p]
#                 state['old_w'] = p.data.clone()
                
#                 # Calculate perturbation vector (e_w)
#                 e_w = p.grad * scale 
                
#                 # Apply perturbation: w <- w + epsilon
#                 p.data.add_(e_w)

#         # STEP 2a (Start): Calculate the gradient at the sharpest point (w + epsilon)
#         # This requires a new forward and backward pass, handled by the closure
        
#         # CRITICAL: We clear gradients here, as they are now pointing to the 
#         # position w + epsilon, and we need a clean slate for the second backward pass.
#         self.zero_grad(set_to_none=True)
        
#         # Run the closure (Forward + Backward pass at w + epsilon)
#         loss = closure()
        
#         # Store the new gradients (gradient at w + epsilon) before resetting
#         self.grad_after_perturb.clear()
#         for group in self.param_groups:
#             for p in group['params']:
#                 if p.grad is None:
#                     continue
#                 # Store the calculated gradient at w + epsilon
#                 self.grad_after_perturb.append(p.grad.data.clone())

#         # STEP 2b: Reset parameters to their original position (w)
#         for group in self.param_groups:
#             for p in group['params']:
#                 if 'old_w' in self.state[p]:
#                     p.data.copy_(self.state[p]['old_w'])
#                     del self.state[p]['old_w']

#         return loss

    
#     def step(self, closure=None):
#         """
#         Performs a single optimization step (parameter update).
        
#         This step ASSUMES the gradient at the original position (w) has already 
#         been calculated via: loss.backward(retain_graph=True).
#         """
#         if closure is None:
#             raise ValueError("SAM requires a closure to perform the two forward/backward passes.")
            
#         # STEP 1 & 2a/2b: Perturb weights, calculate gradient at w+epsilon, and reset weights.
#         loss = self.perturb_weights(closure)

#         # STEP 2c: Overwrite the gradient of w with the gradient of w + epsilon.
#         # This gradient will be used by the base_optimizer.step()
#         grad_idx = 0
#         for group in self.param_groups:
#             for p in group['params']:
#                 if p.grad is None:
#                     continue
#                 # Overwrite the gradient at w with the gradient at w + epsilon
#                 if grad_idx < len(self.grad_after_perturb):
#                      p.grad.data.copy_(self.grad_after_perturb[grad_idx])
                
#                 grad_idx += 1
        
#         # Clear the temporary storage
#         self.grad_after_perturb.clear()

#         # Perform the actual update using the base optimizer (SGD)
#         self.base_optimizer.step()
        
#         return loss

    
#     def zero_grad(self, set_to_none: bool = False):
#         """Zeroes out the gradient buffers for all parameters, using the base optimizer's method."""
#         self.base_optimizer.zero_grad(set_to_none=set_to_none)



# import torch
# from torch.optim import Optimizer
# import torch.nn.functional as F

# class SAM(Optimizer):
#     """
#     Sharpness-Aware Minimization (SAM) Optimizer.

#     Implements the two-step procedure of SAM:
#     1. Finds the perturbation (epsilon) that maximizes the loss within 
#        a neighborhood of radius rho.
#     2. Performs the final optimization step using the gradient computed at the 
#        sharpest point (w + epsilon).
    
#     It wraps a base optimizer (e.g., SGD) which handles the actual weight updates.

#     Based on the paper: Sharpness-Aware Minimization for Efficiently Improving Generalization
#     by Foret et al. (2021).
#     """

#     def __init__(self, params, base_optimizer, rho=0.05, **kwargs):
#         """
#         Initializes the SAM optimizer.
        
#         The kwargs are passed directly to the base_optimizer (e.g., lr, momentum, weight_decay).
#         """
#         # Initialize the base optimizer using the provided parameters and kwargs
#         super(SAM, self).__init__(params, defaults=dict(rho=rho, **kwargs))
        
#         # Instantiate the base optimizer
#         self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        
#         # We need the param_groups list from the base optimizer for the next steps
#         self.param_groups = self.base_optimizer.param_groups 
        
#         # This is used to temporarily store the calculated gradient at w + epsilon
#         self.grad_after_perturb = []


#     @torch.no_grad()
#     def _grad_norm(self, weights):
#         """
#         Calculates the L2 norm of the gradient vector across all parameters.
#         """
#         norm = 0.0
#         for p in weights:
#             if p.grad is not None:
#                 norm += p.grad.norm(2).item() ** 2
#         # Add a small epsilon for numerical stability 
#         return norm ** 0.5


#     def perturb_weights(self, closure):
#         """
#         STEP 1: Calculates the perturbation epsilon and applies it to the weights.
        
#         Args:
#             closure (callable): A function that re-evaluates the model and 
#                                 returns the loss (required for the second backward pass).
#         """
        
#         # --- PHASE 1: Calculate Perturbation and Move Weights (No Grad Required) ---
#         with torch.no_grad():
#             # Calculate the overall gradient norm using the gradient calculated at w
#             grad_norm = self._grad_norm(self.param_groups[0]['params'])
#             epsilon = 1e-12 
            
#             for group in self.param_groups:
#                 # Scale factor for normalization
#                 scale = group['rho'] / (grad_norm + epsilon)
                
#                 for p in group['params']:
#                     if p.grad is None:
#                         continue
                    
#                     # Store original weight
#                     state = self.state[p]
#                     state['old_w'] = p.data.clone()
                    
#                     # Calculate perturbation vector (e_w)
#                     e_w = p.grad * scale 
                    
#                     # Apply perturbation: w <- w + epsilon
#                     p.data.add_(e_w)

#         # --- PHASE 2: Calculate Gradient at w + epsilon (Grad Required) ---
        
#         # We clear gradients here, as they are now pointing to the 
#         # position w + epsilon, and we need a clean slate for the second backward pass.
#         self.zero_grad(set_to_none=True)
        
#         # Run the closure (Forward + Backward pass at w + epsilon)
#         # Gradient tracking is active here, allowing loss_closure.backward() to work.
#         loss = closure()
        
#         # --- PHASE 3: Store Gradient and Reset Weights (No Grad Required) ---
#         with torch.no_grad():
#             # Store the new gradients (gradient at w + epsilon) before resetting
#             self.grad_after_perturb.clear()
#             for group in self.param_groups:
#                 for p in group['params']:
#                     if p.grad is None:
#                         continue
#                     # Store the calculated gradient at w + epsilon
#                     self.grad_after_perturb.append(p.grad.data.clone())

#             # Reset parameters to their original position (w)
#             for group in self.param_groups:
#                 for p in group['params']:
#                     if 'old_w' in self.state[p]:
#                         p.data.copy_(self.state[p]['old_w'])
#                         del self.state[p]['old_w']

#         return loss

    
#     def step(self, closure=None):
#         """
#         Performs a single optimization step (parameter update).
        
#         This step ASSUMES the gradient at the original position (w) has already 
#         been calculated via: loss.backward(retain_graph=True).
#         """
#         if closure is None:
#             raise ValueError("SAM requires a closure to perform the two forward/backward passes.")
            
#         # STEP 1 & 2a/2b: Perturb weights, calculate gradient at w+epsilon, and reset weights.
#         # This call now handles gradient tracking correctly inside the closure.
#         loss = self.perturb_weights(closure)

#         # STEP 2c: Overwrite the gradient of w with the gradient of w + epsilon.
#         # This is done with torch.no_grad() implicitly in the helper method, but good practice to wrap.
#         with torch.no_grad():
#             grad_idx = 0
#             for group in self.param_groups:
#                 for p in group['params']:
#                     if p.grad is None:
#                         continue
#                     # Overwrite the gradient at w with the gradient at w + epsilon
#                     if grad_idx < len(self.grad_after_perturb):
#                         p.grad.data.copy_(self.grad_after_perturb[grad_idx])
                    
#                     grad_idx += 1
            
#             # Clear the temporary storage
#             self.grad_after_perturb.clear()

#         # Perform the actual update using the base optimizer (SGD)
#         self.base_optimizer.step()
        
#         return loss

    
#     def zero_grad(self, set_to_none: bool = False):
#         """Zeroes out the gradient buffers for all parameters, using the base optimizer's method."""
#         self.base_optimizer.zero_grad(set_to_none=set_to_none)



# import torch
# from torch import optim

# class SAM(optim.Optimizer):
#     """
#     Sharpness-Aware Minimization (SAM) optimizer implementation.
#     Reference: https://arxiv.org/abs/2010.01412
    
#     This optimizer perturbs the weights to find the 'sharpest' point in the
#     neighborhood (of radius rho) and calculates the gradient there.
    
#     This implementation uses the explicit two-step approach (first_step and second_step)
#     to integrate cleanly into the train_one_epoch function of train_sam.py.
#     """
#     def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
#         """
#         Initializes the SAM optimizer.
#         :param params: Model parameters.
#         :param base_optimizer: The underlying PyTorch optimizer (e.g., SGD).
#         :param rho: The neighborhood radius hyperparameter.
#         :param adaptive: If True, uses the ASAM/ESAM adaptive mechanism.
#         """
#         assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

#         # Initialize the optimizer with default SAM parameters and pass kwargs to base optimizer
#         defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
#         super(SAM, self).__init__(params, defaults)

#         # Instantiate the base optimizer using the parameter groups created by SAM's super() call
#         self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
#         self.param_groups = self.base_optimizer.param_groups
#         self.defaults.update(self.base_optimizer.defaults)

#     @torch.no_grad()
#     def first_step(self, zero_grad=False):
#         """
#         Calculates the perturbation e_w and applies it to the weights (w <-- w + e_w).
#         Requires gradients calculated at the original point w.
#         """
#         grad_norm = self._grad_norm()
#         for group in self.param_groups:
#             # Scale factor: rho / ||grad||_2
#             scale = group["rho"] / (grad_norm + 1e-12)

#             for p in group["params"]:
#                 if p.grad is None: continue
                
#                 # Store original weights
#                 self.state[p]["old_p"] = p.data.clone()
                
#                 # Compute the perturbation (e_w)
#                 if group["adaptive"]:
#                     e_w = p.grad * p.data.abs().nan_to_num() * scale
#                 else:
#                     e_w = p.grad * scale
                
#                 # Apply the perturbation: w <--- w + e_w
#                 p.add_(e_w) 

#         if zero_grad: self.zero_grad()

#     @torch.no_grad()
#     def second_step(self, zero_grad=False):
#         """
#         Restores the original weights (w <-- w - e_w) and performs the SGD update 
#         using the gradient calculated at the perturbed point (w + e_w).
#         """
#         # 1. Restore the original parameters: w <--- w - e_w
#         for group in self.param_groups:
#             for p in group["params"]:
#                 if "old_p" not in self.state[p]: continue
#                 p.data = self.state[p]["old_p"]  # Restore w
                
#         # 2. Perform the base optimizer step using the gradient computed at w + e_w
#         # The gradients p.grad now hold the gradient at the perturbed point.
#         self.base_optimizer.step() 

#         if zero_grad: self.zero_grad()

#     @torch.no_grad()
#     def step(self, closure=None):
#         raise NotImplementedError("SAM requires a closure for the two-step gradient calculation.")

#     def _grad_norm(self):
#         """Calculates the L2 norm of the gradient vector across all parameters."""
#         norm = torch.linalg.norm(
#             torch.stack([
#                 p.grad.norm(p=2) * (p.abs() if group["adaptive"] else 1.0)
#                 for group in self.param_groups
#                 for p in group["params"]
#                 if p.grad is not None
#             ]),
#             p=2
#         )
#         return norm




# import torch
# import torch.nn as nn


# class SAM(torch.optim.Optimizer):
#     """
#     Sharpness-Aware Minimization (SAM) Optimizer.
#     Implemented based on the paper: "Sharpness-Aware Minimization for Efficiently Improving Generalization" (arXiv:2010.01412)
#     This implementation uses the explicit two-step approach with first_step() and second_step().
#     """

#     def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
#         """
#         :param params: Model parameters to optimize.
#         :param base_optimizer: The base PyTorch optimizer class (e.g., torch.optim.SGD).
#         :param rho: Non-negative scalar, the neighborhood size (radius).
#         :param adaptive: Boolean, if True, uses Adaptive SAM (ASAM).
#         :param kwargs: Keyword arguments passed to the base_optimizer constructor.
#         """
#         if rho <= 0.0:
#             raise ValueError(f"Invalid rho, must be non-negative: {rho}")

#         # Instantiate the base optimizer with model parameters and kwargs
#         defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
#         super(SAM, self).__init__(params, defaults)

#         # Store the base optimizer instance (SAM uses its step method later)
#         self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
#         self.param_groups = self.base_optimizer.param_groups

#         # Initialize attributes
#         self.rho = rho
#         self.adaptive = adaptive
#         self.state = {} # Stores auxiliary information like the perturbation (e_w)


#     @torch.no_grad()
#     def first_step(self, zero_grad=False):
#         """
#         Calculates the perturbation (e_w), perturbs the weights (w <-- w + e_w), 
#         and stores e_w for the second step.
#         """
#         grad_norm = self._grad_norm()
#         scale = self.rho / (grad_norm + 1e-12)

#         for group in self.param_groups:
#             for p in group["params"]:
#                 if p.grad is None: continue
                
#                 # e_w = scale * grad(L)
#                 e_w = p.grad * scale.to(p.device) # Move scale to parameter device for multiplication
                
#                 # Adaptive SAM calculation (ASAM)
#                 if self.adaptive:
#                     # e_w = rho * (grad(L) * p^2) / ||grad(L) * p^2||
#                     e_w = torch.pow(p, 2) * e_w
                
#                 # Store e_w in the optimizer state
#                 self.state[p] = e_w
                
#                 # Apply perturbation: w <-- w + e_w
#                 p.add_(e_w)

#         if zero_grad:
#             self.base_optimizer.zero_grad() # Zero gradients if requested


#     @torch.no_grad()
#     def second_step(self, zero_grad=False):
#         """
#         Restores the original weights (w <-- w + e_w - e_w), 
#         and performs the standard SGD step using the perturbed gradient grad(w + e_w).
#         """
#         for group in self.param_groups:
#             for p in group["params"]:
#                 if p.grad is None: continue
                
#                 # Retrieve e_w
#                 e_w = self.state.pop(p)
                
#                 # Restore weights: w <-- w - e_w
#                 # The gradient p.grad now holds grad(L) at w + e_w
#                 p.sub_(e_w)

#         # Perform the final SGD step (using the restored weights w and the gradient at w + e_w)
#         self.base_optimizer.step()

#         if zero_grad:
#             self.base_optimizer.zero_grad()


#     @torch.no_grad()
#     def _grad_norm(self):
#         """
#         Computes the L2 norm of the full model gradient.
#         CRITICAL FIX: Explicitly move gradient data to CPU before computing linalg.norm
#         to avoid the specific MPS bug.
#         """
#         shared_device = self.param_groups[0]["params"][0].device # Get the device of the first parameter
        
#         norm = torch.zeros(1).to(shared_device) # Initialize norm on the model's device
        
#         for group in self.param_groups:
#             for p in group["params"]:
#                 if p.grad is None: continue
                
#                 # ASAM uses the adaptive gradient based on p^2
#                 if self.adaptive:
#                     grad = p.grad * torch.pow(p, 2)
#                 else:
#                     grad = p.grad

#                 # --- FIX START ---
#                 # Move the tensor to CPU explicitly before calculating the norm.
#                 # The linalg_norm implementation on MPS has issues, but CPU's is stable.
#                 grad_cpu = grad.to("cpu")
#                 # Calculate the L2 norm (p=2)
#                 grad_norm_part = torch.linalg.norm(grad_cpu, ord=2)
#                 # --- FIX END ---
                
#                 norm.add_(grad_norm_part.pow(2))

#         # Sum the squared norms, take the square root to get the final L2 norm.
#         norm = torch.sqrt(norm)
        
#         return norm


#     # Legacy step function that should raise an error if called incorrectly
#     @torch.no_grad()
#     def step(self, closure=None):
#         raise NotImplementedError("SAM requires first_step() and second_step() to be called explicitly.")




import torch
import torch.nn as nn


class SAM(torch.optim.Optimizer):
    """
    Sharpness-Aware Minimization (SAM) Optimizer.
    Implemented based on the paper: "Sharpness-Aware Minimization for Efficiently Improving Generalization" (arXiv:2010.01412)
    This implementation uses the explicit two-step approach with first_step() and second_step().
    """

    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        """
        :param params: Model parameters to optimize.
        :param base_optimizer: The base PyTorch optimizer class (e.g., torch.optim.SGD).
        :param rho: Non-negative scalar, the neighborhood size (radius).
        :param adaptive: Boolean, if True, uses Adaptive SAM (ASAM).
        :param kwargs: Keyword arguments passed to the base_optimizer constructor.
        """
        if rho <= 0.0:
            raise ValueError(f"Invalid rho, must be non-negative: {rho}")

        # Instantiate the base optimizer with model parameters and kwargs
        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)

        # Store the base optimizer instance (SAM uses its step method later)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

        # Initialize attributes
        self.rho = rho
        self.adaptive = adaptive
        self.state = {} # Stores auxiliary information like the perturbation (e_w)


    @torch.no_grad()
    def first_step(self, zero_grad=False):
        """
        Calculates the perturbation (e_w), perturbs the weights (w <-- w + e_w), 
        and stores e_w for the second step.
        """
        grad_norm = self._grad_norm()
        scale = self.rho / (grad_norm + 1e-12)

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                
                # e_w = scale * grad(L)
                e_w = p.grad * scale.to(p.device) # Move scale to parameter device for multiplication
                
                # Adaptive SAM calculation (ASAM)
                if self.adaptive:
                    # e_w = rho * (grad(L) * p^2) / ||grad(L) * p^2||
                    e_w = torch.pow(p, 2) * e_w
                
                # Store e_w in the optimizer state
                self.state[p] = e_w
                
                # Apply perturbation: w <-- w + e_w
                p.add_(e_w)

        if zero_grad:
            self.base_optimizer.zero_grad() # Zero gradients if requested


    @torch.no_grad()
    def second_step(self, zero_grad=False):
        """
        Restores the original weights (w <-- w + e_w - e_w), 
        and performs the standard SGD step using the perturbed gradient grad(w + e_w).
        """
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                
                # Retrieve e_w
                e_w = self.state.pop(p)
                
                # Restore weights: w <-- w - e_w
                # The gradient p.grad now holds grad(L) at w + e_w
                p.sub_(e_w)

        # Perform the final SGD step (using the restored weights w and the gradient at w + e_w)
        self.base_optimizer.step()

        if zero_grad:
            self.base_optimizer.zero_grad()


    @torch.no_grad()
    def _grad_norm(self):
        """
        Computes the L2 norm of the full model gradient.
        FIX: Use torch.norm() instead of torch.linalg.norm() as it correctly handles 
        tensors of arbitrary dimensions (like 4D conv gradients) when computing the 
        Frobenius norm (p=2). The gradient is still moved to CPU to bypass the 
        original MPS bug.
        """
        shared_device = self.param_groups[0]["params"][0].device # Get the device of the first parameter
        
        # Initialize norm on the model's device
        norm = torch.zeros(1).to(shared_device) 
        
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                
                # ASAM uses the adaptive gradient based on p^2
                if self.adaptive:
                    grad = p.grad * torch.pow(p, 2)
                else:
                    grad = p.grad

                # --- FIX START: Use .norm(p=2) instead of linalg.norm() ---
                # Move the tensor to CPU explicitly before calculating the norm.
                grad_cpu = grad.to("cpu")
                # Calculate the L2 norm (p=2)
                # torch.norm() correctly computes the norm across all dimensions for arbitrary tensors
                grad_norm_part = grad_cpu.norm(p=2) 
                # --- FIX END ---
                
                norm.add_(grad_norm_part.pow(2))

        # Sum the squared norms, take the square root to get the final L2 norm.
        norm = torch.sqrt(norm)
        
        return norm


    # Legacy step function that should raise an error if called incorrectly
    @torch.no_grad()
    def step(self, closure=None):
        raise NotImplementedError("SAM requires first_step() and second_step() to be called explicitly.")