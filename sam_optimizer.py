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



import torch
from torch.optim import Optimizer
import torch.nn.functional as F

class SAM(Optimizer):
    """
    Sharpness-Aware Minimization (SAM) Optimizer.

    Implements the two-step procedure of SAM:
    1. Finds the perturbation (epsilon) that maximizes the loss within 
       a neighborhood of radius rho.
    2. Performs the final optimization step using the gradient computed at the 
       sharpest point (w + epsilon).
    
    It wraps a base optimizer (e.g., SGD) which handles the actual weight updates.

    Based on the paper: Sharpness-Aware Minimization for Efficiently Improving Generalization
    by Foret et al. (2021).
    """

    def __init__(self, params, base_optimizer, rho=0.05, **kwargs):
        """
        Initializes the SAM optimizer.
        
        The kwargs are passed directly to the base_optimizer (e.g., lr, momentum, weight_decay).
        """
        # Initialize the base optimizer using the provided parameters and kwargs
        super(SAM, self).__init__(params, defaults=dict(rho=rho, **kwargs))
        
        # Instantiate the base optimizer
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        
        # We need the param_groups list from the base optimizer for the next steps
        self.param_groups = self.base_optimizer.param_groups 
        
        # This is used to temporarily store the calculated gradient at w + epsilon
        self.grad_after_perturb = []


    @torch.no_grad()
    def _grad_norm(self, weights):
        """
        Calculates the L2 norm of the gradient vector across all parameters.
        """
        norm = 0.0
        for p in weights:
            if p.grad is not None:
                norm += p.grad.norm(2).item() ** 2
        # Add a small epsilon for numerical stability 
        return norm ** 0.5


    def perturb_weights(self, closure):
        """
        STEP 1: Calculates the perturbation epsilon and applies it to the weights.
        
        Args:
            closure (callable): A function that re-evaluates the model and 
                                returns the loss (required for the second backward pass).
        """
        
        # --- PHASE 1: Calculate Perturbation and Move Weights (No Grad Required) ---
        with torch.no_grad():
            # Calculate the overall gradient norm using the gradient calculated at w
            grad_norm = self._grad_norm(self.param_groups[0]['params'])
            epsilon = 1e-12 
            
            for group in self.param_groups:
                # Scale factor for normalization
                scale = group['rho'] / (grad_norm + epsilon)
                
                for p in group['params']:
                    if p.grad is None:
                        continue
                    
                    # Store original weight
                    state = self.state[p]
                    state['old_w'] = p.data.clone()
                    
                    # Calculate perturbation vector (e_w)
                    e_w = p.grad * scale 
                    
                    # Apply perturbation: w <- w + epsilon
                    p.data.add_(e_w)

        # --- PHASE 2: Calculate Gradient at w + epsilon (Grad Required) ---
        
        # We clear gradients here, as they are now pointing to the 
        # position w + epsilon, and we need a clean slate for the second backward pass.
        self.zero_grad(set_to_none=True)
        
        # Run the closure (Forward + Backward pass at w + epsilon)
        # Gradient tracking is active here, allowing loss_closure.backward() to work.
        loss = closure()
        
        # --- PHASE 3: Store Gradient and Reset Weights (No Grad Required) ---
        with torch.no_grad():
            # Store the new gradients (gradient at w + epsilon) before resetting
            self.grad_after_perturb.clear()
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is None:
                        continue
                    # Store the calculated gradient at w + epsilon
                    self.grad_after_perturb.append(p.grad.data.clone())

            # Reset parameters to their original position (w)
            for group in self.param_groups:
                for p in group['params']:
                    if 'old_w' in self.state[p]:
                        p.data.copy_(self.state[p]['old_w'])
                        del self.state[p]['old_w']

        return loss

    
    def step(self, closure=None):
        """
        Performs a single optimization step (parameter update).
        
        This step ASSUMES the gradient at the original position (w) has already 
        been calculated via: loss.backward(retain_graph=True).
        """
        if closure is None:
            raise ValueError("SAM requires a closure to perform the two forward/backward passes.")
            
        # STEP 1 & 2a/2b: Perturb weights, calculate gradient at w+epsilon, and reset weights.
        # This call now handles gradient tracking correctly inside the closure.
        loss = self.perturb_weights(closure)

        # STEP 2c: Overwrite the gradient of w with the gradient of w + epsilon.
        # This is done with torch.no_grad() implicitly in the helper method, but good practice to wrap.
        with torch.no_grad():
            grad_idx = 0
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is None:
                        continue
                    # Overwrite the gradient at w with the gradient at w + epsilon
                    if grad_idx < len(self.grad_after_perturb):
                        p.grad.data.copy_(self.grad_after_perturb[grad_idx])
                    
                    grad_idx += 1
            
            # Clear the temporary storage
            self.grad_after_perturb.clear()

        # Perform the actual update using the base optimizer (SGD)
        self.base_optimizer.step()
        
        return loss

    
    def zero_grad(self, set_to_none: bool = False):
        """Zeroes out the gradient buffers for all parameters, using the base optimizer's method."""
        self.base_optimizer.zero_grad(set_to_none=set_to_none)