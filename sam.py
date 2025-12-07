import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer, required


class SAM(Optimizer):
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
        # Note: self.param_groups is now managed by the base optimizer.
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        
        # SAM state is managed within the standard PyTorch structure: self.state[p]["e_w"]
        self.rho = rho
        self.adaptive = adaptive


    @torch.no_grad()
    def first_step(self, zero_grad=False):
        """
        Calculates the perturbation (e_w), perturbs the weights (w <-- w + e_w), 
        and stores e_w in the standard optimizer state dict for the second step.
        """
        grad_norm = self._grad_norm()
        scale = self.rho / (grad_norm + 1e-12)

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                
                # Perturbation base (g / ||g||)
                e_w = p.grad * scale.to(p.device)
                
                # Apply Adaptive SAM component (w^2) if needed
                # Correct ASAM perturbation: e_w = scale * (grad * w^2)
                if self.adaptive:
                    e_w = torch.pow(p, 2) * e_w
                
                # Store e_w in the optimizer state (standard PyTorch convention)
                self.state[p]["e_w"] = e_w
                
                # Apply perturbation: w <-- w + e_w
                p.add_(e_w)

        if zero_grad:
            self.base_optimizer.zero_grad()


    @torch.no_grad()
    def second_step(self, zero_grad=False):
        """
        Restores the original weights (w <-- w - e_w), 
        and performs the standard base optimizer step using the perturbed gradient ∇L(w + e_w).
        """
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None or "e_w" not in self.state[p]: continue
                
                # Retrieve and remove e_w from state
                e_w = self.state[p]["e_w"]
                del self.state[p]["e_w"]
                
                # Restore weights: w <-- w - e_w
                # The gradient p.grad now holds ∇L(w + e_w)
                p.sub_(e_w)

        # Perform the final SGD step (using the restored weights w and the gradient at w + e_w)
        self.base_optimizer.step()

        if zero_grad:
            self.base_optimizer.zero_grad()


    @torch.no_grad()
    def _grad_norm(self):
        """
        Computes the L2 norm of the full model gradient (or adaptive gradient).
        Removes the costly .to("cpu") call for performance.
        """
        shared_device = self.param_groups[0]["params"][0].device
        
        # Initialize norm on the model's device
        norm = torch.zeros(1).to(shared_device) 
        
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                
                # ASAM uses the adaptive gradient based on p^2
                if self.adaptive:
                    # grad = ∇L(w) ⊙ w^2
                    grad = p.grad * torch.pow(p, 2)
                else:
                    # grad = ∇L(w)
                    grad = p.grad

                # Calculate the squared L2 norm (Frobenius norm) and accumulate
                # This computation stays on the GPU/MPS device for speed.
                norm.add_(grad.norm(p=2).pow(2))

        # Sum the squared norms, take the square root to get the final L2 norm.
        norm = torch.sqrt(norm)
        
        return norm


    @torch.no_grad()
    def step(self, closure=None):
        """
        Standard step function must raise an error to ensure explicit two-step call.
        """
        raise NotImplementedError("SAM requires first_step() and second_step() to be called explicitly.")