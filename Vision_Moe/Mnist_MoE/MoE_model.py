from torch import nn, stack, bmm
import torch
import torch.nn.functional as F

class MoE_Model(nn.Module):
    def __init__(self, experts, gating_network, k_training=2, k_inference=1, tau=0.8, lambda_balance=0.05):
        super(MoE_Model, self).__init__()
        self.experts = nn.ModuleList(experts)
        self.gating_network = gating_network
        
        self.tau = tau
        self.lambda_balance = lambda_balance 
        self.k_training = k_training
        self.k_inference = k_inference

    def forward(self,
                x, 
                k=None, 
                selection_policy='soft', 
                gumbell_softmax=True, 
                threshold=None):
        """
        Training forward pass.
        Returns the output and the auxiliary balancing loss.
        """
        logits = self.gating_network(x)  # (B, N)
        
        # Add standard gaussian noise to logits during training to encourage exploration
        if self.training:
            noise = torch.randn_like(logits) * 0.1 
            logits = logits + noise
            
        num_experts = len(self.experts)
        
        k = k if k is not None else self.k_training
        k = min(k, num_experts)

        probs = self.experts_selection(logits, k=k, selection_policy=selection_policy, gumbell_softmax=gumbell_softmax, threshold=threshold)
        

        aux_loss = self.compute_balance_loss(logits, probs, k)
        
        # Dense execution for training stability
        expert_outputs = stack([e(x) for e in self.experts], dim=2)
        output = bmm(expert_outputs, probs.unsqueeze(1).transpose(1, 2)).squeeze(2)

        return output, aux_loss

    def inference(self,
                x,
                k=None, 
                selection_policy='hard', 
                gumbell_softmax=True,
                verbose=False,
                threshold=None):
        """
        Inference pass (Sparse execution).
        """
        B = x.size(0)
        num_experts = len(self.experts)
        k = k if k is not None else self.k_inference
        
        logits = self.gating_network(x)
        probs = self.experts_selection(logits, k=k, selection_policy=selection_policy, gumbell_softmax=gumbell_softmax, threshold=threshold)

        # Initialize output buffer
        final_output = torch.zeros(B, 10, device=x.device, dtype=x.dtype) # Assuming 10 classes for MNIST

        for i in range(num_experts):
            mask = probs[:, i] > 0
            if mask.any():
                active_indices = torch.where(mask)[0]
                expert_out = self.experts[i](x[active_indices])
                
                weights = probs[active_indices, i].unsqueeze(1)
                # Accumulate weighted expert outputs
                final_output[active_indices] += expert_out * weights

        return final_output, probs if verbose else final_output

    def experts_selection(self, logits, k=2, selection_policy='hard', gumbell_softmax=True, threshold=None):
        """
        Selection mask with Straight-Through Estimator support for 'top_k'.
        selection_policy: 'hard', 'soft', 'hybrid'
        gumbell_softmax: whether to use softmax or gumbel_softmax
        """
        if gumbell_softmax:
            probs = F.gumbel_softmax(logits, tau=self.tau, dim=1)
        else:
            probs = F.softmax(logits / self.tau, dim=1)

        if selection_policy == 'hard':
            _, topk_indices = torch.topk(probs, k=k, dim=1)
            mask = torch.zeros_like(probs).scatter_(1, topk_indices, 1.0)
            return (mask - probs).detach() + probs

        elif selection_policy == 'soft':
            return probs 

        elif selection_policy == 'hybrid':
            if threshold is None:
                row_mean = probs.mean(dim=1, keepdim=True)
                row_std = probs.std(dim=1, keepdim=True)
                threshold = (row_mean + 0.5 * row_std)
            
            mask = (probs >= threshold).float()
            
            # Ensure at least one expert is selected per sample
            if mask.sum(dim=1).min() == 0:
                max_indices = torch.argmax(probs, dim=1, keepdim=True)
                mask = torch.max(mask, torch.zeros_like(mask).scatter_(1, max_indices, 1.0))
            
            masked_probs = probs * mask
            renormalized_probs = masked_probs / (masked_probs.sum(dim=1, keepdim=True).clamp_min(1e-12))
            return renormalized_probs

        else:
            raise ValueError(f"Unknown selection policy: {selection_policy}")

    def compute_balance_loss(self, logits, probs, k=None):
        """
        Improved Load Balancing Loss that considers actual weights.
        """
        router_probs = F.softmax(logits / self.tau, dim=1)
        N = router_probs.size(1)
        
        importance = router_probs.mean(dim=0)
        
        
        load = probs.mean(dim=0)
        
        
        importance_cv = importance.std() / (importance.mean() + 1e-8)
        load_cv = load.std() / (load.mean() + 1e-8)
        
        switch_loss = N * torch.sum(importance * load)
        cv_loss = importance_cv + load_cv
        
        return (switch_loss + cv_loss) * self.lambda_balance

    def experts_infos(self):
        experts_list = []
        for idx, expert in enumerate(self.experts):
            experts_list.append(f"Expert {idx+1}: {expert.__class__.__name__}")
        return ", ".join(experts_list)