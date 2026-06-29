import torch
import torch.nn as nn

class DeepIRCompositeLoss(nn.Module):
    """
    The Composite Loss Framework acting as strict teachers
    and truth guardrails to evaluate the pipeline.
    """
    def __init__(self, alpha=1.0, gamma=0.5):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

        # Teacher 1: Pixel-Level Loss (Absolute Error)
        self.pixel_loss_teacher = nn.L1Loss()

        # Truth Guardrail: Semantic Categorization Loss (Cross Entropy)
        self.truth_guardrail = nn.CrossEntropyLoss()

    def forward(self, pred_ir, target_ir, pred_semantic_mask, target_semantic_mask):
        # Calculate individual errors
        loss_pixel = self.pixel_loss_teacher(pred_ir, target_ir)
        loss_semantic = self.truth_guardrail(pred_semantic_mask, target_semantic_mask)

        # Total composite penalty
        total_loss = (self.alpha * loss_pixel) + (self.gamma * loss_semantic)

        return total_loss
