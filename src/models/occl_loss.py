import torch
import torch.nn as nn


class OCCLLoss(nn.Module):
    def __init__(self, embed_dim=128, margin=1.0, lam=0.1, center_momentum=0.9):
        super().__init__()
        self.margin = margin
        self.lam = lam
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(embed_dim))

    @torch.no_grad()
    def init_center(self, embeddings):
        c = embeddings.mean(dim=0)
        c[(c.abs() < 1e-6)] = 1e-6
        self.center.copy_(c)

    @torch.no_grad()
    def update_center(self, genuine_embeddings):
        batch_mean = genuine_embeddings.mean(dim=0)
        self.center.mul_(self.center_momentum).add_(
            batch_mean, alpha=1.0 - self.center_momentum
        )

    def forward(self, z_genuine, z_anomaly):
        c = self.center

        dist_genuine = torch.sum((z_genuine - c) ** 2, dim=1)
        loss_attract = dist_genuine.mean()

        if z_anomaly.shape[0] > 0:
            dist_anomaly = torch.sqrt(torch.sum((z_anomaly - c) ** 2, dim=1) + 1e-8)
            loss_repel = torch.clamp(self.margin - dist_anomaly, min=0.0) ** 2
            loss_repel = loss_repel.mean()
        else:
            loss_repel = torch.tensor(0.0, device=z_genuine.device)

        return loss_attract + self.lam * loss_repel


if __name__ == "__main__":
    criterion = OCCLLoss(embed_dim=128)
    z_g = torch.randn(16, 128) * 0.1
    z_a = torch.randn(16, 128) * 0.5 + 2.0
    criterion.init_center(z_g)
    loss = criterion(z_g, z_a)
    print(f"loss={loss.item():.6f}  ||c||={criterion.center.norm():.4f}")
