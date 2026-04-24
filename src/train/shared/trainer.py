import json
import time
from dataclasses import asdict

import numpy as np
import torch
from tqdm import tqdm

from src.models.occl_loss import OCCLLoss
from src.train.shared.config import TrainConfig
from src.train.shared.utils import count_params, make_loader


class Trainer:
    def __init__(self, model, cfg):
        self.model = model
        self.cfg = cfg
        self.device = next(model.parameters()).device
        self.n_params, self.n_trainable = count_params(model)

        self.criterion = OCCLLoss(
            embed_dim=cfg.embed_dim, margin=cfg.margin, lam=cfg.lam,
            center_momentum=cfg.center_ema_momentum,
        ).to(self.device)

        self.optimiser = torch.optim.Adam(
            model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimiser, T_max=cfg.epochs, eta_min=1e-6,
        )

        self.train_losses = []
        self.val_losses = []
        self.epoch_times = []
        self.best_val_loss = float("inf")
        self.patience_counter = 0

    def _make_train_loader(self):
        loader, n = make_loader(
            self.cfg.processed_dir / "train.pt",
            batch_size=self.cfg.batch_size,
            pseudo_anomaly_ratio=self.cfg.pseudo_anomaly_ratio,
            shuffle=True, num_workers=self.cfg.num_workers, drop_last=True,
        )
        print(f"Train: {n:,} samples ({len(loader):,} batches)")
        return loader

    def _make_val_loader(self):
        loader, n = make_loader(
            self.cfg.processed_dir / "val.pt",
            batch_size=1024, num_workers=self.cfg.num_workers,
            shuffle=True,
        )
        print(f"Val:   {n:,} samples")
        return loader

    def _init_center(self):
        print(f"[{self.cfg.model_name}] Initialising center from all genuine embeddings...")
        self.model.eval()
        loader, _ = make_loader(
            self.cfg.processed_dir / "train.pt",
            batch_size=1024, num_workers=self.cfg.num_workers,
        )
        embeds = []
        with torch.no_grad():
            for x, _ in tqdm(loader, desc="Center init", leave=False, ncols=100):
                embeds.append(self.model(x.to(self.device)).cpu())
        embeds = torch.cat(embeds, dim=0)
        self.criterion.init_center(embeds)
        print(f"Center initialised from {embeds.shape[0]:,} embeddings "
              f"(||c|| = {self.criterion.center.norm():.4f})")

    def _train_one_epoch(self, train_loader, epoch):
        name = self.cfg.model_name
        self.model.train()
        epoch_loss, n = 0.0, 0

        pbar = tqdm(train_loader, desc=f"[{name}] Epoch {epoch:>3d}/{self.cfg.epochs}",
                    leave=False, ncols=120)
        for x, y in pbar:
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)

            z = self.model(x)
            z_gen, z_anom = z[y == 1], z[y == 0]
            if z_gen.shape[0] == 0:
                continue

            loss = self.criterion(z_gen, z_anom)
            self.optimiser.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimiser.step()
            self.criterion.update_center(z_gen.detach())

            epoch_loss += loss.item()
            n += 1
            pbar.set_postfix(loss=f"{epoch_loss / n:.6f}")
        pbar.close()
        return epoch_loss / max(n, 1)

    @torch.no_grad()
    def _compute_val_loss(self, val_loader):
        self.model.eval()
        total, n = 0.0, 0
        for x, y in val_loader:
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
            z = self.model(x)
            mask = y == 1
            if mask.sum() == 0:
                continue
            loss = self.criterion(z[mask], z[y == 0])
            total += loss.item() * mask.sum().item()
            n += mask.sum().item()
        return total / max(n, 1)

    def _save_checkpoint(self, epoch, val_loss):
        path = self.cfg.results_dir / self.cfg.ckpt_filename
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "loss": val_loss,
            "center": self.criterion.center.data.clone(),
            "hparams": asdict(self.cfg),
            "n_params": self.n_params,
            "model_name": self.cfg.model_name,
        }, path)

    def _save_loss_history(self):
        path = self.cfg.metrics_dir / self.cfg.loss_filename
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump({
                "train": self.train_losses,
                "val": self.val_losses,
                "epoch_times_sec": self.epoch_times,
            }, f)
        print(f"Loss history saved to {path}")

    def _print_summary(self, total_time):
        print("=" * 70)
        print("Training complete.")
        print(f"  Model:             {self.cfg.model_name}")
        print(f"  Total params:      {self.n_params:,}")
        print(f"  Trainable params:  {self.n_trainable:,}")
        print(f"  Epochs trained:    {len(self.train_losses)}")
        print(f"  Best val loss:     {self.best_val_loss:.6f}")
        print(f"  Avg epoch time:    {np.mean(self.epoch_times):.1f}s")
        print(f"  Total train time:  {total_time:.1f}s ({total_time / 60:.1f} min)")
        print(f"  Checkpoint:        {self.cfg.results_dir / self.cfg.ckpt_filename}")
        print("=" * 70)

    def run(self):
        name = self.cfg.model_name
        print(f"Device: {self.device}")
        print(f"Config: {json.dumps(asdict(self.cfg), indent=2, default=str)}")
        print(f"{name} — total params: {self.n_params:,}  trainable: {self.n_trainable:,}")

        train_loader = self._make_train_loader()
        val_loader = self._make_val_loader()
        self._init_center()

        total_start = time.time()

        for epoch in range(1, self.cfg.epochs + 1):
            t0 = time.time()

            avg_train = self._train_one_epoch(train_loader, epoch)
            self.scheduler.step()
            self.train_losses.append(avg_train)

            val_loss = self._compute_val_loss(val_loader)
            self.val_losses.append(val_loss)

            epoch_time = time.time() - t0
            self.epoch_times.append(epoch_time)
            total_elapsed = time.time() - total_start

            print(
                f"[{name}] Epoch {epoch:>3d}/{self.cfg.epochs}  |  "
                f"train={avg_train:.6f}  val={val_loss:.6f}  |  "
                f"lr={self.scheduler.get_last_lr()[0]:.2e}  "
                f"||c||={self.criterion.center.norm():.4f}  |  "
                f"epoch={epoch_time:.1f}s  total={total_elapsed:.0f}s"
            )

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self._save_checkpoint(epoch, val_loss)
                print(f"  -> Best model saved (val_loss={self.best_val_loss:.6f})")
            else:
                self.patience_counter += 1
                print(f"  -> No improvement ({self.patience_counter}/{self.cfg.patience})")
                if self.patience_counter >= self.cfg.patience:
                    print(f"  -> Early stopping after {epoch} epochs")
                    break

        total_time = time.time() - total_start
        self._save_loss_history()
        self._print_summary(total_time)

        return self.model, self.criterion, self.train_losses
