import copy
import sys
from typing import Any, Dict, Tuple

import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import dropout_edge

from src.entity import ModelArchitectureConfig, ModelTrainerConfig
from src.models.gat import AdvancedCoraGAT
from utils.custom_exception import CustomException
from utils.helpers import get_device, set_seed
from utils.logger import logger


class ModelTrainer:
    """Trains the Advanced GATv2 model on graph data with edge dropout and early stopping."""

    def __init__(
        self,
        trainer_config: ModelTrainerConfig,
        arch_config: ModelArchitectureConfig,
        dataset: Planetoid,
        data: Data,
    ):
        self.config = trainer_config
        self.arch_config = arch_config
        self.dataset = dataset
        self.data = data
        self.device = get_device(self.config.device)

    def _train_one_epoch(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        data: Data,
    ) -> float:
        """Runs a single training epoch with edge dropout and gradient clipping."""
        model.train()
        optimizer.zero_grad()

        # Random edge dropout for regularization
        train_edge_index, _ = dropout_edge(
            data.edge_index,
            p=self.config.edge_dropout_p,
            force_undirected=False,
        )

        logits = model(data.x, train_edge_index)
        loss = F.cross_entropy(
            logits[data.train_mask],
            data.y[data.train_mask],
            label_smoothing=self.config.label_smoothing,
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=self.config.max_grad_norm,
        )
        optimizer.step()
        return loss.item()

    @torch.no_grad()
    def _evaluate_split(
        self,
        model: torch.nn.Module,
        data: Data,
        mask: torch.Tensor,
    ) -> Tuple[float, float, torch.Tensor, torch.Tensor]:
        """Evaluates model performance on a specified node mask."""
        model.eval()
        logits = model(data.x, data.edge_index)
        predictions = logits.argmax(dim=1)

        y_true = data.y[mask].cpu().numpy()
        y_pred = predictions[mask].cpu().numpy()

        accuracy = accuracy_score(y_true, y_pred)
        macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        return float(accuracy), float(macro_f1), predictions, logits

    def train(self) -> Tuple[torch.nn.Module, Dict[str, Any]]:
        """Executes the full training loop with validation scheduling and early stopping.

        Returns:
            Tuple[torch.nn.Module, Dict[str, Any]]: The trained model and training history.
        """
        try:
            set_seed(self.config.seed)
            logger.info("Initializing AdvancedCoraGAT model...")

            data = self.data.to(self.device)
            input_dim = data.num_node_features
            num_classes = self.dataset.num_classes

            model = AdvancedCoraGAT(
                input_dim=input_dim,
                hidden_dim=self.arch_config.hidden_dim,
                num_classes=num_classes,
                num_layers=self.arch_config.num_layers,
                heads=self.arch_config.heads,
                dropout=self.arch_config.dropout,
                attention_dropout=self.arch_config.attention_dropout,
                classifier_dropout_1=self.arch_config.classifier_dropout_1,
                classifier_dropout_2=self.arch_config.classifier_dropout_2,
                jk_mode=self.arch_config.jk_mode,
            ).to(self.device)

            trainable_params = sum(
                p.numel() for p in model.parameters() if p.requires_grad
            )
            logger.info(f"Model initialized with {trainable_params:,} trainable parameters.")

            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )

            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="max",
                factor=self.config.lr_scheduler_factor,
                patience=self.config.lr_scheduler_patience,
                min_lr=self.config.lr_scheduler_min_lr,
            )

            train_losses = []
            train_accs = []
            val_accs = []
            val_f1s = []
            learning_rates = []

            best_val_acc = 0.0
            best_val_f1 = 0.0
            best_state = None
            best_epoch = 0
            epochs_without_improvement = 0

            logger.info(
                f"Starting training for {self.config.epochs} epochs (Patience: {self.config.patience})..."
            )

            for epoch in range(1, self.config.epochs + 1):
                loss = self._train_one_epoch(model, optimizer, data)
                train_acc, train_f1, _, _ = self._evaluate_split(
                    model, data, data.train_mask
                )
                val_acc, val_f1, _, _ = self._evaluate_split(
                    model, data, data.val_mask
                )

                scheduler.step(val_acc)
                current_lr = optimizer.param_groups[0]["lr"]

                train_losses.append(loss)
                train_accs.append(train_acc)
                val_accs.append(val_acc)
                val_f1s.append(val_f1)
                learning_rates.append(current_lr)

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_val_f1 = val_f1
                    best_state = copy.deepcopy(model.state_dict())
                    best_epoch = epoch
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1

                if epoch == 1 or epoch % self.config.log_every_n_epochs == 0:
                    logger.info(
                        f"Epoch {epoch:03d}/{self.config.epochs:03d} | "
                        f"Loss {loss:.4f} | Train Acc {train_acc:.4f} | "
                        f"Val Acc {val_acc:.4f} | Val F1 {val_f1:.4f} | LR {current_lr:.6f}"
                    )

                if epochs_without_improvement >= self.config.patience:
                    logger.info(
                        f"Early stopping triggered at epoch {epoch}. "
                        f"No improvement for {self.config.patience} epochs."
                    )
                    break

            if best_state is not None:
                model.load_state_dict(best_state)
                logger.info(
                    f"Restored best weights from epoch {best_epoch} "
                    f"(Best Val Acc: {best_val_acc:.4f}, Best Val F1: {best_val_f1:.4f})"
                )

            # Save PyTorch model artifact
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "input_dim": input_dim,
                "hidden_dim": self.arch_config.hidden_dim,
                "num_classes": num_classes,
                "num_layers": self.arch_config.num_layers,
                "heads": self.arch_config.heads,
                "dropout": self.arch_config.dropout,
                "attention_dropout": self.arch_config.attention_dropout,
                "best_epoch": best_epoch,
                "best_val_acc": float(best_val_acc),
                "best_val_f1": float(best_val_f1),
            }
            torch.save(checkpoint, str(self.config.model_path))
            logger.info(f"Saved PyTorch model checkpoint at: {self.config.model_path}")

            history = {
                "train_losses": train_losses,
                "train_accs": train_accs,
                "val_accs": val_accs,
                "val_f1s": val_f1s,
                "learning_rates": learning_rates,
                "best_epoch": best_epoch,
                "best_val_acc": float(best_val_acc),
                "best_val_f1": float(best_val_f1),
            }

            return model, history
        except Exception as e:
            logger.error("Failed during model training")
            raise CustomException(e, sys)
