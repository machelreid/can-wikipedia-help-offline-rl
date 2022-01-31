import numpy as np
import torch
import torch.nn.functional as F

from decision_transformer.training.trainer import Trainer
import torch_semiring_einsum


EQUATION = torch_semiring_einsum.compile_equation("iaj,bj->iabj")


def kmeans_cosine_max_loss(centers, seq, mean=False):
    assert centers.device == seq.device
    # loss = -(torch.einsum("iaj,bj->iabj", [seq, centers]).max(2).values.mean())
    if mean:
        loss = -(
            torch_semiring_einsum.einsum(EQUATION, seq, centers, block_size=5).mean()
        )
    else:
        loss = -(
            torch_semiring_einsum.einsum(EQUATION, seq, centers, block_size=5)
            .max(2)
            .values.mean()
        )


    return loss


kmeans_anneal = lambda x: 1 / (1 + np.exp(-(((5000 - x) / (5000 / 10)) - 5)))


class SequenceTrainer(Trainer):
    def train_step(self):
        (
            states,
            actions,
            rewards,
            dones,
            rtg,
            timesteps,
            attention_mask,
        ) = self.get_batch(self.batch_size)
        action_target = torch.clone(actions)

        state_preds, action_preds, reward_preds, all_embs = self.model.forward(
            states,
            actions,
            rewards,
            rtg[:, :-1],
            timesteps,
            attention_mask=attention_mask,
        )

        self.step += 1
        act_dim = action_preds.shape[2]
        action_preds = action_preds.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        action_target = action_target.reshape(-1, act_dim)[
            attention_mask.reshape(-1) > 0
        ]

        loss = self.loss_fn(
            None,
            action_preds,
            None,
            None,
            action_target,
            None,
        )
        if self.args["gpt_kmeans"]:
            loss += (
                self.args["gpt_kmeans_const"]
                * kmeans_anneal(self.step)
                * kmeans_cosine_max_loss(
                    F.normalize(self.model.cluster_centers, dim=-1),
                    F.normalize(all_embs, dim=-1),
                    mean=self.args["kmeans_mean"],
                )
            )

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.25)
        self.optimizer.step()

        with torch.no_grad():
            self.diagnostics["training/action_error"] = (
                torch.mean((action_preds - action_target) ** 2).detach().cpu().item()
            )

        return loss.detach().cpu().item()
