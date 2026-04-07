from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from fl_privacy_tampering.model import TinyLanguageModel, softmax


@dataclass
class LeakageResult:
    canary_loss: float
    control_loss: float
    exposure_gap: float
    target_rank: int


@dataclass
class MembershipInferenceResult:
    member_mean_loss: float
    nonmember_mean_loss: float
    auc: float


def _pair_nll(model: TinyLanguageModel, src: int, dst: int) -> float:
    probs = softmax(model.predict_logits(np.array([src])))[0]
    return float(-np.log(probs[dst] + 1e-12))


def evaluate_canary_leakage(
    model: TinyLanguageModel,
    canary_pair: tuple[int, int],
    control_pair: tuple[int, int],
) -> LeakageResult:
    src, dst = canary_pair
    csrc, cdst = control_pair
    canary_loss = _pair_nll(model, src, dst)
    control_loss = _pair_nll(model, csrc, cdst)

    probs = softmax(model.predict_logits(np.array([src])))[0]
    target_prob = probs[dst]
    rank = int(np.sum(probs > target_prob) + 1)

    return LeakageResult(
        canary_loss=canary_loss,
        control_loss=control_loss,
        exposure_gap=control_loss - canary_loss,
        target_rank=rank,
    )


def _auc_from_scores(labels: np.ndarray, scores: np.ndarray) -> float:
    """
    Computes ROC-AUC from binary labels and real-valued scores.
    Label 1 means "member", and higher score means "more likely member".
    """
    pos_scores = scores[labels == 1]
    neg_scores = scores[labels == 0]
    n_pos = len(pos_scores)
    n_neg = len(neg_scores)
    if n_pos == 0 or n_neg == 0:
        return 0.5

    wins = 0.0
    for s_pos in pos_scores:
        wins += np.sum(s_pos > neg_scores)
        wins += 0.5 * np.sum(s_pos == neg_scores)
    return float(wins / (n_pos * n_neg))


def evaluate_membership_inference_auc(
    model: TinyLanguageModel,
    member_sequences: list[np.ndarray],
    nonmember_sequences: list[np.ndarray],
) -> MembershipInferenceResult:
    """
    Uses per-sequence negative loss as membership score:
    lower loss => higher likelihood sequence was in training.
    """
    member_losses = np.array([model.average_loss(seq) for seq in member_sequences], dtype=float)
    nonmember_losses = np.array([model.average_loss(seq) for seq in nonmember_sequences], dtype=float)

    labels = np.concatenate(
        [
            np.ones(len(member_losses), dtype=int),
            np.zeros(len(nonmember_losses), dtype=int),
        ]
    )
    scores = -np.concatenate([member_losses, nonmember_losses])
    auc = _auc_from_scores(labels=labels, scores=scores)

    return MembershipInferenceResult(
        member_mean_loss=float(np.mean(member_losses)),
        nonmember_mean_loss=float(np.mean(nonmember_losses)),
        auc=auc,
    )
