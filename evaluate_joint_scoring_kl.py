"""
Evaluate a trained JointScoringModel on images + radiomics (wide format).

Saves:
- predictions_eval.csv (case_id, pred_class, prob_0..prob_4)
- selected_subsets_eval.json (TopM subset indices + names per case)
- metrics_eval.json (if labels provided)


python evaluate_joint_scoring_kl.py ^
  --model-dir training_logs_joint_scoring_k30 ^
  --images-dir "C:\Users\chris\MICCAI2026\nnUNet\nnUNet_raw\Dataset360_oaizib\imagesTs" ^
  --radiomics-csv "C:\Users\chris\MICCAI2026\YaxiiC-OA_KLG_Retrieval\output_test\radiomics_results_wide.csv" ^
  --klgrade-csv "C:\Users\chris\MICCAI2026\YaxiiC-OA_KLG_Retrieval\subInfo_test.xlsx" ^
  --device cuda:0
"""

import argparse
import json
import logging
import random
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data_loader import KLGradeDataset, load_klgrade_labels, load_radiomics_wide_format
from models import JointScoringModel
from training_utils import build_subset_tensors, compute_metrics, sample_random_subsets

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_feature_names(feature_names_path: Path) -> Optional[List[str]]:
    if not feature_names_path.exists():
        return None
    with open(feature_names_path, "r") as f:
        mapping = json.load(f)
    return [mapping[str(i)] for i in sorted(map(int, mapping.keys()))]


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate JointScoringModel on images + radiomics",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--model-dir", type=str, required=True, help="Training output dir")
    parser.add_argument("--images-dir", type=str, required=True, help="Images directory")
    parser.add_argument("--radiomics-csv", type=str, required=True, help="Radiomics CSV (wide format)")
    parser.add_argument("--klgrade-csv", type=str, default=None, help="Optional KLGrade labels")
    parser.add_argument("--outdir", type=str, default=None, help="Output dir (defaults to model-dir)")
    parser.add_argument("--checkpoint-name", type=str, default="best.pth", help="Checkpoint name")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--target-shape", type=int, nargs=3, default=[32, 128, 128], help="D H W")
    parser.add_argument("--pool-size", type=int, default=None, help="Candidate pool size")
    parser.add_argument("--top-m", type=int, default=None, help="TopM subsets for ensemble")
    parser.add_argument("--k", type=int, default=None, help="Subset size K")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--label-mode", type=str, default=None,
                        choices=["multiclass", "binary_oa"],
                        help="Override label mode (default: from checkpoint)")

    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    outdir = Path(args.outdir) if args.outdir else model_dir
    outdir.mkdir(parents=True, exist_ok=True)

    ckpt_dir = model_dir / "checkpoints"
    ckpt_path = ckpt_dir / args.checkpoint_name
    if not ckpt_path.exists():
        fallback = ckpt_dir / "last.pth"
        if fallback.exists():
            logger.warning(f"{args.checkpoint_name} not found; using {fallback.name}")
            ckpt_path = fallback
        else:
            raise FileNotFoundError(f"No checkpoint found in {ckpt_dir}")

    feature_names_path = ckpt_dir / "feature_names.json"
    scaler_path = ckpt_dir / "scaler.joblib"

    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    logger.info(f"Using device: {device}")

    checkpoint = torch.load(ckpt_path, map_location=device)
    ckpt_args = checkpoint.get("args", {})

    k = args.k if args.k is not None else ckpt_args.get("k", 15)
    pool_size = args.pool_size if args.pool_size is not None else ckpt_args.get("pool_size", 320)
    top_m = args.top_m if args.top_m is not None else ckpt_args.get("top_m", 4)

    def infer_num_classes(state_dict: Dict[str, torch.Tensor]) -> int:
        for key in state_dict:
            if key.endswith("classifier.mlp.3.weight"):
                return state_dict[key].shape[0]
        for key in state_dict:
            if key.endswith("classifier.mlp.2.weight") or key.endswith("classifier.mlp.1.weight"):
                return state_dict[key].shape[0]
        raise KeyError("Could not infer num_classes from checkpoint state_dict.")

    num_classes = checkpoint.get("num_classes")
    if num_classes is None:
        num_classes = infer_num_classes(checkpoint["model_state_dict"])
    label_mode = args.label_mode or ckpt_args.get("label_mode")
    if label_mode is None:
        label_mode = "binary_oa" if num_classes == 2 else "multiclass"
    logger.info(f"Label mode: {label_mode} | num_classes={num_classes}")

    # Load radiomics (wide format)
    radiomics, feature_names, _, feature_meta = load_radiomics_wide_format(
        Path(args.radiomics_csv)
    )

    # Validate feature order with training mapping if available
    trained_feature_names = load_feature_names(feature_names_path)
    if trained_feature_names and len(trained_feature_names) != len(feature_names):
        logger.warning(
            "Feature count mismatch (train vs eval): "
            f"{len(trained_feature_names)} vs {len(feature_names)}"
        )

    # Apply scaler if available
    if scaler_path.exists():
        scaler = joblib.load(scaler_path)
        for cid in list(radiomics.keys()):
            radiomics[cid] = scaler.transform(radiomics[cid].reshape(1, -1))[0]
    else:
        logger.warning("Scaler not found; proceeding without feature scaling.")

    labels_dict = None
    if args.klgrade_csv:
        labels_dict_raw = load_klgrade_labels(Path(args.klgrade_csv))
        if label_mode == "binary_oa":
            labels_dict = {cid: (0 if lab <= 1 else 1) for cid, lab in labels_dict_raw.items()}
        else:
            labels_dict = labels_dict_raw
        logger.info(f"Loaded {len(labels_dict)} labels")

    case_ids = list(radiomics.keys())
    if labels_dict:
        case_ids = [cid for cid in case_ids if cid in labels_dict]
        logger.info(f"Cases with labels: {len(case_ids)}")
    else:
        logger.info(f"Cases for inference: {len(case_ids)}")

    dataset = KLGradeDataset(
        case_ids,
        Path(args.images_dir),
        radiomics,
        labels_dict=labels_dict,
        target_shape=tuple(args.target_shape)
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if device.type == "cuda" else False
    )

    num_features = checkpoint.get("num_features", len(feature_names))
    feature_to_roi = np.array(feature_meta["feature_to_roi"], dtype=np.int64)
    feature_to_type = np.array(feature_meta["feature_to_type"], dtype=np.int64)
    num_rois = len(feature_meta["roi_names"])
    num_types = len(feature_meta["type_names"])

    model = JointScoringModel(
        num_features=num_features,
        num_rois=num_rois,
        num_types=num_types,
        num_classes=num_classes
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    rng = random.Random(args.seed)
    predictions = []
    selected_subsets = []
    all_labels = []
    all_preds = []
    all_probas = []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            case_ids_batch = batch["case_id"]

            for i, case_id in enumerate(case_ids_batch):
                image = images[i:i + 1]
                radiomics_vec = radiomics[case_id]

                candidate_subsets = sample_random_subsets(num_features, k, pool_size, rng)
                subset_scores = []
                for subset in candidate_subsets:
                    f_ids, r_ids, t_ids, vals = build_subset_tensors(
                        radiomics_vec, subset, feature_to_roi, feature_to_type, device
                    )
                    score = model.score_subset(
                        image, f_ids.unsqueeze(0), r_ids.unsqueeze(0), t_ids.unsqueeze(0), vals.unsqueeze(0)
                    )
                    subset_scores.append(score.item())

                subset_scores = np.array(subset_scores)
                topm_indices = np.argsort(subset_scores)[-top_m:]
                topm_subsets = candidate_subsets[topm_indices]

                logits_list = []
                for subset in topm_subsets:
                    f_ids, r_ids, t_ids, vals = build_subset_tensors(
                        radiomics_vec, subset, feature_to_roi, feature_to_type, device
                    )
                    logits = model.classify_subset(
                        image, f_ids.unsqueeze(0), r_ids.unsqueeze(0), t_ids.unsqueeze(0), vals.unsqueeze(0)
                    )
                    logits_list.append(logits)

                logits_ensemble = torch.mean(torch.cat(logits_list, dim=0), dim=0, keepdim=True)
                probas = F.softmax(logits_ensemble, dim=1).cpu().numpy()[0]
                pred_class = int(np.argmax(probas))

                predictions.append({
                    "case_id": case_id,
                    "pred_class": pred_class,
                    **{f"prob_{i}": float(probas[i]) for i in range(probas.shape[0])}
                })

                subset_names = None
                if trained_feature_names:
                    subset_names = [
                        [trained_feature_names[idx] for idx in subset.tolist()]
                        for subset in topm_subsets
                    ]

                selected_subsets.append({
                    "case_id": case_id,
                    "topm_indices": topm_subsets.tolist(),
                    "topm_scores": subset_scores[topm_indices].tolist(),
                    "topm_names": subset_names
                })

                if labels_dict:
                    all_labels.append(int(batch["label"][i].item()))
                    all_preds.append(pred_class)
                    all_probas.append(probas)

    pred_df = pd.DataFrame(predictions)
    pred_path = outdir / "predictions_eval.csv"
    pred_df.to_csv(pred_path, index=False)
    logger.info(f"Saved predictions to {pred_path}")

    selected_path = outdir / "selected_subsets_eval.json"
    with open(selected_path, "w") as f:
        json.dump(selected_subsets, f, indent=2)
    logger.info(f"Saved selected subsets to {selected_path}")

    if labels_dict and all_labels:
        y_true = np.array(all_labels)
        y_pred = np.array(all_preds)
        y_proba = np.vstack(all_probas)
        metrics = compute_metrics(y_true, y_pred, y_proba, return_per_class=True)
        metrics_path = outdir / "metrics_eval.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info(
            f"Metrics saved to {metrics_path} | "
            f"Acc={metrics['accuracy']:.4f} Macro-F1={metrics['macro_f1']:.4f} QWK={metrics['qwk']:.4f}"
        )
    else:
        logger.info("Labels not provided; skipped metrics.")


if __name__ == "__main__":
    main()

