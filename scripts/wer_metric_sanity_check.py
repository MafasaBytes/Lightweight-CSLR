"""
Compare WER implementations on the *same* subset of utterances.

Why this exists
---------------
We observed that:
- AdaptSign reports ~18.5% dev WER (their official evaluation)
- Our greedy evaluation on SI5 dev reports much higher WER

There are two major potential causes:
1) Different split/protocol (e.g., SI5 dev has 111 samples; AdaptSign STM dev has ~536)
2) Different WER scoring (merge_same=True, non-uniform penalties, etc.)

This script:
- Loads a CTM (predictions) produced by `src/evaluation/eval_adaptsign.py --dump_ctm ...`
- Loads AdaptSign STM groundtruth
- Computes:
  a) "simple" WER using our `src.utils.metrics.compute_wer`
  b) AdaptSign official-style WER on the INTERSECTION of file IDs present in CTM

Usage
-----
venv\\Scripts\\python scripts\\wer_metric_sanity_check.py ^
  --ctm tmp_teacher_dev_raw.ctm ^
  --stm adaptsign_repo\\preprocess\\phoenix2014\\phoenix2014-groundtruth-dev.stm
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from src.utils.metrics import compute_wer


def load_stm(stm_path: Path) -> Dict[str, List[str]]:
    # Mirrors AdaptSign's `load_groundtruth`: tokens start at field 6+
    gt: Dict[str, List[str]] = {}
    for line in stm_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split(" ")
        file_id = parts[0]
        toks = [t for t in parts[5:] if t]
        gt[file_id] = toks
    return gt


def load_ctm(ctm_path: Path) -> Dict[str, List[str]]:
    # Mirrors AdaptSign's `load_prediction`
    pred: Dict[str, List[str]] = {}
    # Some Windows toolchains can accidentally write literal "\n" sequences instead of real newlines.
    # Normalize those so we can still parse the CTM deterministically.
    raw = ctm_path.read_text(encoding="utf-8")
    raw = raw.replace("\\n", "\n")
    for line in raw.splitlines():
        if not line.strip():
            continue
        # CTM is whitespace-delimited, but may contain multiple spaces/tabs.
        parts = line.split()
        if len(parts) < 5:
            raise ValueError(f"Malformed CTM line (expected >=5 fields): {line!r}")
        # file_name channel start duration word
        file_name = parts[0]
        wd = parts[-1]
        pred.setdefault(file_name, []).append(wd)
    return pred


def _merge_same(tokens: List[str]) -> List[str]:
    if not tokens:
        return []
    out = [tokens[0]]
    for t in tokens[1:]:
        if t != out[-1]:
            out.append(t)
    return out


def _adaptsign_align(
    ref: List[str],
    hyp: List[str],
    penalty: Dict[str, int],
    merge_same: bool = True,
) -> Tuple[List[str], List[str]]:
    """
    Replicates AdaptSign's official *python* evaluator alignment (DP + backtrace).

    Important: penalties are used for ALIGNMENT ONLY; the WER computation in AdaptSign is a token
    mismatch rate on the aligned sequences.

    Returns: (aligned_gt_tokens, aligned_pred_tokens)
    """
    # This is a lightly adapted copy of AdaptSign's
    # `adaptsign_repo/evaluation/slr_eval/python_wer_evaluation.py:get_wer_delsubins`
    # with modern numpy dtypes.
    if merge_same:
        hyp = _merge_same(hyp)

    ref_lgt = len(ref) + 1
    hyp_lgt = len(hyp) + 1

    costs = np.ones((ref_lgt, hyp_lgt), dtype=np.int32) * 10**9
    costs[0, :] = np.arange(hyp_lgt, dtype=np.int32) * int(penalty["ins"])
    costs[:, 0] = np.arange(ref_lgt, dtype=np.int32) * int(penalty["del"])

    backtrace = np.zeros((ref_lgt, hyp_lgt), dtype=np.int32)
    backtrace[0, :] = 2  # insert
    backtrace[:, 0] = 3  # delete

    for i in range(1, ref_lgt):
        for j in range(1, hyp_lgt):
            if ref[i - 1] == hyp[j - 1]:
                # correct
                if costs[i - 1, j - 1] < costs[i, j]:
                    costs[i, j] = costs[i - 1, j - 1]
                    backtrace[i, j] = 0
            else:
                sub_cost = costs[i - 1, j - 1] + int(penalty["sub"])
                ins_cost = costs[i, j - 1] + int(penalty["ins"])
                del_cost = costs[i - 1, j] + int(penalty["del"])
                min_cost = min(sub_cost, ins_cost, del_cost)
                if min_cost < costs[i, j]:
                    costs[i, j] = min_cost
                    # 1=sub, 2=ins, 3=del (matches AdaptSign file)
                    backtrace[i, j] = [sub_cost, ins_cost, del_cost].index(min_cost) + 1

    # Backtrace to aligned token streams
    bt_i, bt_j = ref_lgt - 1, hyp_lgt - 1
    aligned_gt: List[str] = []
    aligned_pred: List[str] = []
    while bt_i > 0 or bt_j > 0:
        op = backtrace[bt_i, bt_j]
        if bt_i > 0 and bt_j > 0 and op in (0, 1):
            # correct or substitute
            aligned_gt.append(ref[bt_i - 1])
            aligned_pred.append(hyp[bt_j - 1])
            bt_i -= 1
            bt_j -= 1
        elif bt_j > 0 and (bt_i == 0 or op == 2):
            # insert: consume hyp
            aligned_gt.append("*" * max(1, len(hyp[bt_j - 1])))
            aligned_pred.append(hyp[bt_j - 1])
            bt_j -= 1
        else:
            # delete: consume ref
            aligned_gt.append(ref[bt_i - 1])
            aligned_pred.append("*" * max(1, len(ref[bt_i - 1])))
            bt_i -= 1

    aligned_gt.reverse()
    aligned_pred.reverse()
    return aligned_gt, aligned_pred


def adaptsign_official_python_wer(
    gt: Dict[str, List[str]],
    pred: Dict[str, List[str]],
    penalty: Dict[str, int],
    merge_same: bool = True,
) -> Tuple[float, int]:
    keys = sorted(set(gt.keys()) & set(pred.keys()))
    wer_err = 0
    cnt = 0
    for k in keys:
        ref = gt[k]
        hyp = pred[k]
        aligned_gt, aligned_pred = _adaptsign_align(ref, hyp, penalty=penalty, merge_same=merge_same)
        for g, p in zip(aligned_gt, aligned_pred):
            if "*" in g:
                continue
            cnt += 1
            if g != p:
                wer_err += 1
    return 100.0 * wer_err / max(1, cnt), cnt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctm", type=str, required=True)
    ap.add_argument("--stm", type=str, required=True)
    args = ap.parse_args()

    ctm_path = Path(args.ctm)
    stm_path = Path(args.stm)

    gt = load_stm(stm_path)
    pred = load_ctm(ctm_path)

    inter = sorted(set(gt.keys()) & set(pred.keys()))
    print(f"STM utterances: {len(gt)}")
    print(f"CTM utterances: {len(pred)}")
    print(f"Intersection:   {len(inter)}")
    if not inter:
        raise SystemExit("No overlapping file IDs between STM and CTM.")

    # Simple WER on intersection (uniform costs; no merge-same)
    refs_str = [" ".join(gt[k]) for k in inter]
    hyps_str = [" ".join(pred[k]) for k in inter]
    simple = compute_wer(refs_str, hyps_str)

    # AdaptSign official python evaluator
    official, n_cnt = adaptsign_official_python_wer(
        gt, pred, penalty={"ins": 3, "del": 3, "sub": 4}, merge_same=True
    )

    print(f"\nSimple WER (uniform, no merge-same):      {simple:.2f}%")
    print(f"AdaptSign official python WER:            {official:.2f}%  (N_counted_tokens={n_cnt})")


if __name__ == "__main__":
    main()


