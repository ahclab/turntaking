from tqdm import tqdm
from turntaking.utils import to_device
import torch
from turntaking.utils import everything_deterministic
import pandas as pd

everything_deterministic()


def get_curves(preds, target, pos_label=1, thresholds=None, EPS=1e-6):
    """
    precision = tp / (tp+fp)
    recall = tp / (tp+fn)

    """

    if thresholds is None:
        thresholds = torch.linspace(0, 1, steps=101)

    if pos_label == 0:
        raise NotImplementedError("Have not done this")

    ba, f1 = [], []
    auc0, auc1 = [], []
    prec0, rec0 = [], []
    prec1, rec1 = [], []
    pos_label_idx = torch.where(target == 1)
    neg_label_idx = torch.where(target == 0)

    for t in thresholds:
        pred_labels = (preds >= t).float()
        correct = pred_labels == target

        # POSITIVES
        tp = correct[pos_label_idx].sum()
        n_p = (target == 1).sum()
        fn = n_p - tp
        # NEGATIVES
        tn = correct[neg_label_idx].sum()
        n_n = (target == 0).sum()
        fp = n_n - tn
        ###################################3
        # Balanced Accuracy
        ###################################3
        # TPR, TNR
        tpr = tp / n_p
        tnr = tn / n_n
        # BA
        ba_tmp = (tpr + tnr) / 2
        ba.append(ba_tmp)
        ###################################3
        # F1
        ###################################3
        precision1 = tp / (tp + fp + EPS)
        recall1 = tp / (tp + fn + EPS)
        f1_1 = 2 * precision1 * recall1 / (precision1 + recall1 + EPS)
        prec1.append(precision1)
        rec1.append(recall1)
        auc1.append(precision1 * recall1)

        precision0 = tn / (tn + fn + EPS)
        recall0 = tn / (tn + fp + EPS)
        f1_0 = 2 * precision0 * recall0 / (precision0 + recall0 + EPS)
        prec0.append(precision0)
        rec0.append(recall0)
        auc0.append(precision0 * recall0)

        f1w = (f1_0 * n_n + f1_1 * n_p) / (n_n + n_p)
        f1.append(f1w)

    return {
        "bacc": torch.stack(ba),
        "f1": torch.stack(f1),
        "prec1": torch.stack(prec1),
        "rec1": torch.stack(rec1),
        "prec0": torch.stack(prec0),
        "rec0": torch.stack(rec0),
        "auc0": torch.stack(auc0),
        "auc1": torch.stack(auc1),
        "thresholds": thresholds,
    }


def find_threshold(cfg_dict, model, dm, min_thresh=0.01):
    dm.change_frame_mode(True)
    """Find the best threshold using PR-curves"""

    def get_best_thresh(curves, metric, measure, min_thresh):
        ts = curves[metric]["thresholds"]
        over = min_thresh <= ts
        under = ts <= (1 - min_thresh)
        w = torch.where(torch.logical_and(over, under))
        values = curves[metric][measure][w]
        ts = ts[w]
        _, best_idx = values.max(0)
        return ts[best_idx]

    # Init metric:
    if cfg_dict["model"]["vap"]["type"] == "comparative":
        model.test_metric = model.init_metric(
            shift_hold_pr_curve=True,
            bc_pred_pr_curve=False,
            shift_pred_pr_curve=True,
            ov_pred_pr_curve=False,
            long_short_pr_curve=True,
        )
    elif cfg_dict["model"]["vap"]["type"] == "discrete":
        model.test_metric = model.init_metric(
            shift_hold_pr_curve=True,
            bc_pred_pr_curve=True,
            shift_pred_pr_curve=True,
            ov_pred_pr_curve=True,
            long_short_pr_curve=True,
        )
    else:
        model.test_metric = model.init_metric(
            shift_hold_pr_curve=True,
            bc_pred_pr_curve=True,
            shift_pred_pr_curve=True,
            ov_pred_pr_curve=False,
            long_short_pr_curve=True,
        )

    # Find Thresholds
    probs = []
    val_loss = 0
    for ii, batch in tqdm(
        enumerate(dm.val_dataloader()),
        total=len(dm.val_dataloader()),
        dynamic_ncols=True,
        leave=False,
    ):
        # batch = to_device(batch, model.device)
        # Forward Pass through the model
        loss, out, batch = model.shared_step(batch)
        val_loss += loss["total"]
        for o in out["logits_vp"]:
            probs.append(o)

    val_loss /= len(dm.val_dataloader())

    probs = torch.cat(probs).unsqueeze(0).to(cfg_dict["train"]["device"])
    # print(probs.shape)

    d = to_device(dm.get_full_sample("val"), cfg_dict["train"]["device"])
    # print(d["vad"].shape)
    events = model.test_metric.extract_events(va=d["vad"])
    turn_taking_probs = model.VAP(logits=probs, va=d["vad"])
    model.test_metric.update(
        p=turn_taking_probs["p"],
        bc_pred_probs=turn_taking_probs.get("bc_prediction", None),
        events=events,
    )

    _ = model.test_metric.compute()

    ############################################
    # Save predictions
    predictions = {}
    metric_list = []
    if hasattr(model.test_metric, "shift_hold_pr"):
        predictions["shift_hold"] = {
            "preds": torch.cat(model.test_metric.shift_hold_pr.preds),
            "target": torch.cat(model.test_metric.shift_hold_pr.target),
        }
        metric_list.append("shift_hold")
    if hasattr(model.test_metric, "long_short_pr"):
        predictions["long_short"] = {
            "preds": torch.cat(model.test_metric.long_short_pr.preds),
            "target": torch.cat(model.test_metric.long_short_pr.target),
        }
        metric_list.append("long_short")
    if hasattr(model.test_metric, "bc_pred_pr"):
        predictions["bc_preds"] = {
            "preds": torch.cat(model.test_metric.bc_pred_pr.preds),
            "target": torch.cat(model.test_metric.bc_pred_pr.target),
        }
        metric_list.append("bc_preds")
    if hasattr(model.test_metric, "shift_pred_pr"):
        predictions["shift_preds"] = {
            "preds": torch.cat(model.test_metric.shift_pred_pr.preds),
            "target": torch.cat(model.test_metric.shift_pred_pr.target),
        }
        metric_list.append("shift_preds")
    if hasattr(model.test_metric, "ov_pred_pr"):
        predictions["ov_preds"] = {
            "preds": torch.cat(model.test_metric.ov_pred_pr.preds),
            "target": torch.cat(model.test_metric.ov_pred_pr.target),
        }
        metric_list.append("ov_preds")

    ############################################
    # Curves
    curves = {}
    for metric in metric_list:
        curves[metric] = get_curves(
            preds=predictions[metric]["preds"], target=predictions[metric]["target"]
        )

    ############################################
    # find best thresh
    shift_hold_threshold = torch.tensor(0.5)
    bc_pred_threshold = torch.tensor(0.5)
    shift_pred_threshold = torch.tensor(0.5)
    ov_pred_threshold = torch.tensor(0.5)
    long_short_threshold = torch.tensor(0.5)
    if "shift_hold" in curves:
        shift_hold_threshold = get_best_thresh(curves, "shift_hold", "f1", min_thresh)
    if "bc_preds" in curves:
        bc_pred_threshold = get_best_thresh(curves, "bc_preds", "f1", min_thresh)
    if "shift_preds" in curves:
        shift_pred_threshold = get_best_thresh(curves, "shift_preds", "f1", min_thresh)
    if "ov_preds" in curves:
        ov_pred_threshold = get_best_thresh(curves, "ov_preds", "f1", min_thresh)
    if "long_short" in curves:
        long_short_threshold = get_best_thresh(curves, "long_short", "f1", min_thresh)

    thresholds = {
        "shift_hold": shift_hold_threshold,
        "pred_shift": shift_pred_threshold,
        "pred_ov": ov_pred_threshold,
        "pred_bc": bc_pred_threshold,
        "short_long": long_short_threshold,
    }

    model.test_metric = model.init_metric(
        threshold_shift_hold=thresholds.get("shift_hold", 0.5),
        threshold_pred_shift=thresholds.get("pred_shift", 0.3),
        threshold_pred_ov=thresholds.get("pred_ov", 0.1),
        threshold_short_long=thresholds.get("short_long", 0.5),
        threshold_bc_pred=thresholds.get("pred_bc", 0.1),
    )
    model.test_metric.update(
        p=turn_taking_probs["p"],
        bc_pred_probs=turn_taking_probs.get("bc_prediction", None),
        events=events,
    )

    events_score = model.test_metric.compute()

    result = {
        "val_loss": val_loss.item(),
        "shift_hold": events_score["f1_hold_shift"].item(),
        "short_long": events_score["f1_short_long"].item(),
        "shift_pred": events_score["f1_predict_shift"].item(),
        "ov_pred": events_score["f1_predict_ov"].item(),
        "bc_pred": events_score["f1_bc_prediction"].item(),
        "shift_f1": events_score["shift"]["f1"].item(),
        "shift_precision": events_score["shift"]["precision"].item(),
        "shift_recall": events_score["shift"]["recall"].item(),
        "hold_f1": events_score["hold"]["f1"].item(),
        "hold_precision": events_score["hold"]["precision"].item(),
        "hold_recall": events_score["hold"]["recall"].item(),
    }

    print("-" * 60)
    print("### Validation ###")
    print(pd.DataFrame([result]))
    print("-" * 60)

    return thresholds, predictions, curves


def roc_to_threshold(cfg_dict, model, dloader, min_thresh=0.01):
    return find_threshold(cfg_dict, model, dloader, min_thresh)
