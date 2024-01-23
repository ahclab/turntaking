from pprint import pprint

import torch
from torchmetrics import F1Score, Metric, PrecisionRecallCurve, StatScores
from turntaking.vap_to_turntaking.events import TurnTakingEvents


class F1_Hold_Shift(Metric):
    # Set to True if the metric during 'update' requires access to the global metric
    # state for its calculations. If not, setting this to False indicates that all
    # batch states are independent and we will optimize the runtime of 'forward'
    full_state_update: bool = False

    def __init__(self, threshold=0.5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stat_scores = StatScores(threshold=threshold, reduce="macro", multiclass=True, num_classes=2)
        self.probs = None
        self.labels = None

    def probs_shift_hold(self, p, shift, hold):
        probs, labels = [], []

        for next_speaker in [0, 1]:
            ws = torch.where(shift[..., next_speaker])
            if len(ws[0]) > 0:
                tmp_probs = p[ws][..., next_speaker]
                tmp_lab = torch.ones_like(tmp_probs, dtype=torch.long)
                probs.append(tmp_probs)
                labels.append(tmp_lab)

            # Hold label -> 0
            # Hold prob -> 1 - p  # opposite guess
            wh = torch.where(hold[..., next_speaker])
            if len(wh[0]) > 0:
                # complement in order to be combined with shifts
                tmp_probs = 1 - p[wh][..., next_speaker]
                tmp_lab = torch.zeros_like(tmp_probs, dtype=torch.long)
                probs.append(tmp_probs)
                labels.append(tmp_lab)

        if len(probs) > 0:
            probs = torch.cat(probs)
            #probs = torch.zeros_like(probs, dtype=torch.long)
            labels = torch.cat(labels)
        else:
            probs = None
            labels = None

        #pprint(probs)
        #pprint(labels)
        return probs, labels

    def get_score(self, tp, fp, tn, fn, EPS=1e-9):
        precision = tp / (tp + fp + EPS)
        recall = tp / (tp + fn + EPS)
        f1 = tp / (tp + 0.5 * (fp + fn) + EPS)
        return f1, precision, recall

    def reset(self):
        self.stat_scores.reset()

    def update(self, p, hold, shift):
        self.probs, self.labels = self.probs_shift_hold(p, shift=shift, hold=hold)
        if self.probs is not None:
           self.stat_scores.update(self.probs, self.labels)

    def compute(self):
        hold, shift = self.stat_scores.compute()

        # HOLD
        h_tp, h_fp, h_tn, h_fn, h_sup = hold
        h_f1, h_precision, h_recall = self.get_score(h_tp, h_fp, h_tn, h_fn)

        # SHIFT
        s_tp, s_fp, s_tn, s_fn, s_sup = shift
        s_f1, s_precision, s_recall = self.get_score(s_tp, s_fp, s_tn, s_fn)

        # Weighted F1
        f1h = h_f1 * h_sup
        f1s = s_f1 * s_sup
        tot = h_sup + s_sup
        f1_weighted = (f1h + f1s) / tot
        return {
            "f1_weighted": f1_weighted,
            "hold": {
                "f1": h_f1,
                "precision": h_precision,
                "recall": h_recall,
                "support": h_sup,
            },
            "shift": {
                "f1": s_f1,
                "precision": s_precision,
                "recall": s_recall,
                "support": s_sup,
            },
        }




class TurnTakingMetrics(Metric):
    """
    Used with discrete model, VAProjection.
    """

    # Set to True if the metric during 'update' requires access to the global metric
    # state for its calculations. If not, setting this to False indicates that all
    # batch states are independent and we will optimize the runtime of 'forward'
    full_state_update: bool = False


    def __init__(
        self,
        hs_kwargs,
        bc_kwargs,
        metric_kwargs,
        threshold_shift_hold=0.5,
        threshold_pred_shift=0.5,
        threshold_pred_ov=0.5,
        threshold_short_long=0.5,
        threshold_bc_pred=0.5,
        shift_hold_pr_curve=False,
        bc_pred_pr_curve=False,
        shift_pred_pr_curve=False,
        ov_pred_pr_curve=False,
        long_short_pr_curve=False,
        frame_hz=100,
        dist_sync_on_step=False,
        seed=42
    ):
        # call `self.add_state`for every internal state that is needed for the metrics computations
        # dist_reduce_fx indicates the function that should be used to reduce
        # state from multiple processes
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.seed = seed

        # Metrics
        # self.f1: class to provide f1-weighted as well as other stats tp,fp,support, etc...
        self.hs = F1_Hold_Shift(threshold=threshold_shift_hold)

        def create_f1_score(threshold, num_classes=2, multiclass=True, average="weighted"):
            return F1Score(
                threshold=threshold,
                num_classes=num_classes,
                multiclass=multiclass,
                average=average
            )

        self.predict_shift = create_f1_score(threshold_pred_shift)
        self.predict_shift_0 = create_f1_score(threshold_pred_shift)
        self.predict_shift_1 = create_f1_score(threshold_pred_shift)
        self.predict_ov = create_f1_score(threshold_pred_ov)
        self.predict_ov_0 = create_f1_score(threshold_pred_ov)
        self.predict_ov_1 = create_f1_score(threshold_pred_ov)
        self.short_long = create_f1_score(threshold_short_long)
        self.short_long_0 = create_f1_score(threshold_short_long)
        self.short_long_1 = create_f1_score(threshold_short_long)
        self.predict_backchannel = create_f1_score(threshold_bc_pred)
        self.predict_backchannel_0 = create_f1_score(threshold_bc_pred)
        self.predict_backchannel_1 = create_f1_score(threshold_bc_pred)

        self.pr_curve_shift_hold = shift_hold_pr_curve
        if self.pr_curve_shift_hold:
            self.shift_hold_pr = PrecisionRecallCurve(pos_label=1)

        self.pr_curve_bc_pred = bc_pred_pr_curve
        if self.pr_curve_bc_pred:
            self.bc_pred_pr = PrecisionRecallCurve(pos_label=1)
            # print("::::::::::::::::::::::::::::::::::::::")
            # print(self.bc_pred_pr)
            # print("::::::::::::::::::::::::::::::::::::::")

        self.pr_curve_shift_pred = shift_pred_pr_curve
        if self.pr_curve_shift_pred:
            self.shift_pred_pr = PrecisionRecallCurve(pos_label=1)
        
        self.pr_curve_ov_pred = ov_pred_pr_curve
        if self.pr_curve_ov_pred:
            self.ov_pred_pr = PrecisionRecallCurve(pos_label=1)

        self.pr_curve_long_short = long_short_pr_curve
        if self.pr_curve_long_short:
            self.long_short_pr = PrecisionRecallCurve(pos_label=1)

        # Extract the frames of interest for the given metrics
        self.eventer = TurnTakingEvents(
            hs_kwargs=hs_kwargs,
            bc_kwargs=bc_kwargs,
            metric_kwargs=metric_kwargs,
            frame_hz=frame_hz,
            seed = self.seed
        )
        
    def update_predictions(self, probs, labels, predictor):
        """
        Update the given predictor with probabilities and labels.
        """
        if probs:
            predictor.update(torch.cat(probs), torch.cat(labels).long())

    @torch.no_grad()
    def extract_events(self, va, max_frame=None):
        return self.eventer(va, max_frame=max_frame)

    def __repr__(self):
        s = "TurnTakingMetrics"
        s += self.eventer.__repr__()
        return s
    
    def update_shift_hold(self, p, shift, hold):
        self.hs.update(p, hold=hold, shift=shift)

        if self.pr_curve_shift_hold:
            self.shift_hold_pr.update(self.hs.probs, self.hs.labels)

    def update_short_long(self, p, short, long):
        def process_events(event, prob, label):
            probs_dim, labels_dim = [], []
            if event.sum() > 0:
                w = torch.where(event)
                p_event = prob[w]
                probs_dim.append(p_event)
                labels_dim.append(torch.full_like(p_event, label))
            return probs_dim, labels_dim

        # Split pos and neg into two dimensions
        short_0, short_1 = short.clone(), short.clone()
        short_0[:, :, 1] = 0
        short_1[:, :, 0] = 0
        long_0, long_1 = long.clone(), long.clone()
        long_0[:, :, 1] = 0
        long_1[:, :, 0] = 0

        # Process events
        pos_probs, pos_labels = process_events(short, p, 1)
        neg_probs, neg_labels = process_events(long, p, 0)
        probs = pos_probs + neg_probs
        labels = pos_labels + neg_labels

        # Process events for pos_0 and neg_0
        pos_probs_0, pos_labels_0 = process_events(short_0, p, 1)
        neg_probs_0, neg_labels_0 = process_events(long_0, p, 0)
        probs_0 = pos_probs_0 + neg_probs_0
        labels_0 = pos_labels_0 + neg_labels_0

        # Process events for pos_1 and neg_1
        pos_probs_1, pos_labels_1 = process_events(short_1, p, 1)
        neg_probs_1, neg_labels_1 = process_events(long_1, p, 0)
        probs_1 = pos_probs_1 + neg_probs_1
        labels_1 = pos_labels_1 + neg_labels_1

        # Concatenate and update predictions
        if probs:
            self.update_predictions(probs, labels, self.short_long)
            self.update_predictions(probs_0, labels_0, self.short_long_0)
            self.update_predictions(probs_1, labels_1, self.short_long_1)

            if self.pr_curve_shift_pred:
                self.long_short_pr.update(torch.cat(probs), torch.cat(labels).long())

    
    def update_predict_shift(self, p, pos, neg):
        """
        Predict upcoming speaker shift. The events pos/neg are given for the
        correct next speaker.

        * pos next_speaker -> label 1
        * neg next_speaker -> label 0 (flip to have label 0 and take 1-p as their predictions)
        """

        def process_events(events, p, label):
            """
            Process the given events to extract probabilities and labels.
            """
            processed_probs, processed_labels = [], []
            if events.sum() > 0:
                indices = torch.where(events)
                p_events = p[indices]
                if label == 0:
                    p_events = 1 - p_events  # reverse probabilities for negative cases
                processed_probs.append(p_events)
                processed_labels.append(torch.full_like(p_events, label))
            return processed_probs, processed_labels

        # Split pos and neg for each dimension
        pos_0, pos_1 = pos.clone(), pos.clone()
        neg_0, neg_1 = neg.clone(), neg.clone()
        pos_0[:, :, 1] = 0
        pos_1[:, :, 0] = 0
        neg_0[:, :, 1] = 0
        neg_1[:, :, 0] = 0

        # Process events
        pos_probs, pos_labels = process_events(pos, p, 1)
        neg_probs, neg_labels = process_events(neg, p, 0)
        probs = pos_probs + neg_probs
        labels = pos_labels + neg_labels

        # Process events for pos_0 and neg_0
        pos_probs_0, pos_labels_0 = process_events(pos_0, p, 1)
        neg_probs_0, neg_labels_0 = process_events(neg_0, p, 0)
        probs_0 = pos_probs_0 + neg_probs_0
        labels_0 = pos_labels_0 + neg_labels_0

        # Process events for pos_1 and neg_1
        pos_probs_1, pos_labels_1 = process_events(pos_1, p, 1)
        neg_probs_1, neg_labels_1 = process_events(neg_1, p, 0)
        probs_1 = pos_probs_1 + neg_probs_1
        labels_1 = pos_labels_1 + neg_labels_1

        # Concatenate and update predictions
        if probs:
            self.update_predictions(probs, labels, self.predict_shift)
            self.update_predictions(probs_0, labels_0, self.predict_shift_0)
            self.update_predictions(probs_1, labels_1, self.predict_shift_1)

            if self.pr_curve_shift_pred:
                self.shift_pred_pr.update(torch.cat(probs), torch.cat(labels).long())

    def update_predict_overlap(self, p, pos, neg):
        def process_events(events, p, label):
            """
            Process the given events to extract probabilities and labels.
            """
            processed_probs, processed_labels = [], []
            if events.sum() > 0:
                indices = torch.where(events)
                p_events = p[indices]
                if label == 0:
                    p_events = 1 - p_events  # reverse probabilities for negative cases
                processed_probs.append(p_events)
                processed_labels.append(torch.full_like(p_events, label))
            return processed_probs, processed_labels

        # Split pos and neg for each dimension
        pos_0, pos_1 = pos.clone(), pos.clone()
        neg_0, neg_1 = neg.clone(), neg.clone()
        pos_0[:, :, 1] = 0
        pos_1[:, :, 0] = 0
        neg_0[:, :, 1] = 0
        neg_1[:, :, 0] = 0

        # Process events
        pos_probs, pos_labels = process_events(pos, p, 1)
        neg_probs, neg_labels = process_events(neg, p, 0)
        probs = pos_probs + neg_probs
        labels = pos_labels + neg_labels

        # Process events for pos_0 and neg_0
        pos_probs_0, pos_labels_0 = process_events(pos_0, p, 1)
        neg_probs_0, neg_labels_0 = process_events(neg_0, p, 0)
        probs_0 = pos_probs_0 + neg_probs_0
        labels_0 = pos_labels_0 + neg_labels_0

        # Process events for pos_1 and neg_1
        pos_probs_1, pos_labels_1 = process_events(pos_1, p, 1)
        neg_probs_1, neg_labels_1 = process_events(neg_1, p, 0)
        probs_1 = pos_probs_1 + neg_probs_1
        labels_1 = pos_labels_1 + neg_labels_1

        # Concatenate and update predictions
        if probs:
            self.update_predictions(probs, labels, self.predict_ov)
            self.update_predictions(probs_0, labels_0, self.predict_ov_0)
            self.update_predictions(probs_1, labels_1, self.predict_ov_1)

            if self.pr_curve_ov_pred:
                self.ov_pred_pr.update(torch.cat(probs), torch.cat(labels).long())


    def update_predict_backchannel(self, bc_pred_probs, pos, neg):
        def process_events(events, p, label, is_neg=False):
            """
            Process the given events to extract probabilities and labels.
            """
            processed_probs, processed_labels = [], []
            if events.sum() > 0:
                wb, wn, w_speaker = torch.where(events)
                if is_neg:
                    w_backchanneler = torch.logical_not(w_speaker).long()
                    indices = (wb, wn, w_backchanneler)
                else:
                    indices = (wb, wn, w_speaker)
                p_events = p[indices]
                if label == 0 and is_neg:
                    p_events = 1 - p_events  # reverse probabilities for negative cases
                processed_probs.append(p_events)
                processed_labels.append(torch.full_like(p_events, label))
            return processed_probs, processed_labels

        # Split pos and neg for each dimension
        pos_0, pos_1 = pos.clone(), pos.clone()
        neg_0, neg_1 = neg.clone(), neg.clone()
        pos_0[:, :, 1] = 0
        pos_1[:, :, 0] = 0
        neg_0[:, :, 1] = 0
        neg_1[:, :, 0] = 0

        # Process events
        pos_probs, pos_labels = process_events(pos, bc_pred_probs, 1)
        neg_probs, neg_labels = process_events(neg, bc_pred_probs, 0)
        probs = pos_probs + neg_probs
        labels = pos_labels + neg_labels

        # Process events for each dimension
        pos_probs_0, pos_labels_0 = process_events(pos_0, bc_pred_probs, 1)
        neg_probs_0, neg_labels_0 = process_events(neg_0, bc_pred_probs, 0, is_neg=True)
        pos_probs_1, pos_labels_1 = process_events(pos_1, bc_pred_probs, 1)
        neg_probs_1, neg_labels_1 = process_events(neg_1, bc_pred_probs, 0, is_neg=True)

        # Combine probabilities and labels
        probs_0 = pos_probs_0 + neg_probs_0
        labels_0 = pos_labels_0 + neg_labels_0
        probs_1 = pos_probs_1 + neg_probs_1
        labels_1 = pos_labels_1 + neg_labels_1

        # Update predictions
        if probs:
            self.update_predictions(probs, labels, self.predict_backchannel)
            self.update_predictions(probs_0, labels_0, self.predict_backchannel_0)
            self.update_predictions(probs_1, labels_1, self.predict_backchannel_1)

        if self.pr_curve_bc_pred:
            self.bc_pred_pr.update(torch.cat(probs), torch.cat(labels).long())


    def reset(self):
        super().reset()
        self.hs.reset()
        self.predict_shift.reset()
        self.predict_ov.reset()
        self.short_long.reset()
        self.predict_backchannel.reset()
        if self.pr_curve_shift_hold:
            self.shift_hold_pr.reset()

        if self.pr_curve_bc_pred:
            self.bc_pred_pr.reset()

        if self.pr_curve_shift_pred:
            self.shift_pred_pr.reset()
        
        if self.pr_curve_ov_pred:
            self.ov_pred_pr.reset()

        if self.pr_curve_long_short:
            self.long_short_pr.reset()

 
    def update(self, p, bc_pred_probs=None, events=None, va=None):
        """
        p:              tensor, next_speaker probability. Must take into account current speaker such that it can be used for pre-shift/hold, backchannel-pred/ongoing
        pre_probs:      tensor, on active next speaker probability for independent
        bc_pred_probs:  tensor, Special probability associated with a backchannel prediction
        events:         dict, containing information about the events in the sequences
        vad:            tensor, VAD activity. Only used if events is not given.


        events: [
                    'shift',
                    'hold',
                    'short',
                    'long',
                    'predict_shift_pos',
                    'predict_shift_neg',
                    'predict_bc_pos',
                    'predict_bc_neg'
                    'predict_shift_ov_pos',
                    'predict_shift_ov_neg'
                ]
        """

        # Find valid event-frames if event is not given
        if events is None:
            events = self.extract_events(va)

        # SHIFT/HOLD
        # self.hs.update(p, hold=events["hold"], shift=events["shift"])
        self.update_shift_hold(
            p, hold=events["hold"], shift=events["shift"]
        )

        # Predict Shifts
        self.update_predict_shift(
            p, pos=events["predict_shift_pos"], neg=events["predict_shift_neg"]
        )

        # Predict Overlaps
        self.update_predict_overlap(
            p, pos=events["predict_shift_ov_pos"], neg=events["predict_shift_ov_neg"]
        )

        # PREDICT BACKCHANNELS & Short/Long
        if bc_pred_probs is not None:
            self.update_predict_backchannel(
                bc_pred_probs,
                pos=events["predict_bc_pos"],
                neg=events["predict_bc_neg"],
            )

            # Long/Short
            self.update_short_long(
                bc_pred_probs, short=events["short"], long=events["long"]
            )
        else:
            # Long/Short
            self.update_short_long(p, short=events["short"], long=events["long"])


    def compute(self):
        f1_hs = self.hs.compute()
        f1_predict_shift = self.predict_shift.compute()
        f1_predict_shift_0 = self.predict_shift_0.compute()
        f1_predict_shift_1 = self.predict_shift_1.compute()
        f1_short_long = self.short_long.compute()
        f1_short_long_0 = self.short_long_0.compute()
        f1_short_long_1 = self.short_long_1.compute()

        ret = {
            "f1_hold_shift": f1_hs["f1_weighted"],
            "f1_predict_shift": f1_predict_shift,
            "f1_predict_shift_0": f1_predict_shift_0,
            "f1_predict_shift_1": f1_predict_shift_1,
            "f1_short_long": f1_short_long,
            "f1_short_long_0": f1_short_long_0,
            "f1_short_long_1": f1_short_long_1,
        }

        try:
            ret["f1_bc_prediction"] = self.predict_backchannel.compute()
            ret["f1_bc_prediction_0"] = self.predict_backchannel_0.compute()
            ret["f1_bc_prediction_1"] = self.predict_backchannel_1.compute()
        except:
            ret["f1_bc_prediction"] = -1
            ret["f1_bc_prediction_0"] = -1
            ret["f1_bc_prediction_1"] = -1
        
        try:
            ret["f1_predict_ov"] = self.predict_ov.compute()
            ret["f1_predict_ov_0"] = self.predict_ov_0.compute()
            ret["f1_predict_ov_1"] = self.predict_ov_1.compute()
        except:
            ret["f1_predict_ov"] = -1
            ret["f1_predict_ov_0"] = -1
            ret["f1_predict_ov_1"] = -1

        if self.pr_curve_shift_hold:
            ret["pr_curve_shift_hold"] = self.shift_hold_pr.compute()

        if self.pr_curve_bc_pred:
            ret["pr_curve_bc_pred"] = self.bc_pred_pr.compute()

        if self.pr_curve_shift_pred:
            ret["pr_curve_shift_pred"] = self.shift_pred_pr.compute()

        if self.pr_curve_ov_pred:
            ret["pr_curve_ov_pred"] = self.ov_pred_pr.compute()

        if self.pr_curve_long_short:
            ret["pr_curve_long_short"] = self.long_short_pr.compute()

        ret["shift"] = f1_hs["shift"]
        ret["hold"] = f1_hs["hold"]
        return ret


def main_old():
    import matplotlib.pyplot as plt
    from conv_ssl.evaluation.utils import load_dm, load_model
    from conv_ssl.utils import to_device
    from tqdm import tqdm

    # Load Datai
    # The only required data is VAD (onehot encoding of voice activity) e.g. (B, N_FRAMES, 2) for two speakers
    dm = load_dm(batch_size=12)
    # diter = iter(dm.val_dataloader())

    ###################################################
    # Load Model
    ###################################################
    # run_path = "how_so/VPModel/10krujrj"  # independent
    #run_path = "/ahc/work2/kazuyo-oni/conv_ssl/VAP/3f2ehpiy"  # discrete
    checkpoint_path = "/ahc/work2/kazuyo-oni/conv_ssl/VAP/3f2ehpiy/checkpoints/epoch=4-step=11154.ckpt"

    # run_path = "how_so/VPModel/2608x2g0"  # independent (same bin size)
    model = load_model(checkpoint_path=checkpoint_path, strict=False)
    model = model.eval()
    # model = model.to("cpu")
    # model = model.to("cpu")

    event_kwargs = dict(
        post_onset_shift=1,
        pre_offset_shift=1,
        post_onset_hold=1,
        pre_offset_hold=1,
        min_silence=0.15,
        non_shift_horizon=2.0,
        non_shift_majority_ratio=0.95,
        metric_pad=0.05,
        metric_dur=0.1,
        metric_onset_dur=0.3,
        metric_pre_label_dur=0.5,
        metric_min_context=1.0,
        bc_max_duration=1.0,
        bc_pre_silence=1.0,
        bc_post_silence=3.0,
    )

    # # update vad_projection metrics
    # metric_kwargs = {
    #     "event_pre": 0.5,  # seconds used to estimate PRE-f1-SHIFT/HOLD
    #     "event_min_context": 1.0,  # min context duration before extracting metrics
    #     "event_min_duration": 0.15,  # the minimum required segment to extract SHIFT/HOLD (start_pad+target_duration)
    #     "event_horizon": 1.0,  # SHIFT/HOLD requires lookahead to determine mutual starts etc
    #     "event_start_pad": 0.05,  # Predict SHIFT/HOLD after this many seconds of silence after last speaker
    #     "event_target_duration": 0.10,  # duration of segment to extract each SHIFT/HOLD guess
    #     "event_bc_target_duration": 0.25,  # duration of activity, in a backchannel, to extract BC-ONGOING metrics
    #     "event_bc_pre_silence": 1,  # du
    #     "event_bc_post_silence": 2,
    #     "event_bc_max_active": 1.0,
    #     "event_bc_prediction_window": 0.4,
    #     "event_bc_neg_active": 1,
    #     "event_bc_neg_prefix": 1,
    #     "event_bc_ongoing_threshold": 0.5,
    #     "event_bc_pred_threshold": 0.5,
    # }
    # # Updatemetric_kwargs metrics
    # for metric, val in metric_kwargs.items():
    #     model.conf["vad_projection"][metric] = val

    N = 10
    model.test_metric = model.init_metric(
        model.conf, 
        model.frame_hz, 
        bc_pred_pr_curve=True,
        shift_pred_pr_curve=True,
        long_short_pr_curve=True,
    )
    
    
    from conv_ssl.callbacks import SymmetricSpeakersCallback
    from pytorch_lightning import Trainer
    trainer = Trainer(
        logger=None,
        callbacks=[SymmetricSpeakersCallback()],
        gpus=-1,
        fast_dev_run=0,
        deterministic=True,
    )

    #result = trainer.test(model, dataloaders=dm.val_dataloader(), verbose=False)
    #pprint(trainer.model.test_metric.long_short_pr.preds)


    # tt_metrics = TurnTakingMetricsDiscrete(bin_times=model.conf['vad_projection']['bin_times'])
    for ii, batch in tqdm(enumerate(dm.val_dataloader()), total=N):
        batch = to_device(batch, model.device)
        ########################################################################
        # Extract events/labels on full length (with horizon) VAD
        events = model.test_metric.extract_events(batch["vad"], max_frame=1000)
        ########################################################################
        # Forward Pass through the model
        loss, out, batch = model.shared_step(batch)
        turn_taking_probs = model.VAP(
            logits=out["logits_vp"], va=batch["vad"]
        )
        ########################################################################
        # Update metrics
        model.test_metric.update(
            p=turn_taking_probs["p"],
            bc_pred_probs=turn_taking_probs.get("bc_prediction", None),
            events=events,
        )
        if ii == N:
            break
    result = model.test_metric.compute()
    print(result.keys())

    for k, v in result.items():
        print(f"{k}: {v}")

    #pprint(model.test_metric.long_short_pr.preds)


if __name__ == "__main__":
    main_old()

    """
    from turntaking.vap_to_turntaking.config.example_data import example, event_conf

    metric = TurnTakingMetrics(
        hs_kwargs=event_conf["hs"],
        bc_kwargs=event_conf["bc"],
        metric_kwargs=event_conf["metric"],
        bc_pred_pr_curve=True,
        shift_pred_pr_curve=True,
        long_short_pr_curve=True,
        frame_hz=100,
    )

    # Update
    metric.update(
        p=turn_taking_probs["p"],
        bc_pred_probs=turn_taking_probs.get("bc_prediction", None),
        events=events,
    )

    # Compute
    result = metric.compute()
    """
