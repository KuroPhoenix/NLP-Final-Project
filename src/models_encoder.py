from __future__ import annotations
import gc
import numpy as np
import pandas as pd

# ModernBERT-large router: shared trunk -> 11 performance logits (BCE with soft {0,0.5,1}
# targets) + 1 auxiliary difficulty head (predict mean #correct). Inference uses
# sigmoid(perf_logits) as expected per-model performance, ensembled with the classical models.
# torch/transformers are imported lazily so the module is importable on a CPU-only box.


def _headtail_trim(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    head = int(max_chars * 0.7)
    return text[:head] + "\n[...]\n" + text[-(max_chars - head):]


def _train_predict(cfg, train_texts, labels, diff, eval_texts):
    """Fine-tune one model on (train_texts, labels, diff); return sigmoid perf preds for eval_texts."""
    import torch
    from torch.utils.data import Dataset, DataLoader
    from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(cfg.encoder_id)
    max_chars = cfg.encoder_max_len * 6  # head+tail char budget before token truncation

    class DS(Dataset):
        def __init__(self, texts, labels=None, diff=None):
            self.enc = tok([_headtail_trim(t, max_chars) for t in texts],
                           truncation=True, max_length=cfg.encoder_max_len)
            self.labels = labels
            self.diff = diff

        def __len__(self):
            return len(self.enc["input_ids"])

        def __getitem__(self, i):
            d = {k: self.enc[k][i] for k in self.enc}
            if self.labels is not None:
                d["labels"] = self.labels[i]
                d["difficulty"] = float(self.diff[i])
            return d

    def collate(batch):
        labeled = "labels" in batch[0]
        feats = [{k: b[k] for k in b if k not in ("labels", "difficulty")} for b in batch]
        out = tok.pad(feats, return_tensors="pt")
        if labeled:
            out["labels"] = torch.tensor(np.stack([b["labels"] for b in batch]), dtype=torch.float32)
            out["difficulty"] = torch.tensor([b["difficulty"] for b in batch], dtype=torch.float32)
        return out

    class Router(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = AutoModel.from_pretrained(cfg.encoder_id)
            h = self.backbone.config.hidden_size
            self.dropout = torch.nn.Dropout(0.1)
            self.perf_head = torch.nn.Linear(h, 11)
            self.diff_head = torch.nn.Linear(h, 1)

        def forward(self, input_ids, attention_mask, **kw):
            out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
            m = attention_mask.unsqueeze(-1).to(out.last_hidden_state.dtype)
            pooled = (out.last_hidden_state * m).sum(1) / m.sum(1).clamp(min=1e-6)
            pooled = self.dropout(pooled)
            return self.perf_head(pooled), self.diff_head(pooled).squeeze(-1)

    model = Router().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.encoder_lr, weight_decay=0.01)
    bce = torch.nn.BCEWithLogitsLoss()
    mse = torch.nn.MSELoss()

    dl = DataLoader(DS(train_texts, labels, diff), batch_size=cfg.encoder_bs,
                    shuffle=True, collate_fn=collate)
    steps = max(1, len(dl) // cfg.encoder_grad_accum) * cfg.encoder_epochs
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
    use_amp = device == "cuda"

    model.train()
    opt.zero_grad()
    for epoch in range(cfg.encoder_epochs):
        last = 0.0
        for step, batch in enumerate(dl):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                perf_logits, diff_pred = model(batch["input_ids"], batch["attention_mask"])
                loss = bce(perf_logits, batch["labels"]) + cfg.aux_diff_weight * mse(diff_pred, batch["difficulty"])
            (loss / cfg.encoder_grad_accum).backward()
            if (step + 1) % cfg.encoder_grad_accum == 0:
                opt.step(); sched.step(); opt.zero_grad()
            last = float(loss.item())
        print(f"    epoch {epoch + 1}/{cfg.encoder_epochs} loss={last:.4f}", flush=True)

    model.eval()
    edl = DataLoader(DS(eval_texts), batch_size=cfg.encoder_bs * 2, shuffle=False, collate_fn=collate)
    preds = []
    with torch.no_grad():
        for batch in edl:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                perf_logits, _ = model(batch["input_ids"], batch["attention_mask"])
            preds.append(torch.sigmoid(perf_logits).float().cpu().numpy())
    out = np.vstack(preds).astype(np.float32)
    del model, opt, sched
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    return out


def cache_tag(cfg, n_train: int) -> str:
    """Cache key includes encoder config so changing it never reuses stale predictions."""
    return f"{n_train}_L{cfg.encoder_max_len}_e{cfg.encoder_epochs}"


def encoder_oof_and_test(cfg, train_df, test_df, perf, folds):
    """5-fold OOF + full-fit test preds, checkpointed per fold so an interruption only
    loses the current fold (resumable on re-run within a live session)."""
    labels = perf.astype(np.float32)
    diff = labels.sum(1) / 11.0
    tr_texts = train_df["query"].fillna("").astype(str).tolist()
    te_texts = test_df["query"].fillna("").astype(str).tolist()
    etag = cache_tag(cfg, len(train_df))
    oof = np.zeros_like(perf, dtype=np.float32)
    for i, (tr, va) in enumerate(folds, 1):
        fp = cfg.cache_dir / f"enc_fold{i}_{etag}.npz"
        if fp.exists():
            z = np.load(fp)
            oof[z["va"]] = z["pred"]
            print(f"[encoder] fold {i}/{len(folds)} loaded from cache", flush=True)
            continue
        print(f"[encoder] fold {i}/{len(folds)} ({len(tr)} train, {len(va)} val)...", flush=True)
        pred = _train_predict(cfg, [tr_texts[j] for j in tr], labels[tr], diff[tr],
                              [tr_texts[j] for j in va])
        oof[va] = pred
        if not cfg.smoke:
            np.savez(fp, va=va, pred=pred)
    tfp = cfg.cache_dir / f"enc_testfull_{etag}.npy"
    if tfp.exists():
        print("[encoder] test full-fit loaded from cache", flush=True)
        test_pred = np.load(tfp)
    else:
        print("[encoder] full-fit for test predictions...", flush=True)
        test_pred = _train_predict(cfg, tr_texts, labels, diff, te_texts)
        if not cfg.smoke:
            np.save(tfp, test_pred)
    return oof, test_pred
