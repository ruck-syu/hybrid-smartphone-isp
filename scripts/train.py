"""Train learned / hybrid ISP models on SIDD patches (optionally with knowledge distillation).

  python scripts/train.py --model hybrid_teacher --data /content/prep --out /content/runs/hybrid_teacher
  python scripts/train.py --model hybrid_student --teacher /content/runs/hybrid_teacher/best.pt --kd 1.0 ...
  python scripts/train.py --model learned --data /content/prep --out /content/runs/learned
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from isp.data import SIDDPatches, fixed_val_crops  # noqa: E402
from isp.models import TorchColorBack, build_model, count_params, load_checkpoint  # noqa: E402


def render(model, batch, back):
    out = model(batch["inp"])
    if model.mode == "hybrid":
        # classical colour back-end always in fp32 (pow'(x) near 0 overflows in fp16)
        with torch.autocast(device_type=out.device.type, enabled=False):
            lin = out.float() / batch["k"].view(-1, 1, 1, 1)
            return back(lin.clamp_min(0), batch["ccm"].float())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--iters", type=int, default=30000)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--patch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--teacher", default=None)
    ap.add_argument("--kd", type=float, default=0.0)
    ap.add_argument("--val_every", type=int, default=1000)
    ap.add_argument("--time_budget_min", type=float, default=1e9)
    ap.add_argument("--init", default=None)
    ap.add_argument("--scale_prob", type=float, default=0.0, help="probability of a 2x-binned training patch")
    ap.add_argument("--grad_w", type=float, default=0.0, help="weight of the L1 image-gradient (edge) loss")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    model = build_model(a.model).to(dev)
    if a.init:
        model.load_state_dict(load_checkpoint(a.init).state_dict())
    mode = model.mode
    print(f"model={a.model} params={count_params(model)/1e6:.3f}M device={dev}", flush=True)
    teacher = load_checkpoint(a.teacher).to(dev).eval() if a.teacher else None
    back = TorchColorBack().to(dev)

    ds = SIDDPatches(a.data, "train", a.patch, mode, length=a.iters * a.bs, scale_prob=a.scale_prob)
    vds = SIDDPatches(a.data, "val", a.patch, mode, augment=False)
    val = fixed_val_crops(vds, per_image=6)
    val_x2 = fixed_val_crops(vds, per_image=6, scaled=True)  # sharp-per-pixel validation (detail retention)
    print(f"train recs={len(ds.recs)} val crops={len(val)}", flush=True)
    kw = dict(persistent_workers=True, prefetch_factor=4) if a.workers > 0 else {}
    dl = DataLoader(ds, batch_size=a.bs, num_workers=a.workers, pin_memory=dev == "cuda", **kw)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)

    def lr_at(i, elapsed_min):
        prog = min(1.0, max(i / a.iters, elapsed_min / a.time_budget_min))
        return a.lr * min(1.0, (i + 1) / 500) * (0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * prog)))
    scaler = torch.amp.GradScaler(enabled=dev == "cuda")
    best, best_comb, log = -1.0, -1.0, []
    t0 = time.time()
    it = 0
    model.train()
    for batch in dl:
        batch = {k: v.to(dev, non_blocking=True) for k, v in batch.items()}
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=dev == "cuda"):
            pred = render(model, batch, back)
        pred = pred.float()
        loss = F.l1_loss(pred, batch["gt"])
        if a.grad_w > 0:  # edge loss: match horizontal/vertical image gradients
            gx = lambda z: z[..., :, 1:] - z[..., :, :-1]
            gy = lambda z: z[..., 1:, :] - z[..., :-1, :]
            loss = loss + a.grad_w * (F.l1_loss(gx(pred), gx(batch["gt"])) + F.l1_loss(gy(pred), gy(batch["gt"])))
        if teacher is not None and a.kd > 0:
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=dev == "cuda"):
                tp = render(teacher, batch, back).float()
            loss = loss + a.kd * F.l1_loss(pred, tp)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        it += 1
        for gparam in opt.param_groups:
            gparam["lr"] = lr_at(it, (time.time() - t0) / 60)
        if it % 100 == 0:
            print(f"it {it} loss {loss.item():.4f} lr {opt.param_groups[0]['lr']:.2e} {time.time()-t0:.0f}s", flush=True)
        stop = it >= a.iters or (time.time() - t0) / 60 > a.time_budget_min
        if it % a.val_every == 0 or stop:
            model.eval()

            def score(items):
                ps = []
                with torch.no_grad():
                    for v in items:
                        vb = {k: x.unsqueeze(0).to(dev) for k, x in v.items()}
                        p = render(model, vb, back).clamp(0, 1)
                        ps.append((10 * torch.log10(1 / F.mse_loss(p, vb["gt"]))).item())
                return sum(ps) / len(ps)
            psnr, psnr_x2 = score(val), score(val_x2)
            log.append({"it": it, "val_psnr": psnr, "val_psnr_x2": psnr_x2, "loss": loss.item(),
                        "time_s": time.time() - t0})
            print(f"== it {it} val PSNR {psnr:.3f}  2x-binned {psnr_x2:.3f}", flush=True)
            ck = {"config": model.config(), "state_dict": model.state_dict(), "it": it, "val_psnr": psnr,
                  "name": a.model}
            torch.save(ck, out / "last.pt")
            if psnr > best:
                best = psnr
                torch.save(ck, out / "best.pt")
            comb = 0.5 * (psnr + psnr_x2)  # native + sharp-per-pixel validation
            if comb > best_comb:
                best_comb = comb
                torch.save(dict(ck, val_psnr_x2=psnr_x2), out / "best_combined.pt")
            json.dump(log, open(out / "log.json", "w"), indent=1)
            model.train()
        if stop:
            break
    print(f"DONE best val PSNR {best:.3f}", flush=True)


if __name__ == "__main__":
    main()
