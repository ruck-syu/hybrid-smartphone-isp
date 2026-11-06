"""HTTP server for the Hybrid ISP demo: a small JSON API plus the static single-page front end.

  python app/server.py            # http://localhost:7860
"""
from __future__ import annotations

import asyncio
import functools
import json
import os
import sys
import tempfile
import traceback
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

sys.path.insert(0, str(Path(__file__).resolve().parent))
import core  # noqa: E402

STATIC = Path(__file__).resolve().parent / "static"
MAX_UPLOAD_MB = 80
app = FastAPI(title="Hybrid ISP", docs_url=None, redoc_url=None)
_slots = asyncio.Semaphore(int(os.environ.get("ISP_CONCURRENCY", "2")))


@functools.lru_cache(maxsize=1)
def _samples():
    return core.list_samples()


@functools.lru_cache(maxsize=24)
def _develop_sample(sample: str, pids: tuple, cond: tuple, stages: bool, view: str) -> str:
    """Bundled samples are deterministic, so their results are cached (instant repeat views)."""
    raw, gt, label = core.load_input(None, sample)
    out = core.develop(raw, gt, list(pids), dict(cond), stages, view)
    out["source"] = label
    return json.dumps(out)


@functools.lru_cache(maxsize=1)
def _benchmark():
    return core.benchmark()


@app.get("/api/meta")
def meta():
    return {"pipelines": core.available(), "samples": _samples(), "max_mp": core.MAX_MP,
            "repo": "https://github.com/ruck-syu/hybrid-smartphone-isp"}


@app.get("/api/benchmark")
def bench():
    b = _benchmark()
    if b is None:
        raise HTTPException(404, "Benchmark results are not bundled with this deployment.")
    return b


@app.post("/api/develop")
async def develop_endpoint(file: UploadFile | None = File(None), sample: str | None = Form(None),
                           pipelines: str = Form("traditional,hybrid"), noise: float = Form(1.0),
                           exposure: float = Form(0.0), wb_error: float = Form(0.0),
                           drop_metadata: bool = Form(False), stages: bool = Form(True),
                           view: str = Form("detail")):
    view = "full" if view == "full" else "detail"
    pids = [p for p in pipelines.split(",") if p]
    if not pids:
        raise HTTPException(400, "Pick at least one pipeline to develop with.")
    cond = {"noise_mult": float(min(max(noise, 1), 8)), "exposure_stops": float(min(max(exposure, -4), 0)),
            "wb_err": float(min(max(wb_error, -0.3), 0.3)), "drop_meta": bool(drop_metadata)}
    tmp = None
    try:
        if file is not None and file.filename:
            data = await file.read()
            if len(data) > MAX_UPLOAD_MB * 1e6:
                raise HTTPException(413, f"That file is larger than {MAX_UPLOAD_MB} MB.")
            suffix = Path(file.filename).suffix.lower() or ".dng"
            if suffix in (".jpg", ".jpeg", ".png", ".heic", ".heif", ".webp", ".gif"):
                raise HTTPException(415, "That is a processed image, not a RAW file. Upload a DNG or another "
                                         "camera RAW (CR2, NEF, ARW, ORF, RW2).")
            tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
            tmp.write(data)
            tmp.close()
            path = tmp.name
        else:
            path = None
        async with _slots:
            loop = asyncio.get_running_loop()
            if path is None and sample:
                body = await loop.run_in_executor(
                    None, _develop_sample, sample, tuple(pids), tuple(sorted(cond.items())), stages, view)
                return Response(body, media_type="application/json")

            def work():
                raw, gt, label = core.load_input(path, sample)
                out = core.develop(raw, gt, pids, cond, stages, view)
                out["source"] = file.filename if path else label
                return out
            return JSONResponse(await loop.run_in_executor(None, work))
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(422, str(e))
    except Exception as e:  # LibRaw errors, corrupt files, unsupported sensors
        traceback.print_exc()
        msg = str(e) or type(e).__name__
        if "LibRaw" in type(e).__name__ or "unsupported" in msg.lower() or "corrupt" in msg.lower():
            msg = "This file could not be read as a camera RAW. It may be corrupt or from an unsupported sensor."
        raise HTTPException(422, msg)
    finally:
        if tmp is not None:
            Path(tmp.name).unlink(missing_ok=True)


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


app.mount("/static", StaticFiles(directory=STATIC), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 7860)))
