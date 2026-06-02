#!/usr/bin/env python3
"""
Load test for VoxCPM2 streaming server.

Fires exactly N requests in parallel at POST /api/stream — all started at the
same moment, no queuing. Reports per-request TTFT (time-to-first-audio-frame),
total latency, audio duration, and RTF, plus aggregate p50/p90/p95/p99 stats and
throughput. Multiple N's can be swept.

Usage:
    python loadtest.py --url http://localhost:8000 --concurrency 1,2,4,8,16
    python loadtest.py --url http://localhost:8000 --concurrency 8
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import struct
import time
from dataclasses import dataclass, field
from typing import Optional

import httpx


DEFAULT_TEXT = (
    "The quick brown fox jumps over the lazy dog. Pack my box with five dozen "
    "liquor jugs. How vexingly quick daft zebras jump! Sphinx of black quartz, "
    "judge my vow."
)


@dataclass
class Result:
    ok: bool
    ttft_s: Optional[float] = None
    total_s: Optional[float] = None
    audio_s: Optional[float] = None      # generated audio duration
    sample_rate: int = 0
    n_audio_frames: int = 0
    n_audio_samples: int = 0
    error: str = ""
    started_at: float = 0.0
    finished_at: float = 0.0


async def _stream_frames(resp: httpx.Response):
    """Yield (type_byte, payload_bytes) from the length-prefixed frame protocol."""
    buf = bytearray()
    HEADER = 5  # 1 byte type + 4 byte LE length
    async for chunk in resp.aiter_bytes():
        buf.extend(chunk)
        while True:
            if len(buf) < HEADER:
                break
            ftype = buf[0]
            length = struct.unpack_from("<I", buf, 1)[0]
            if len(buf) < HEADER + length:
                break
            payload = bytes(buf[HEADER : HEADER + length])
            del buf[: HEADER + length]
            yield ftype, payload


async def one_request(client: httpx.AsyncClient, url: str, text: str,
                      ref_path: str = "", req_timeout: float = 600.0) -> Result:
    body = {"text": text, "max_len": 4096}
    if ref_path:
        body["reference_wav_path"] = ref_path

    r = Result(ok=False)
    r.started_at = time.perf_counter()
    try:
        async with client.stream(
            "POST",
            f"{url}/api/stream",
            json=body,
            timeout=httpx.Timeout(req_timeout, connect=30.0),
        ) as resp:
            if resp.status_code != 200:
                body_text = (await resp.aread())[:300].decode(errors="replace")
                r.error = f"HTTP {resp.status_code}: {body_text}"
                r.finished_at = time.perf_counter()
                return r

            async for ftype, payload in _stream_frames(resp):
                if ftype == 0:  # JSON
                    try:
                        msg = json.loads(payload.decode())
                    except Exception:
                        continue
                    mtype = msg.get("type")
                    if mtype == "meta":
                        r.sample_rate = int(msg.get("sample_rate", 0))
                    elif mtype == "error":
                        r.error = f"server error: {msg.get('message','?')}"
                        r.finished_at = time.perf_counter()
                        return r
                    elif mtype == "done":
                        # full pass-through
                        pass
                elif ftype == 1:  # float32 PCM audio
                    if r.ttft_s is None:
                        r.ttft_s = time.perf_counter() - r.started_at
                    r.n_audio_frames += 1
                    r.n_audio_samples += len(payload) // 4  # float32

        r.finished_at = time.perf_counter()
        r.total_s = r.finished_at - r.started_at
        if r.sample_rate > 0 and r.n_audio_samples > 0:
            r.audio_s = r.n_audio_samples / r.sample_rate
        r.ok = r.ttft_s is not None and not r.error
        if not r.ok and not r.error:
            r.error = "no audio frames received"
        return r

    except Exception as exc:
        r.finished_at = time.perf_counter()
        r.error = f"{type(exc).__name__}: {exc}"
        return r


def pct(xs: list[float], p: float) -> float:
    if not xs:
        return float("nan")
    xs2 = sorted(xs)
    k = (len(xs2) - 1) * p
    f = int(k)
    c = min(f + 1, len(xs2) - 1)
    return xs2[f] + (xs2[c] - xs2[f]) * (k - f)


async def run_level(url: str, text: str, ref_path: str, concurrency: int,
                    req_timeout: float) -> list[Result]:
    """Fire exactly `concurrency` requests simultaneously — no queuing."""
    limits = httpx.Limits(
        max_connections=concurrency * 2,
        max_keepalive_connections=concurrency * 2,
    )

    async with httpx.AsyncClient(limits=limits, http2=False) as client:
        # Use a barrier so every request crosses the start line together,
        # even if task scheduling is slightly staggered.
        barrier = asyncio.Barrier(concurrency)

        async def worker(_i: int) -> Result:
            await barrier.wait()
            return await one_request(client, url, text, ref_path, req_timeout)

        tasks = [asyncio.create_task(worker(i)) for i in range(concurrency)]
        return await asyncio.gather(*tasks)


def summarize(level: int, results: list[Result], wall_s: float) -> dict:
    ok = [r for r in results if r.ok]
    fail = [r for r in results if not r.ok]
    ttfts = [r.ttft_s for r in ok if r.ttft_s is not None]
    totals = [r.total_s for r in ok if r.total_s is not None]
    audios = [r.audio_s for r in ok if r.audio_s]
    rtfs = [r.audio_s / r.total_s for r in ok if r.audio_s and r.total_s]

    def stats(xs):
        if not xs:
            return {k: float("nan") for k in ("min", "p50", "p90", "p95", "p99", "max", "mean")}
        return {
            "min":  min(xs),
            "p50":  pct(xs, 0.50),
            "p90":  pct(xs, 0.90),
            "p95":  pct(xs, 0.95),
            "p99":  pct(xs, 0.99),
            "max":  max(xs),
            "mean": statistics.fmean(xs),
        }

    return {
        "concurrency":     level,
        "requests_total":  len(results),
        "requests_ok":     len(ok),
        "requests_failed": len(fail),
        "wall_clock_s":    wall_s,
        "throughput_rps":  len(ok) / wall_s if wall_s > 0 else float("nan"),
        "audio_rtf":       stats(rtfs),       # generated_audio_s / total_s; >1 = faster-than-realtime
        "ttft_s":          stats(ttfts),
        "total_s":         stats(totals),
        "audio_s":         stats(audios),
        "errors":          [r.error for r in fail][:10],
    }


def fmt_row(label: str, s: dict) -> str:
    return (
        f"  {label:<10}"
        f" min={s['min']:.3f}  p50={s['p50']:.3f}  p90={s['p90']:.3f}"
        f"  p95={s['p95']:.3f}  p99={s['p99']:.3f}  max={s['max']:.3f}"
        f"  mean={s['mean']:.3f}"
    )


def print_summary(summary: dict) -> None:
    print(f"\n=== concurrency={summary['concurrency']} ===")
    print(
        f"  requests: {summary['requests_ok']}/{summary['requests_total']} ok"
        f" ({summary['requests_failed']} failed)"
        f"   wall={summary['wall_clock_s']:.2f}s"
        f"   throughput={summary['throughput_rps']:.3f} req/s"
    )
    print(fmt_row("ttft (s)",  summary["ttft_s"]))
    print(fmt_row("total (s)", summary["total_s"]))
    print(fmt_row("audio (s)", summary["audio_s"]))
    print(fmt_row("rtf",       summary["audio_rtf"]))
    if summary["errors"]:
        print("  sample errors:")
        for e in summary["errors"]:
            print(f"    - {e}")


async def main_async(args):
    levels = [int(x) for x in args.concurrency.split(",") if x.strip()]
    text = args.text or DEFAULT_TEXT

    # quick health check
    try:
        async with httpx.AsyncClient(timeout=10.0) as c:
            h = await c.get(f"{args.url}/health")
            print(f"health: {h.status_code} {h.text}")
    except Exception as e:
        print(f"WARN: health check failed: {e}")

    all_summaries = []
    for level in levels:
        print(f"\n--- firing {level} requests in parallel ---", flush=True)
        t0 = time.perf_counter()
        results = await run_level(
            url=args.url,
            text=text,
            ref_path=args.ref_path,
            concurrency=level,
            req_timeout=args.req_timeout,
        )
        wall = time.perf_counter() - t0
        summary = summarize(level, results, wall)
        print_summary(summary)
        all_summaries.append(summary)
        if args.cooldown > 0 and level != levels[-1]:
            await asyncio.sleep(args.cooldown)

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(all_summaries, f, indent=2)
        print(f"\nwrote JSON results to {args.json_out}")

    # final compact table
    print("\n=== summary table ===")
    print(f"{'conc':>4} {'ok/total':>10} {'rps':>7}  "
          f"{'ttft p50':>9} {'ttft p90':>9}  "
          f"{'tot p50':>9} {'tot p90':>9}  {'rtf p50':>8}")
    for s in all_summaries:
        print(
            f"{s['concurrency']:>4} "
            f"{s['requests_ok']:>4}/{s['requests_total']:<5}"
            f" {s['throughput_rps']:>6.3f}  "
            f"{s['ttft_s']['p50']:>9.3f} {s['ttft_s']['p90']:>9.3f}  "
            f"{s['total_s']['p50']:>9.3f} {s['total_s']['p90']:>9.3f}  "
            f"{s['audio_rtf']['p50']:>8.3f}"
        )


def main():
    ap = argparse.ArgumentParser(description="VoxCPM2 streaming load test")
    ap.add_argument("--url", default="http://localhost:8000",
                    help="Base URL of the VoxCPM2 server")
    ap.add_argument("--concurrency", default="1,2,4,8,16",
                    help="Comma-separated levels; each fires exactly N parallel requests")
    ap.add_argument("--text", default="",
                    help="Synthesis text (default: built-in pangram set)")
    ap.add_argument("--ref-path", default="",
                    help="Optional server-side reference WAV path (already uploaded)")
    ap.add_argument("--req-timeout", type=float, default=600.0,
                    help="Per-request timeout in seconds")
    ap.add_argument("--cooldown", type=float, default=2.0,
                    help="Seconds to sleep between concurrency levels")
    ap.add_argument("--json-out", default="",
                    help="If set, write per-level summary JSON to this path")
    args = ap.parse_args()

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
