"""Report-grade offline vLLM benchmark with warmup exclusion and trial stats."""
import asyncio
import json
import os
import random
import statistics
import subprocess
import time
from datetime import datetime

import aiohttp


MODEL = os.getenv("MODEL", "/models/Llama-3.1-70B-Instruct-FP8")
VLLM_URL = os.getenv("VLLM_URL", "")
PROMPTS_FILE = os.getenv("PROMPTS_FILE", "prompts.json")
GPU_INDEX = os.getenv("GPU_INDEX", "0,1,2,3,4,5,6,7")

DURATION_S = int(os.getenv("DURATION", "600"))
WARMUP_S = int(os.getenv("WARMUP_S", "60"))
MAX_CONCURRENT = int(os.getenv("CONCURRENCIES", "512"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "1024"))
TRIALS = int(os.getenv("TRIALS", "3"))
RANDOM_SEED = int(os.getenv("SEED", "42"))
POWER_SAMPLE_S = float(os.getenv("POWER_SAMPLE_S", "1.0"))
REQUEST_TIMEOUT_S = int(os.getenv("REQUEST_TIMEOUT_S", "300"))
LABEL = os.getenv("LABEL", "exp_inference_only_report_grade")


def parse_gpu_indices(index_str):
    parsed = []
    for token in index_str.split(","):
        token = token.strip()
        if token.isdigit():
            parsed.append(int(token))
    return parsed or [0]


def load_prompts(path):
    with open(path, "r") as f:
        data = json.load(f)

    prompts = []
    prompts_by_class = {}

    if isinstance(data, list):
        for item in data:
            if isinstance(item, str):
                prompt = item.strip()
                length_class = ""
            elif isinstance(item, dict):
                prompt = str(item.get("prompt", "")).strip()
                length_class = str(item.get("length_class", "")).strip().lower()
            else:
                continue
            if not prompt:
                continue
            prompts.append(prompt)
            if length_class:
                prompts_by_class.setdefault(length_class, []).append(prompt)

    if not prompts:
        raise RuntimeError(f"no valid prompts found in {path}")
    return prompts, prompts_by_class


def percentile(sorted_vals, pct):
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    k = (len(sorted_vals) - 1) * pct
    lo = int(k)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = k - lo
    return float(sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * frac)


def _extract_first_number(value):
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        cleaned = value.replace(",", "")
        num = []
        seen_dot = False
        started = False
        for ch in cleaned:
            if ch.isdigit():
                num.append(ch)
                started = True
            elif ch == "." and started and not seen_dot:
                num.append(ch)
                seen_dot = True
            elif started:
                break
        if num and any(c.isdigit() for c in num):
            try:
                return float("".join(num))
            except ValueError:
                return None
    return None


def _pick_value(dct, preferred_keys):
    # Prefer exact field names when possible; fall back to fuzzy matching.
    for k in preferred_keys:
        if k in dct:
            val = _extract_first_number(dct[k])
            if val is not None:
                return val
    lowered = {str(k).lower(): k for k in dct.keys()}
    for pref in preferred_keys:
        pref_l = pref.lower()
        for lk, orig in lowered.items():
            if pref_l in lk:
                val = _extract_first_number(dct[orig])
                if val is not None:
                    return val
    return 0.0


def read_rocm_json():
    by_idx = {}
    try:
        result = subprocess.run(
            ["rocm-smi", "--showpower", "--showtemp", "--showsclk", "--json"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if result.returncode != 0 or not result.stdout.strip():
            return by_idx

        payload = json.loads(result.stdout)
        if not isinstance(payload, dict):
            return by_idx

        for key, val in payload.items():
            if not isinstance(val, dict):
                continue
            key_l = str(key).lower().strip()
            if not key_l.startswith("card"):
                continue
            idx_str = key_l.replace("card", "").strip()
            if not idx_str.isdigit():
                continue
            idx = int(idx_str)
            power = _pick_value(
                val,
                [
                    "Average Graphics Package Power (W)",
                    "Current Socket Graphics Package Power (W)",
                    "Graphics Package Power (W)",
                ],
            )
            temp = _pick_value(
                val,
                [
                    "Temperature (Sensor edge) (C)",
                    "Temperature (Sensor junction) (C)",
                    "Temperature (Sensor memory) (C)",
                    "Temperature (C)",
                ],
            )
            sclk = _pick_value(
                val,
                [
                    "sclk clock speed:",
                    "sclk clock speed",
                    "sclk",
                ],
            )
            by_idx[idx] = {"power": power, "temp": temp, "sclk": sclk}
    except Exception:
        return {}
    return by_idx


class PromptStream:
    """Deterministic prompt stream for reproducible runs."""

    def __init__(self, prompts, prompts_by_class, seed):
        self.prompts = prompts
        self.prompts_by_class = prompts_by_class
        self.prompt_classes = [
            name for name in ("short", "medium", "long") if prompts_by_class.get(name)
        ]
        if not self.prompt_classes:
            self.prompt_classes = sorted(list(prompts_by_class.keys()))
        self.rng = random.Random(seed)
        self.cls_idx = 0

    def next_prompt(self):
        if self.prompt_classes:
            # Round-robin classes to keep short/medium/long balanced.
            selected_class = self.prompt_classes[self.cls_idx % len(self.prompt_classes)]
            self.cls_idx += 1
            return self.rng.choice(self.prompts_by_class[selected_class])
        return self.rng.choice(self.prompts)


class TrialState:
    def __init__(self, trial_index, seed, gpu_indices):
        self.trial_index = trial_index
        self.seed = seed
        self.gpu_indices = gpu_indices
        self.start_mono = 0.0
        self.end_mono = 0.0
        self.warmup_cutoff = 0.0
        self.stop_at = 0.0

        self.total_requests = 0
        self.total_success = 0
        self.total_fail = 0

        self.events = []  # {"t": monotonic, "pt": int, "ct": int, "lat_ms": float, "ok": bool}
        self.power_samples = []  # {"t": monotonic, "pwr": float, "temp": float, "sclk": float}
        self.per_gpu_power = {idx: [] for idx in gpu_indices}
        self.per_gpu_peak_power = {idx: 0.0 for idx in gpu_indices}

    def add_request_event(self, now_t, pt, ct, lat_ms, ok):
        self.total_requests += 1
        if ok:
            self.total_success += 1
        else:
            self.total_fail += 1
        self.events.append({"t": now_t, "pt": pt, "ct": ct, "lat_ms": lat_ms, "ok": ok})

    def add_power_sample(self, now_t, by_idx):
        selected = [by_idx.get(idx, {"power": 0.0, "temp": 0.0, "sclk": 0.0}) for idx in self.gpu_indices]
        pwr = sum(g["power"] for g in selected)
        temps = [g["temp"] for g in selected if g["temp"] > 0]
        sclk = [g["sclk"] for g in selected if g["sclk"] > 0]
        temp = sum(temps) / len(temps) if temps else 0.0
        sclk_avg = sum(sclk) / len(sclk) if sclk else 0.0

        self.power_samples.append({"t": now_t, "pwr": pwr, "temp": temp, "sclk": sclk_avg})

        for idx in self.gpu_indices:
            gp = by_idx.get(idx, {"power": 0.0})["power"]
            self.per_gpu_power[idx].append(gp)
            if gp > self.per_gpu_peak_power[idx]:
                self.per_gpu_peak_power[idx] = gp

    def compute_metrics(self):
        measurement_start = self.warmup_cutoff
        measurement_end = self.end_mono
        active_s = max(0.0, measurement_end - measurement_start)

        measured_events = [e for e in self.events if e["t"] >= measurement_start]
        completion_tokens = sum(e["ct"] for e in measured_events)
        prompt_tokens = sum(e["pt"] for e in measured_events)
        ok_events = [e for e in measured_events if e["ok"] and e["ct"] > 0]
        avg_lat = sum(e["lat_ms"] for e in ok_events) / len(ok_events) if ok_events else 0.0

        tps = completion_tokens / active_s if active_s > 0 else 0.0
        rps = len(ok_events) / active_s if active_s > 0 else 0.0

        # True energy integral from power samples using trapezoid integration.
        sorted_samples = sorted(self.power_samples, key=lambda x: x["t"])
        energy_j = 0.0
        if len(sorted_samples) >= 2:
            for i in range(1, len(sorted_samples)):
                t0 = sorted_samples[i - 1]["t"]
                t1 = sorted_samples[i]["t"]
                if t1 <= measurement_start or t0 >= measurement_end:
                    continue
                seg_start = max(t0, measurement_start)
                seg_end = min(t1, measurement_end)
                if seg_end <= seg_start:
                    continue
                p0 = sorted_samples[i - 1]["pwr"]
                p1 = sorted_samples[i]["pwr"]
                # Linear interpolation for clipped segment endpoints.
                ratio0 = (seg_start - t0) / (t1 - t0) if t1 != t0 else 0.0
                ratio1 = (seg_end - t0) / (t1 - t0) if t1 != t0 else 0.0
                ps = p0 + (p1 - p0) * ratio0
                pe = p0 + (p1 - p0) * ratio1
                energy_j += 0.5 * (ps + pe) * (seg_end - seg_start)

        avg_power = energy_j / active_s if active_s > 0 else 0.0
        tok_per_watt = tps / avg_power if avg_power > 0 else 0.0

        in_window_samples = [s for s in sorted_samples if measurement_start <= s["t"] <= measurement_end]
        peak_power = max((s["pwr"] for s in in_window_samples), default=0.0)
        avg_temp = (
            sum(s["temp"] for s in in_window_samples if s["temp"] > 0)
            / max(1, sum(1 for s in in_window_samples if s["temp"] > 0))
        )
        avg_sclk = (
            sum(s["sclk"] for s in in_window_samples if s["sclk"] > 0)
            / max(1, sum(1 for s in in_window_samples if s["sclk"] > 0))
        )

        per_gpu_summary = {}
        for idx in self.gpu_indices:
            vals = [v for v in self.per_gpu_power[idx]]
            per_gpu_summary[str(idx)] = {
                "peak_power_w": round(self.per_gpu_peak_power[idx], 2),
                "avg_power_w": round(sum(vals) / len(vals), 2) if vals else 0.0,
                "samples": len(vals),
            }

        return {
            "trial": self.trial_index,
            "seed": self.seed,
            "duration_s": round(self.end_mono - self.start_mono, 3),
            "warmup_s": WARMUP_S,
            "active_s": round(active_s, 3),
            "measurement_start_s_from_trial_start": round(measurement_start - self.start_mono, 3),
            "measurement_end_s_from_trial_start": round(measurement_end - self.start_mono, 3),
            "total_requests": self.total_requests,
            "successful_requests": self.total_success,
            "failed_requests": self.total_fail,
            "measured_successful_requests": len(ok_events),
            "completion_tokens": completion_tokens,
            "prompt_tokens": prompt_tokens,
            "throughput_tokens_per_sec": round(tps, 4),
            "requests_per_sec": round(rps, 4),
            "avg_latency_ms": round(avg_lat, 3),
            "energy_joules": round(energy_j, 3),
            "avg_power_w": round(avg_power, 3),
            "peak_power_w": round(peak_power, 3),
            "avg_temp_c": round(avg_temp, 3),
            "avg_sclk_mhz": round(avg_sclk, 3),
            "tokens_per_watt": round(tok_per_watt, 6),
            "power_samples_total": len(self.power_samples),
            "per_gpu": per_gpu_summary,
        }


async def send_request(session, semaphore, state, prompt_stream):
    prompt = prompt_stream.next_prompt()
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "max_tokens": MAX_TOKENS,
        # Greedy decoding for deterministic benchmark behavior.
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": state.seed,
    }

    async with semaphore:
        t1 = time.monotonic()
        pt = 0
        ct = 0
        ok = False
        try:
            async with session.post(
                VLLM_URL,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT_S),
            ) as resp:
                data = await resp.json()
                usage = data.get("usage", {})
                pt = int(usage.get("prompt_tokens", 0) or 0)
                ct = int(usage.get("completion_tokens", 0) or 0)
                ok = ct > 0 and resp.status < 400
        except Exception:
            ok = False
        t2 = time.monotonic()
        state.add_request_event(now_t=t2, pt=pt, ct=ct, lat_ms=(t2 - t1) * 1000.0, ok=ok)


async def power_monitor(state):
    while time.monotonic() < state.stop_at + 2:
        now_t = time.monotonic()
        by_idx = read_rocm_json()
        state.add_power_sample(now_t=now_t, by_idx=by_idx)

        elapsed = now_t - state.start_mono
        rem = max(0.0, state.stop_at - now_t)
        print(
            f"  [trial {state.trial_index}] t={elapsed:6.1f}s "
            f"| power={state.power_samples[-1]['pwr']:7.1f}W "
            f"| ok={state.total_success:6d} fail={state.total_fail:5d} "
            f"| rem={rem:6.1f}s",
            flush=True,
        )
        await asyncio.sleep(POWER_SAMPLE_S)


async def request_loop(session, semaphore, state, prompt_stream):
    pending = set()
    while time.monotonic() < state.stop_at:
        while len(pending) < MAX_CONCURRENT and time.monotonic() < state.stop_at:
            task = asyncio.create_task(send_request(session, semaphore, state, prompt_stream))
            pending.add(task)
            task.add_done_callback(pending.discard)
        await asyncio.sleep(0.01)
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)


async def run_trial(trial_idx, gpu_indices, prompts, prompts_by_class):
    seed = RANDOM_SEED + trial_idx - 1
    random.seed(seed)
    prompt_stream = PromptStream(prompts, prompts_by_class, seed=seed)
    state = TrialState(trial_index=trial_idx, seed=seed, gpu_indices=gpu_indices)

    state.start_mono = time.monotonic()
    state.warmup_cutoff = state.start_mono + WARMUP_S
    state.stop_at = state.start_mono + DURATION_S

    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    connector = aiohttp.TCPConnector(limit=MAX_CONCURRENT, keepalive_timeout=300)

    async with aiohttp.ClientSession(connector=connector) as session:
        monitor_task = asyncio.create_task(power_monitor(state))
        request_task = asyncio.create_task(request_loop(session, semaphore, state, prompt_stream))
        await asyncio.gather(request_task, monitor_task, return_exceptions=True)

    state.end_mono = time.monotonic()
    return state.compute_metrics()


def summarize_trials(trials):
    tpw = sorted(t["tokens_per_watt"] for t in trials)
    tps = sorted(t["throughput_tokens_per_sec"] for t in trials)
    avgp = sorted(t["avg_power_w"] for t in trials)

    return {
        "trial_count": len(trials),
        "tokens_per_watt": {
            "median": round(statistics.median(tpw), 6),
            "p10": round(percentile(tpw, 0.10), 6),
            "p90": round(percentile(tpw, 0.90), 6),
        },
        "throughput_tokens_per_sec": {
            "median": round(statistics.median(tps), 4),
            "p10": round(percentile(tps, 0.10), 4),
            "p90": round(percentile(tps, 0.90), 4),
        },
        "avg_power_w": {
            "median": round(statistics.median(avgp), 3),
            "p10": round(percentile(avgp, 0.10), 3),
            "p90": round(percentile(avgp, 0.90), 3),
        },
    }


async def main():
    if not VLLM_URL:
        raise SystemExit(
            "VLLM_URL is not set. Look up the lmstack-router ClusterIP and export "
            "VLLM_URL=http://<router-ip>/v1/completions — see README.md in this directory."
        )

    gpu_indices = parse_gpu_indices(GPU_INDEX)
    prompts, prompts_by_class = load_prompts(PROMPTS_FILE)

    print("=" * 72)
    print(" REPORT-GRADE OFFLINE vLLM BENCHMARK")
    print("=" * 72)
    print(f" model={MODEL}")
    print(f" url={VLLM_URL}")
    print(
        f" trials={TRIALS} duration={DURATION_S}s warmup={WARMUP_S}s "
        f"concurrency={MAX_CONCURRENT} max_tokens={MAX_TOKENS}"
    )
    print(f" power_sample_s={POWER_SAMPLE_S} seed_base={RANDOM_SEED} gpus={gpu_indices}")
    print(
        " prompt pools: "
        f"short={len(prompts_by_class.get('short', []))} "
        f"medium={len(prompts_by_class.get('medium', []))} "
        f"long={len(prompts_by_class.get('long', []))} "
        f"total={len(prompts)}"
    )
    print("=" * 72)

    trials = []
    for trial_idx in range(1, TRIALS + 1):
        print(f"\n--- Trial {trial_idx}/{TRIALS} ---")
        result = await run_trial(trial_idx, gpu_indices, prompts, prompts_by_class)
        trials.append(result)
        print(
            f" trial={trial_idx} tpw={result['tokens_per_watt']:.6f} "
            f"tps={result['throughput_tokens_per_sec']:.3f} "
            f"avg_power={result['avg_power_w']:.2f}W "
            f"success={result['successful_requests']} fail={result['failed_requests']}"
        )

    agg = summarize_trials(trials)
    final = {
        "label": LABEL,
        "timestamp": datetime.now().isoformat(),
        "model": MODEL,
        "url": VLLM_URL,
        "gpu_indices": gpu_indices,
        "prompts_file": PROMPTS_FILE,
        "duration_s": DURATION_S,
        "warmup_s": WARMUP_S,
        "max_concurrent": MAX_CONCURRENT,
        "max_tokens": MAX_TOKENS,
        "trials": TRIALS,
        "seed_base": RANDOM_SEED,
        "power_sample_s": POWER_SAMPLE_S,
        "notes": [
            "Warmup excluded from throughput and power calculations.",
            "Average power is based on energy integral of all power samples.",
            "Decoding uses greedy parameters for determinism.",
            "Prompt stream uses deterministic seed and balanced class cycling.",
            "Power comes from rocm-smi JSON fields by key name.",
        ],
        "trial_results": trials,
        "aggregate": agg,
    }

    print("\n" + "=" * 72)
    print(" AGGREGATE (median / p10 / p90)")
    print("=" * 72)
    print(
        " tokens_per_watt: "
        f"{agg['tokens_per_watt']['median']:.6f} / "
        f"{agg['tokens_per_watt']['p10']:.6f} / "
        f"{agg['tokens_per_watt']['p90']:.6f}"
    )
    print(
        " throughput_tok_s: "
        f"{agg['throughput_tokens_per_sec']['median']:.3f} / "
        f"{agg['throughput_tokens_per_sec']['p10']:.3f} / "
        f"{agg['throughput_tokens_per_sec']['p90']:.3f}"
    )
    print(
        " avg_power_w: "
        f"{agg['avg_power_w']['median']:.2f} / "
        f"{agg['avg_power_w']['p10']:.2f} / "
        f"{agg['avg_power_w']['p90']:.2f}"
    )

    run_ts = datetime.now().strftime("%Y%m%dT%H%M%S")
    out_path = f"./{LABEL}_{run_ts}_summary.json"
    with open(out_path, "w") as f:
        json.dump(final, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
