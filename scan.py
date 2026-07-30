"""
scan.py — Hyperliquid 고래 스캐너 (단일 파일, 의존성 requests 하나)

GitHub Actions에서 돌리고 결과 whales.json을 저장소에 커밋한다.
환경변수로 조절: MIN_CAP / TOP_N / WORKERS

로컬 실행도 가능:
    python scan.py
    MIN_CAP=5000000 TOP_N=30 python scan.py
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import requests

INFO_URL = "https://api.hyperliquid.xyz/info"
LB_URL = "https://stats-data.hyperliquid.xyz/Mainnet/leaderboard"
WINDOWS = ["day", "week", "month", "allTime"]
HEADERS = {"Content-Type": "application/json"}

MIN_CAP = float(os.getenv("MIN_CAP", "1000000"))
TOP_N = int(os.getenv("TOP_N", "25"))
WORKERS = int(os.getenv("WORKERS", "3"))
OUT = os.getenv("OUT", "whales.json")


# ── 유틸 ─────────────────────────────────────────────────────
def f(x, d=0.0):
    try:
        v = float(x)
        return v if math.isfinite(v) else d
    except (TypeError, ValueError):
        return d


def clamp01(x):
    return max(0.0, min(1.0, x))


def log_norm(v, lo, hi):
    return 0.0 if v <= lo else clamp01(math.log(v / lo) / math.log(hi / lo))


session = requests.Session()


def post(payload, retries=3):
    last = None
    for i in range(retries):
        try:
            r = session.post(INFO_URL, headers=HEADERS, json=payload, timeout=15)
            if r.status_code == 429:
                time.sleep(2 ** i)
                continue
            r.raise_for_status()
            return r.json()
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(0.5 * (2 ** i))
    raise RuntimeError(f"{payload.get('type')} 실패: {last}")


# ── 1단계: 리더보드 ──────────────────────────────────────────
def parse_lb_row(row):
    perf = {}
    for wp in row.get("windowPerformances") or []:
        if isinstance(wp, list) and len(wp) == 2 and isinstance(wp[1], dict):
            perf[wp[0]] = {
                "pnl": f(wp[1].get("pnl")),
                "roi": f(wp[1].get("roi")),
                "vlm": f(wp[1].get("vlm")),
            }
    av = f(row.get("accountValue"))
    return {
        "address": str(row.get("ethAddress", "")).lower(),
        "name": row.get("displayName") or None,
        "accountValue": av,
        "perf": perf,
        "positivePeriods": sum(1 for k in WINDOWS if perf.get(k, {}).get("roi", 0) > 0),
        "turnover": (perf.get("month", {}).get("vlm", 0) / av) if av > 0 else 0.0,
    }


def fetch_leaderboard():
    r = session.get(LB_URL, timeout=90)
    r.raise_for_status()
    j = r.json()
    rows = j.get("leaderboardRows") or j.get("rows") or (j if isinstance(j, list) else [])
    return [parse_lb_row(x) for x in rows if x.get("ethAddress")]


def pre_score(r):
    size = log_norm(r["accountValue"], 1e5, 5e7)
    periods = r["positivePeriods"] / 4
    roi = clamp01((r["perf"].get("month", {}).get("roi", 0) + 0.1) / 0.6)
    return (0.35 * size + 0.30 * periods + 0.35 * roi) * 100


# ── 2단계: 정밀 분석 ─────────────────────────────────────────
def max_drawdown(curve):
    peak, mdd, dd = -math.inf, 0.0, []
    for v in curve:
        peak = max(peak, v)
        d = (peak - v) / peak if peak > 0 else 0.0
        dd.append(d)
        mdd = max(mdd, d)
    return mdd, dd


def downsample(seq, n=90):
    if len(seq) <= n:
        return seq
    step = len(seq) / n
    return [seq[min(len(seq) - 1, int(i * step))] for i in range(n)]


def analyze_fills(fills):
    trades = [
        {"coin": x.get("coin"), "time": int(x.get("time", 0)),
         "pnl": f(x.get("closedPnl")) - f(x.get("fee"))}
        for x in fills
        if f(x.get("closedPnl")) != 0 or "Close" in str(x.get("dir", ""))
    ]
    if not trades:
        return {"n": 0, "winRate": 0.0, "pf": 0.0, "net": 0.0,
                "top1Share": 0.0, "avgHold": None}

    wins = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] < 0]
    gw = sum(t["pnl"] for t in wins)
    gl = abs(sum(t["pnl"] for t in losses))

    opens, holds = {}, []
    for x in sorted(fills, key=lambda y: int(y.get("time", 0))):
        d, c = str(x.get("dir", "")), x.get("coin")
        if "Open" in d:
            opens[c] = int(x.get("time", 0))
        elif "Close" in d and c in opens:
            holds.append(int(x.get("time", 0)) - opens.pop(c))

    return {
        "n": len(trades),
        "winRate": len(wins) / len(trades),
        "pf": (gw / gl) if gl > 0 else (3.0 if gw > 0 else 0.0),
        "net": sum(t["pnl"] for t in trades),
        "top1Share": clamp01(max(t["pnl"] for t in trades) / gw) if gw > 0 else 0.0,
        "avgHold": (sum(holds) / len(holds)) if holds else None,
    }


def mark_prices():
    try:
        mc = post({"type": "metaAndAssetCtxs"})
        universe = mc[0].get("universe", [])
        return {a["name"]: f(c.get("markPx")) for a, c in zip(universe, mc[1])}
    except Exception:  # noqa: BLE001
        return {}


def deep_scan(cand, marks):
    a = cand["address"]
    try:
        state = post({"type": "clearinghouseState", "user": a})
        pf_raw = post({"type": "portfolio", "user": a})
        fills = post({"type": "userFills", "user": a, "aggregateByTime": True})
    except Exception as e:  # noqa: BLE001
        print(f"  ! {a[:10]} 실패: {e}", file=sys.stderr)
        return None

    periods = ({p[0]: p[1] for p in pf_raw if isinstance(p, list) and len(p) == 2}
               if isinstance(pf_raw, list) else {})
    hist = ((periods.get("allTime") or {}).get("accountValueHistory")
            or (periods.get("month") or {}).get("accountValueHistory") or [])
    curve = downsample([f(p[1]) for p in hist] or
                       [cand["accountValue"], cand["accountValue"]], 90)
    mdd, dd = max_drawdown(curve)
    st = analyze_fills(fills if isinstance(fills, list) else [])

    positions = []
    for ap in state.get("assetPositions", []):
        p = ap.get("position", {})
        szi = f(p.get("szi"))
        if szi == 0:
            continue
        coin = p.get("coin", "?")
        positions.append({
            "coin": coin,
            "side": "LONG" if szi > 0 else "SHORT",
            "notional": round(f(p.get("positionValue")), 2),
            "unrealizedPnl": round(f(p.get("unrealizedPnl")), 2),
            "roe": round(f(p.get("returnOnEquity")), 4),
            "leverage": f((p.get("leverage") or {}).get("value")),
            "markPx": marks.get(coin, 0.0),
        })
    positions.sort(key=lambda x: -x["notional"])

    return {
        "address": a,
        "label": cand["name"],
        "accountValue": round(f(state.get("marginSummary", {}).get("accountValue"))
                              or cand["accountValue"], 2),
        "curve": [round(v, 2) for v in curve],
        "ddSeries": [round(v, 4) for v in dd],
        "mdd": round(mdd, 4),
        "rois": {k: round(cand["perf"].get(k, {}).get("roi", 0.0), 4) for k in WINDOWS},
        "positivePeriods": cand["positivePeriods"],
        "turnover": round(cand["turnover"], 2),
        "winRate": round(st["winRate"], 4),
        "pf": round(min(st["pf"], 99), 3),
        "netPnl": round(st["net"], 2),
        "tradeCount": st["n"],
        "top1Share": round(st["top1Share"], 4),
        "avgHold": st["avgHold"],
        "positions": positions[:6],
    }


def main():
    t0 = time.time()
    print(f"설정 · 최소자본 ${MIN_CAP:,.0f} · 상위 {TOP_N}명 · 동시 {WORKERS}")

    print("1단계 · 리더보드")
    rows = fetch_leaderboard()
    pool = [r for r in rows if r["accountValue"] >= MIN_CAP]
    print(f"  전체 {len(rows):,}명 → 자본 조건 통과 {len(pool):,}명")

    for r in pool:
        r["pre"] = pre_score(r)
    cands = sorted(pool, key=lambda r: -r["pre"])[:TOP_N]
    if not cands:
        print("조건에 맞는 지갑 없음. MIN_CAP을 낮춰라.", file=sys.stderr)
        sys.exit(1)

    print(f"2단계 · 정밀 분석 {len(cands)}명 (요청 {len(cands) * 3}회)")
    marks = mark_prices()
    out = []
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for i, res in enumerate(ex.map(lambda c: deep_scan(c, marks), cands), 1):
            if res:
                out.append(res)
            if i % 5 == 0 or i == len(cands):
                print(f"  {i}/{len(cands)}")
            time.sleep(0.12)

    payload = {
        "generatedAt": int(time.time() * 1000),
        "source": "hyperliquid-leaderboard",
        "filters": {"minCap": MIN_CAP, "top": TOP_N},
        "scanned": len(rows),
        "passedCap": len(pool),
        "whales": out,
    }
    blob = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    with open(OUT, "w", encoding="utf-8") as fp:
        fp.write(blob)

    print(f"완료 · {len(out)}개 지갑 · {len(blob)/1024:.0f}KB · {time.time()-t0:.0f}초 → {OUT}")


if __name__ == "__main__":
    main()
