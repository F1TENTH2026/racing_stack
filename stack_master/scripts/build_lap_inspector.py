#!/usr/bin/env python3
r"""
build_lap_inspector.py
======================
Generate the F1TENTH "lap inspector" HTML from ForzaETH race_stack debug logs.

Given the two debug logs plus the map/waypoint files, this produces one
self-contained HTML file with:
  - a Frenet (unrolled-track) view: ego path colored by state, obstacles,
    per-cycle overtake decisions, near-stops
  - a real-map (cartesian) view: the same reconstructed onto the actual track,
    with the occupancy map underlay
  - speed profile + state timeline, all sharing one s-axis, cross-linked hover
  - a per-cycle "how it decided" panel (path_exists/ttl/on_spline/safe/locked)

Paths resolve against machine defaults so you usually pass almost nothing:
  logs -> DEFAULT_LOG_DIR   (bare filename looked up here; a path is used as-is)
  maps -> DEFAULT_MAPS_DIR  (bare map name = a folder holding global_waypoints.json
                             + <name>.yaml; a path is used as the map dir)

Usage:
  # all defaults: latest_state_machine.log, latest_static_avoidance.log, map 0815test3
  python build_lap_inspector.py

  # pick a different map folder and/or specific log files (bare names are enough)
  python build_lap_inspector.py --map 0815test3 \
      --state run12_state_machine.log --static run12_static_avoidance.log

  # full paths still work (e.g. running off-car), plus --out / --log-dir / --maps-dir
  python build_lap_inspector.py --state /tmp/a.log --static /tmp/b.log --map /tmp/mymap

Only the two logs change run-to-run. Stdlib only, no third-party packages.
"""
import argparse, base64, bisect, json, math, re, struct, sys
from pathlib import Path

# Machine defaults (override on the command line with --log-dir / --maps-dir).
# "~" is expanded to the current user's home, so these work regardless of username.
#   logs:  a bare filename is looked up here; a path (or absolute) is used as-is.
#   maps:  a bare map name (e.g. "0815test3") is a folder under here holding
#          global_waypoints.json + <name>.yaml; a path is used as the map dir.
DEFAULT_LOG_DIR  = "~/roboracer_ws/src/racing_stack/logfile"
DEFAULT_MAPS_DIR = "~/roboracer_ws/src/racing_stack/stack_master/maps"


def resolve_log(name_or_path, log_dir):
    """Bare filename -> <log_dir>/<name>; anything with a directory -> as given."""
    p = Path(name_or_path).expanduser()
    if p.is_absolute() or p.parent != Path("."):
        return p
    return Path(log_dir).expanduser() / p


def resolve_map(name_or_path, maps_dir):
    """Map name/dir -> (global_waypoints.json, map yaml) inside that folder."""
    p = Path(name_or_path).expanduser()
    mapdir = p if (p.is_absolute() or p.parent != Path(".")) else Path(maps_dir).expanduser() / p
    if not mapdir.is_dir():
        sys.exit(f"Map directory not found: {mapdir}")
    wp = mapdir / "global_waypoints.json"
    if not wp.exists():
        sys.exit(f"global_waypoints.json not found in {mapdir}")
    yaml = mapdir / (mapdir.name + ".yaml")          # e.g. 0815test3/0815test3.yaml
    if not yaml.exists():                            # fall back to any *.yaml
        cands = sorted(mapdir.glob("*.yaml"))
        if not cands:
            sys.exit(f"No .yaml map file found in {mapdir}")
        yaml = cands[0]
    return wp, yaml


# ----------------------------------------------------------------------------
# 1. state_machine debug log
#      <t> [OBSTACLES]    cur_s= cur_d= vs= n= [id=.. static=.. gap=.. d=.. size=..; ...]
#      <t> [STATE_CHANGE] FROM -> TO src=.. s=.. d=.. vs=..
#      <t> [STATIC_OT]    state=.. path_exists=.. path_ttl_ok=.. path_safe=..
#                         on_spline=.. path_locked=.. speed=.. raw_wpnts=.. decision=.. reason=..
# ----------------------------------------------------------------------------
def parse_state_machine(path):
    ego, changes, sot = [], [], []
    for line in Path(path).read_text().splitlines():
        m = re.match(r"([\d.]+) \[OBSTACLES\] cur_s=([-\d.]+) cur_d=([-\d.]+) "
                     r"vs=([-\d.]+) n=(\d+) \[(.*)\]", line)
        if m:
            t, s, d, vs, n, body = m.groups()
            obs = []
            if body != "none":
                for part in body.split(";"):
                    mm = dict(re.findall(r"(\w+)=([-\d.]+)", part))
                    if "id" in mm:
                        obs.append({
                            "id":     int(mm["id"]),
                            "static": int(float(mm.get("static", 1))),
                            "gap":    float(mm["gap"])  if "gap"  in mm else None,
                            "d":      float(mm["d"])    if "d"    in mm else None,
                            "size":   float(mm["size"]) if "size" in mm else None,
                        })
            ego.append({"t": float(t), "s": float(s), "d": float(d),
                        "vs": float(vs), "n": int(n), "obs": obs})
            continue
        m = re.match(r"([\d.]+) \[STATE_CHANGE\] (\S+) -> (\S+) src=(\S+)", line)
        if m:
            changes.append({"t": float(m.group(1)), "to": m.group(3), "src": m.group(4)})
            continue
        if "[STATIC_OT]" in line and "TRANSITION" not in line:
            d = dict(re.findall(r"(\w+)=([^\s]+)", line))
            sot.append({
                "t": float(line.split()[0]),
                "state":       d.get("state"),
                "path_exists": int(d.get("path_exists", 0)),
                "path_ttl_ok": int(d.get("path_ttl_ok", 0)),
                "on_spline":   int(d.get("on_spline", 0)),
                "path_safe":   int(d.get("path_safe", 0)),
                "path_locked": int(d.get("path_locked", 0)),
                "speed":       float(d.get("speed", 0)),
                "raw_wpnts":   int(d.get("raw_wpnts", 0)),
                "decision":    d.get("decision"),
                "reason":      d.get("reason", ""),
            })
    if not ego:
        sys.exit(f"No [OBSTACLES] lines in {path} - is this a state_machine log?")
    return ego, changes, sot


# ----------------------------------------------------------------------------
# 2. static_avoidance_node debug log ([STATIC_AVOID] lines)
# ----------------------------------------------------------------------------
def parse_static_avoidance(path):
    san = []
    for line in Path(path).read_text().splitlines():
        if "[STATIC_AVOID]" not in line:
            continue
        d = dict(re.findall(r"(\w+)=(-?[\w.]+)", line))
        def f(k):
            try:    return float(d.get(k))
            except (TypeError, ValueError): return None
        san.append({
            "t": float(line.split()[0]),
            "obs_id":   d.get("obs_id"),
            "obs_s":    f("obs_s"),
            "obs_dist": f("obs_dist"),
            "ego_s":    f("ego_s"),
            "side":     d.get("side"),
            "lc":       f("left_clearance"),
            "rc":       f("right_clearance"),
            "valid":    d.get("candidate_valid") == "true",
            "rung":     d.get("rung"),
            "ppts":     int(d.get("path_points", 0)),
            "reason":   d.get("reason", ""),
        })
    return san


# ----------------------------------------------------------------------------
# 3. DATA object: per-sample state, lap segmentation, per-lap metadata, and the
#    nearest STATIC_OT / STATIC_AVOID index per ego sample (for the hover panel).
# ----------------------------------------------------------------------------
def nearest_index(times, t, max_dt=0.6):
    if not times:
        return -1
    i = bisect.bisect_left(times, t)
    best, bd = -1, max_dt
    for j in (i - 1, i):
        if 0 <= j < len(times) and abs(times[j] - t) < bd:
            bd, best = abs(times[j] - t), j
    return best


def build_data(ego, changes, sot, san):
    track_len = round(max(e["s"] for e in ego) + 0.5, 1)   # for s-wrap detection

    ct = [c["t"] for c in changes]
    cs = [c["to"] for c in changes]
    def state_at(t):
        i = bisect.bisect_right(ct, t) - 1
        return cs[i] if i >= 0 else "GB_TRACK"

    sott = [x["t"] for x in sot]
    sant = [x["t"] for x in san]
    for e in ego:
        e["st"]  = state_at(e["t"])
        e["soi"] = nearest_index(sott, e["t"])
        e["sai"] = nearest_index(sant, e["t"])

    lap_bounds, cur, last = [], [], None
    for e in ego:
        if last is not None and last - e["s"] > track_len * 0.5:
            lap_bounds.append({"t0": cur[0]["t"], "t1": cur[-1]["t"]})
            cur = []
        cur.append(e)
        last = e["s"]
    if cur:
        lap_bounds.append({"t0": cur[0]["t"], "t1": cur[-1]["t"]})

    laps = []
    for i, lb in enumerate(lap_bounds):
        seg = [e for e in ego if lb["t0"] <= e["t"] <= lb["t1"]]
        if not seg:
            continue
        laps.append({
            "i": i, "t0": lb["t0"], "t1": lb["t1"],
            "dur":    round(lb["t1"] - lb["t0"], 1),
            "ns":     len(seg),
            "nobs":   sum(1 for e in seg if e["n"] > 0),
            "nstop":  sum(1 for e in seg if e["vs"] < 0.3 and e["n"] > 0),
            "ov":     sum(1 for e in seg if e["st"] == "OVERTAKE"),
            "lost":   sum(1 for e in seg if e["st"] == "LOSTLINE"),
            "moving": (max(e["s"] for e in seg) - min(e["s"] for e in seg)) > 5,
        })

    return {"track_len": track_len, "ego": ego, "changes": changes,
            "sot": sot, "san": san, "lap_bounds": lap_bounds, "laps": laps}


# ----------------------------------------------------------------------------
# 4. MAPDATA object: IQP raceline, track bounds from d_left/d_right + heading,
#    and the occupancy map (base64 + world extent). Enables (s,d) -> world.
# ----------------------------------------------------------------------------
def read_yaml_map(map_yaml):
    txt = Path(map_yaml).read_text()
    res = float(re.search(r"resolution:\s*([-\d.]+)", txt).group(1))
    ox, oy = [float(v) for v in
              re.search(r"origin:\s*\[([-\d.]+),\s*([-\d.]+)", txt).groups()]
    img = re.search(r"image:\s*(\S+)", txt).group(1)
    return res, ox, oy, img


def png_size(png_bytes):
    if png_bytes[:8] != b"\x89PNG\r\n\x1a\n":
        sys.exit("Map image must be a PNG (the browser can't render .pgm).")
    return struct.unpack(">II", png_bytes[16:24])   # IHDR width, height


def build_mapdata(waypoints_json, map_yaml, map_png=None):
    wp = json.loads(Path(waypoints_json).read_text())["global_traj_wpnts_iqp"]["wpnts"]
    rl, left, right = [], [], []
    for w in wp:
        psi = w["psi_rad"]
        nx, ny = -math.sin(psi), math.cos(psi)          # left-pointing normal
        rl.append([round(w["s_m"], 3), round(w["x_m"], 3), round(w["y_m"], 3),
                   round(psi, 4), round(w["vx_mps"], 2)])
        left.append([round(w["x_m"] + w["d_left"]  * nx, 3),
                     round(w["y_m"] + w["d_left"]  * ny, 3)])
        right.append([round(w["x_m"] - w["d_right"] * nx, 3),
                      round(w["y_m"] - w["d_right"] * ny, 3)])

    res, ox, oy, img_name = read_yaml_map(map_yaml)
    png_path = Path(map_png) if map_png else Path(map_yaml).with_name(img_name)
    png_bytes = png_path.read_bytes()
    W, H = png_size(png_bytes)
    extent = [ox, oy, ox + W * res, oy + H * res]        # world [xmin,ymin,xmax,ymax]

    return {"rl": rl, "left": left, "right": right,
            "track_s": round(max(r[0] for r in rl), 3),
            "map": {"b64": base64.b64encode(png_bytes).decode("ascii"),
                    "extent": extent, "w": W, "h": H}}


# ----------------------------------------------------------------------------
# 5. Inject both objects into the template and write the HTML.
# ----------------------------------------------------------------------------
def build_html(data, mapdata):
    j = lambda o: json.dumps(o, separators=(",", ":"))
    return (TEMPLATE
            .replace("__DATA__",    j(data))
            .replace("__MAPDATA__", j(mapdata)))

def extract_run_id(state_log):
    """
    state_machine 로그 첫 부분에서
    '# state_machine debug log started 20260815_214826 map=...'
    형태의 run ID를 추출한다.
    """
    pattern = re.compile(
        r"#\s*state_machine\s+debug\s+log\s+started\s+(\d{8}_\d{6})"
    )

    try:
        with Path(state_log).open("r", encoding="utf-8", errors="ignore") as f:
            for _ in range(50):  # 보통 헤더는 파일 앞부분에 있음
                line = f.readline()
                if not line:
                    break

                m = pattern.search(line)
                if m:
                    return m.group(1)
    except OSError:
        pass

    return None

def main():
    ap = argparse.ArgumentParser(description="Build the F1TENTH lap inspector HTML.")
    ap.add_argument("--state",  default="latest_state_machine.log",
                    help="state_machine log: bare name (looked up in --log-dir) or a path")
    ap.add_argument("--static", default="latest_static_avoidance.log",
                    help="static_avoidance log: bare name (in --log-dir) or a path")
    ap.add_argument("--map",    default="0815test3",
                    help="map name (folder under --maps-dir) or a path to a map dir")
    ap.add_argument("--map-png", default=None, help="override map PNG (default: image: in yaml)")
    ap.add_argument("--out",    default="f1tenth_lap_inspector.html")
    ap.add_argument("--log-dir",  default=DEFAULT_LOG_DIR,  help="base dir for bare log names")
    ap.add_argument("--maps-dir", default=DEFAULT_MAPS_DIR, help="base dir for bare map names")
    a = ap.parse_args()

    state  = resolve_log(a.state,  a.log_dir)
    static = resolve_log(a.static, a.log_dir)
    for f in (state, static):
        if not Path(f).exists():
            sys.exit(f"Log not found: {f}")
    wp, yaml = resolve_map(a.map, a.maps_dir)
    print(f"state  : {state}")
    print(f"static : {static}")
    print(f"map    : {wp.parent}  (yaml: {yaml.name})")

    ego, changes, sot = parse_state_machine(state)
    san = parse_static_avoidance(static)
    data = build_data(ego, changes, sot, san)
    mapdata = build_mapdata(wp, yaml, a.map_png)

    # state_machine 로그의 header에서 run ID 추출
    # 예:
    #   # state_machine debug log started 20260815_214826  map='0815test3'
    # -> 20260815_214826
    # ------------------------------------------------------------
    # Output
    # ------------------------------------------------------------
    run_id = extract_run_id(state)

    if run_id:
        print(f"run_id : {run_id}")

        # 기본 출력 위치:
        # ~/roboracer_ws/src/racing_stack/debugfile/debug_<run_id>.html
        out_dir = Path("~/roboracer_ws/src/racing_stack/debugfile").expanduser()
        out_dir.mkdir(parents=True, exist_ok=True)

        # --out을 따로 지정하지 않았으면 자동 이름 사용
        if a.out == "f1tenth_lap_inspector.html":
            out_path = out_dir / f"debug_{run_id}.html"
        else:
            out_path = Path(a.out).expanduser()
    else:
        print("run_id : not found")
        out_path = Path(a.out).expanduser()

    out_path.parent.mkdir(parents=True, exist_ok=True)

    out_path.write_text(
        build_html(data, mapdata),
        encoding="utf-8"
    )

    print(f"Wrote {out_path}")
    
    print(f"  {len(data['ego'])} ego samples, {len(data['laps'])} laps, "
          f"{len(data['sot'])} STATIC_OT cycles, track ~= {data['track_len']} m")
    hot = [l["i"] for l in data["laps"] if l["nstop"] > 0]
    if hot:
        print(f"  laps with near-stops (start here): {hot}")


# The finalized HTML/CSS/JS with __DATA__ / __MAPDATA__ placeholders.
TEMPLATE = r'''<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>F1TENTH · Static-Avoidance Lap Inspector</title>
<style>
  :root{
    --bg:#080b11; --panel:#0f141d; --panel2:#131a25; --line:#1d2735; --line2:#263243;
    --txt:#c7d3e2; --mut:#69788d; --dim:#455467;
    --gb:#3fd0b0; --ot:#f0b24a; --tr:#e06fae; --lost:#ff4d4d; --rec:#6f9dff;
    --obs:#ff8a3d; --accent:#3fd0b0;
    --mono:"JetBrains Mono",ui-monospace,"SF Mono",Menlo,Consolas,monospace;
  }
  *{box-sizing:border-box}
  html,body{margin:0;background:var(--bg);color:var(--txt);font-family:var(--mono);-webkit-font-smoothing:antialiased}
  body{padding:18px 20px 40px;max-width:1400px;margin:0 auto}
  a{color:var(--accent)}
  .eyebrow{font-size:10px;letter-spacing:.32em;text-transform:uppercase;color:var(--dim)}
  h1{font-size:19px;font-weight:700;letter-spacing:.04em;margin:2px 0 0}
  h1 .amp{color:var(--accent)}
  header{display:flex;justify-content:space-between;align-items:flex-end;flex-wrap:wrap;gap:12px;
    border-bottom:1px solid var(--line);padding-bottom:14px;margin-bottom:16px}
  .meta{font-size:11px;color:var(--mut);text-align:right;line-height:1.7}
  .meta b{color:var(--txt);font-weight:600}

  .rail-wrap{margin-bottom:16px}
  .rail-hd{display:flex;justify-content:space-between;align-items:center;margin-bottom:7px}
  .rail{display:flex;gap:6px;overflow-x:auto;padding-bottom:8px;scrollbar-width:thin}
  .chip{flex:0 0 auto;background:var(--panel);border:1px solid var(--line);border-radius:7px;
    padding:7px 9px;cursor:pointer;min-width:62px;text-align:center;transition:.12s;position:relative}
  .chip:hover{border-color:var(--line2);background:var(--panel2)}
  .chip.on{border-color:var(--accent);background:#0c1a1a;box-shadow:0 0 0 1px var(--accent) inset}
  .chip .ln{font-size:14px;font-weight:700}
  .chip .sub{font-size:9px;color:var(--mut);margin-top:2px;letter-spacing:.02em}
  .chip.warn::after{content:"";position:absolute;top:5px;right:6px;width:6px;height:6px;border-radius:50%;background:var(--lost);box-shadow:0 0 6px var(--lost)}
  .chip.idle{opacity:.5}

  .grid{display:grid;grid-template-columns:1fr 330px;gap:14px}
  @media(max-width:940px){.grid{grid-template-columns:1fr}}
  .card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:13px 14px}
  .card h2{font-size:10px;letter-spacing:.2em;text-transform:uppercase;color:var(--mut);margin:0 0 10px;font-weight:600}
  .card h2 .r{float:right;color:var(--dim);letter-spacing:.02em;text-transform:none}
  canvas{display:block;width:100%;cursor:crosshair}

  .legend{display:flex;flex-wrap:wrap;gap:12px 16px;margin-top:12px;font-size:11px;color:var(--mut)}
  .legend .k{display:inline-flex;align-items:center;gap:6px}
  .sw{width:11px;height:11px;border-radius:3px;display:inline-block}
  .sw.line{width:16px;height:3px;border-radius:2px}

  .detail .row{display:flex;justify-content:space-between;padding:5px 0;border-bottom:1px solid var(--line);font-size:12px}
  .detail .row:last-child{border-bottom:0}
  .detail .k{color:var(--mut)} .detail .v{color:var(--txt);font-weight:600}
  .detail .v.big{font-size:15px}
  .gates{display:flex;flex-wrap:wrap;gap:5px;margin-top:6px}
  .pill{font-size:10px;padding:3px 7px;border-radius:5px;border:1px solid var(--line2);letter-spacing:.03em}
  .pill.y{background:#0d1f1a;border-color:#1f5c4a;color:var(--gb)}
  .pill.n{background:#22110f;border-color:#5c2320;color:var(--lost)}
  .pill.dec-ot{background:#241a08;border-color:#7a561a;color:var(--ot)}
  .pill.dec-tr{background:#231120;border-color:#6e2a55;color:var(--tr)}
  .sect{margin-top:14px;padding-top:12px;border-top:1px solid var(--line)}
  .sect .lbl{font-size:9px;letter-spacing:.22em;text-transform:uppercase;color:var(--dim);margin-bottom:8px}
  .hint{font-size:11px;color:var(--dim);line-height:1.6}
  .stat-grid{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:4px}
  .stat{background:var(--panel2);border:1px solid var(--line);border-radius:7px;padding:8px 9px}
  .stat .n{font-size:17px;font-weight:700} .stat .l{font-size:9px;color:var(--mut);letter-spacing:.05em;margin-top:2px}
  .stat.hot .n{color:var(--lost)}
  .foot{margin-top:20px;font-size:11px;color:var(--dim);line-height:1.7;border-top:1px solid var(--line);padding-top:12px}
</style>
</head>
<body>
<header>
  <div>
    <div class="eyebrow">F1TENTH · racing_stack telemetry</div>
    <h1>Static-Avoidance Lap Inspector <span class="amp">/</span> <span id="mapname"></span></h1>
  </div>
  <div class="meta" id="runmeta"></div>
</header>

<div class="rail-wrap">
  <div class="rail-hd">
    <span class="eyebrow">랩 선택 · 빨간 점 = near-stop 발생 랩</span>
    <span class="eyebrow" id="railnote"></span>
  </div>
  <div class="rail" id="rail"></div>
</div>

<div class="grid">
  <div>
    <div class="card">
      <h2>실제 맵 뷰 <span class="r"><label style="cursor:pointer;color:var(--mut)"><input type="checkbox" id="mapchk" checked style="vertical-align:-1px"> occupancy</label> · <span id="cartsub"></span></span></h2>
      <canvas id="cart" height="360"></canvas>
      <div class="legend">
        <span class="k"><span class="sw line" style="background:var(--mut)"></span>트랙 경계 (d_left/d_right)</span>
        <span class="k"><span class="sw line" style="background:var(--accent);opacity:.6"></span>raceline (IQP)</span>
        <span class="k"><span class="sw" style="background:var(--obs);border-radius:50%;opacity:.75"></span>장애물</span>
        <span class="k"><span class="sw" style="background:var(--lost);border-radius:50%"></span>near-stop</span>
        <span class="k" style="color:var(--dim)">주행경로 색 = 상태 (아래 범례)</span>
      </div>
    </div>
    <div class="card">
      <h2>Frenet 트랙 뷰 <span class="r" id="frenetsub"></span></h2>
      <canvas id="frenet" height="300"></canvas>
      <div class="legend">
        <span class="k"><span class="sw line" style="background:var(--gb)"></span>GB_TRACK</span>
        <span class="k"><span class="sw line" style="background:var(--ot)"></span>OVERTAKE</span>
        <span class="k"><span class="sw line" style="background:var(--tr)"></span>TRAILING</span>
        <span class="k"><span class="sw line" style="background:var(--lost)"></span>LOSTLINE</span>
        <span class="k"><span class="sw line" style="background:var(--rec)"></span>RECOVERY</span>
        <span class="k"><span class="sw" style="background:var(--obs);border-radius:50%;opacity:.75"></span>장애물 감지</span>
        <span class="k"><span class="sw" style="background:var(--lost);border-radius:50%"></span>near-stop (vs&lt;0.3)</span>
      </div>
    </div>
    <div class="card" style="margin-top:14px">
      <h2>속도 프로파일 <span class="r">m/s vs s</span></h2>
      <canvas id="speed" height="150"></canvas>
    </div>
    <div class="card" style="margin-top:14px">
      <h2>상태 타임라인 <span class="r">state vs s (트랙 정렬)</span></h2>
      <canvas id="ribbon" height="34"></canvas>
    </div>
  </div>

  <div>
    <div class="card detail">
      <h2>판단 상세 <span class="r">hover to inspect</span></h2>
      <div class="row"><span class="k">t / s / d</span><span class="v" id="d_pos">—</span></div>
      <div class="row"><span class="k">속도 vs</span><span class="v big" id="d_vs">—</span></div>
      <div class="row"><span class="k">상태</span><span class="v" id="d_st">—</span></div>
      <div class="row"><span class="k">장애물 n</span><span class="v" id="d_n">—</span></div>

      <div class="sect">
        <div class="lbl">state_machine · 추월 게이트</div>
        <div class="gates" id="d_gates"><span class="hint">이 지점엔 STATIC_OT 판단 로그가 없음</span></div>
        <div class="row" style="margin-top:8px"><span class="k">decision</span><span class="v" id="d_dec">—</span></div>
        <div class="row"><span class="k">raw_wpnts</span><span class="v" id="d_wp">—</span></div>
      </div>

      <div class="sect">
        <div class="lbl">static_avoidance 노드 · 여유</div>
        <div class="row"><span class="k">obstacle</span><span class="v" id="d_oid">—</span></div>
        <div class="row"><span class="k">회피 side / rung</span><span class="v" id="d_side">—</span></div>
        <div class="row"><span class="k">left clearance</span><span class="v" id="d_lc">—</span></div>
        <div class="row"><span class="k">right clearance</span><span class="v" id="d_rc">—</span></div>
        <div class="row"><span class="k">candidate</span><span class="v" id="d_valid">—</span></div>
      </div>
    </div>

    <div class="card" style="margin-top:14px">
      <h2>랩 요약 <span class="r" id="lapsub"></span></h2>
      <div class="stat-grid" id="lapstats"></div>
      <div class="hint" style="margin-top:10px" id="lapread"></div>
    </div>
  </div>
</div>

<div class="foot" id="foot"></div>

<script>
const DATA = __DATA__;
const MAPDATA = __MAPDATA__;
const SC = {GB_TRACK:'--gb',OVERTAKE:'--ot',TRAILING:'--tr',LOSTLINE:'--lost',RECOVERY:'--rec'};
const css = k => getComputedStyle(document.documentElement).getPropertyValue(k).trim();
const scol = s => css(SC[s]||'--mut');
const TRACK = DATA.track_len;
document.getElementById('mapname').textContent = "map 0815test3";
document.getElementById('runmeta').innerHTML =
  `run <b>20260816_120652</b> · 약 <b>1187 s</b> · <b>47</b> laps<br>ego <b>${DATA.ego.length}</b> samples · state changes <b>${DATA.changes.length}</b> · track ≈ <b>${TRACK} m</b>`;

// ---- lap rail ----
const rail = document.getElementById('rail');
let sel = 7; // default to an interesting lap
DATA.laps.forEach(l=>{
  const c=document.createElement('div');
  c.className='chip'+(l.nstop>0?' warn':'')+(l.moving?'':' idle');
  c.innerHTML=`<div class="ln">L${l.i}</div><div class="sub">${l.dur}s · obs${l.nobs}${l.nstop?' · ⚠'+l.nstop:''}</div>`;
  c.onclick=()=>{sel=l.i;draw();};
  c.dataset.i=l.i;
  rail.appendChild(c);
});
document.getElementById('railnote').textContent = `near-stop 랩: 7, 8, 9`;

function lapMeta(){return DATA.laps.find(l=>l.i===sel);}
function lapSamples(){const m=lapMeta();return DATA.ego.filter(e=>e.t>=m.t0&&e.t<=m.t1);}

// ---- canvas helpers ----
function fit(cv){const dpr=devicePixelRatio||1;
  const cssH = cv._h || (cv._h = parseInt(cv.getAttribute('height'))||150);
  cv.style.height=cssH+'px';
  const r=cv.getBoundingClientRect();
  cv.width=Math.max(1,Math.round(r.width*dpr));cv.height=Math.round(cssH*dpr);
  const x=cv.getContext('2d');x.setTransform(dpr,0,0,dpr,0,0);
  return {x,w:r.width,h:cssH};}

let hoverIdx=-1, curSamples=[];


// ===== CARTESIAN (real map) view =====
const RL=MAPDATA.rl, RLs=RL.map(r=>r[0]), TL=MAPDATA.track_s;
const mapImg=new Image(); let mapReady=false;
mapImg.onload=()=>{mapReady=true;drawCartesian();};
mapImg.src='data:image/png;base64,'+MAPDATA.map.b64;
document.getElementById('mapchk').addEventListener('change',drawCartesian);
// view bbox from boundaries (+pad)
let _bx=[...MAPDATA.left.map(p=>p[0]),...MAPDATA.right.map(p=>p[0])];
let _by=[...MAPDATA.left.map(p=>p[1]),...MAPDATA.right.map(p=>p[1])];
const PAD=0.6;
const VX0=Math.min(..._bx)-PAD,VX1=Math.max(..._bx)+PAD,VY0=Math.min(..._by)-PAD,VY1=Math.max(..._by)+PAD;
const WASP=(VX1-VX0)/(VY1-VY0);
function binS(s){let lo=0,hi=RLs.length-1;if(s<=RLs[0])return 0;if(s>=RLs[hi])return hi;
  while(lo<hi){let m=(lo+hi+1)>>1; if(RLs[m]<=s)lo=m;else hi=m-1;} return lo;}
function f2w(s,d){s=((s%TL)+TL)%TL;let i=binS(s);let a=RL[i],b=RL[(i+1)%RL.length];
  let ds=b[0]-a[0];if(ds<=0)ds+=TL;let f=ds>1e-6?(s-a[0])/ds:0;
  let x=a[1]+(b[1]-a[1])*f,y=a[2]+(b[2]-a[2])*f;
  let ca=Math.cos(a[3]),sa=Math.sin(a[3]),cb=Math.cos(b[3]),sb=Math.sin(b[3]);
  let psi=Math.atan2(sa+(sb-sa)*f,ca+(cb-ca)*f);
  return [x-d*Math.sin(psi),y+d*Math.cos(psi)];}
let _cartMap=null;
function fitCart(cv){const dpr=devicePixelRatio||1;const r=cv.getBoundingClientRect();
  const w=r.width,hh=Math.round(w/WASP);cv.style.height=hh+'px';
  cv.width=Math.max(1,Math.round(w*dpr));cv.height=Math.round(hh*dpr);
  const x=cv.getContext('2d');x.setTransform(dpr,0,0,dpr,0,0);return {x,w,h:hh};}
function drawCartesian(){
  const cv=document.getElementById('cart');const {x,w,h}=fitCart(cv);
  const S=curSamples;
  const sc=Math.min(w/(VX1-VX0),h/(VY1-VY0));
  const offx=(w-(VX1-VX0)*sc)/2, offy=(h-(VY1-VY0)*sc)/2;
  const WX=wx=>offx+(VX1-wx)*sc, WY=wy=>h-offy-(VY1-wy)*sc;
  x.clearRect(0,0,w,h);
  // occupancy underlay (rotated 180° to match mirrored axes)
  if(mapReady && document.getElementById('mapchk').checked){
    const e=MAPDATA.map.extent; // [xmin,ymin,xmax,ymax]
    const left=WX(e[2]), top=WY(e[1]), iw=(e[2]-e[0])*sc, ih=(e[3]-e[1])*sc;
    x.globalAlpha=0.5; x.imageSmoothingEnabled=false;
    x.save(); x.translate(left+iw/2, top+ih/2); x.rotate(Math.PI);
    x.drawImage(mapImg, -iw/2,-ih/2, iw, ih); x.restore(); x.globalAlpha=1;
  }
  // track corridor fill (between left & right)
  x.beginPath();
  MAPDATA.left.forEach((p,i)=>{const cx=WX(p[0]),cy=WY(p[1]);i?x.lineTo(cx,cy):x.moveTo(cx,cy);});
  for(let i=MAPDATA.right.length-1;i>=0;i--){const p=MAPDATA.right[i];x.lineTo(WX(p[0]),WY(p[1]));}
  x.closePath(); x.fillStyle=css('--accent'); x.globalAlpha=0.05; x.fill(); x.globalAlpha=1;
  // boundaries
  x.strokeStyle=css('--line2'); x.lineWidth=1.4;
  [MAPDATA.left,MAPDATA.right].forEach(bd=>{x.beginPath();
    bd.forEach((p,i)=>{const cx=WX(p[0]),cy=WY(p[1]);i?x.lineTo(cx,cy):x.moveTo(cx,cy);});
    x.closePath();x.stroke();});
  // raceline
  x.strokeStyle=css('--accent');x.globalAlpha=.5;x.lineWidth=1;x.setLineDash([4,4]);x.beginPath();
  RL.forEach((r,i)=>{const cx=WX(r[1]),cy=WY(r[2]);i?x.lineTo(cx,cy):x.moveTo(cx,cy);});
  x.closePath();x.stroke();x.setLineDash([]);x.globalAlpha=1;
  // start/finish
  const s0=f2w(0,0);x.fillStyle=css('--txt');x.font='9px var(--mono)';x.fillText('S/F',WX(s0[0])+4,WY(s0[1])-4);
  x.beginPath();x.arc(WX(s0[0]),WY(s0[1]),3,0,7);x.fill();
  if(!S||!S.length){return;}
  // obstacles
  S.forEach(e=>e.obs.forEach(o=>{if(o.gap==null||o.d==null)return;const p=f2w(e.s+o.gap,o.d);
    const r=Math.max(2.5,(o.size||.35)/2*sc);
    x.fillStyle=css('--obs');x.globalAlpha=.14;x.beginPath();x.arc(WX(p[0]),WY(p[1]),r,0,7);x.fill();x.globalAlpha=1;}));
  // ego path by state
  x.lineWidth=2.4;let prev=null;
  S.forEach((e,i)=>{const p=f2w(e.s,e.d);const cx=WX(p[0]),cy=WY(p[1]);
    if(prev && Math.abs(e.s-prev.s)<=TL*0.5){x.strokeStyle=scol(e.st);x.beginPath();x.moveTo(WX(prev.w[0]),WY(prev.w[1]));x.lineTo(cx,cy);x.stroke();}
    prev={s:e.s,w:p};});
  // decision + near-stop markers
  S.forEach(e=>{const p=f2w(e.s,e.d);
    if(e.n>0&&e.soi>=0){const dec=DATA.sot[e.soi].decision;x.fillStyle=dec&&dec.startsWith('OVER')?css('--ot'):css('--tr');
      x.beginPath();x.arc(WX(p[0]),WY(p[1]),2.2,0,7);x.fill();}
    if(e.vs<0.3&&e.n>0){x.fillStyle=css('--lost');x.beginPath();const px=WX(p[0]),py=WY(p[1]);
      x.moveTo(px,py-5);x.lineTo(px+5,py);x.lineTo(px,py+5);x.lineTo(px-5,py);x.closePath();x.fill();}});
  // hover highlight
  if(hoverIdx>=0&&hoverIdx<S.length){const e=S[hoverIdx];const p=f2w(e.s,e.d);
    x.strokeStyle=css('--txt');x.globalAlpha=.5;x.lineWidth=1;x.beginPath();x.arc(WX(p[0]),WY(p[1]),9,0,7);x.stroke();x.globalAlpha=1;
    x.fillStyle=css('--accent');x.beginPath();x.arc(WX(p[0]),WY(p[1]),4,0,7);x.fill();x.strokeStyle=css('--bg');x.lineWidth=1.5;x.stroke();}
  _cartMap={WX,WY,sc};document.getElementById('cartsub').textContent='L'+lapMeta().i;
}
// cartesian hover: nearest ego sample by world distance
document.getElementById('cart').addEventListener('mousemove',ev=>{
  const S=curSamples;if(!S.length||!_cartMap)return;const cv=ev.currentTarget;const r=cv.getBoundingClientRect();
  const mx=ev.clientX-r.left,my=ev.clientY-r.top;let best=0,bd=1e9;
  S.forEach((e,i)=>{const p=f2w(e.s,e.d);const dx=_cartMap.WX(p[0])-mx,dy=_cartMap.WY(p[1])-my;const dd=dx*dx+dy*dy;if(dd<bd){bd=dd;best=i;}});
  hoverIdx=best;drawCartesian();drawFrenet();drawSpeed();drawRibbon();updateDetail();});
document.getElementById('cart').addEventListener('mouseleave',()=>{hoverIdx=-1;drawCartesian();drawFrenet();drawSpeed();drawRibbon();});

function drawFrenet(){
  const cv=document.getElementById('frenet');const {x,w,h}=fit(cv);
  const S=curSamples; if(!S.length)return;
  const PL=44,PR=14,PT=14,PB=26;
  // domains
  let sMin=Math.min(...S.map(e=>e.s)), sMax=Math.max(...S.map(e=>e.s));
  // include obstacle s
  const obsPts=[];
  S.forEach(e=>e.obs.forEach(o=>{ if(o.gap!=null){let os=e.s+o.gap; if(os>TRACK)os-=TRACK; obsPts.push({s:os,d:o.d,size:o.size,t:e.t});}}));
  let dv=[...S.map(e=>e.d), ...obsPts.map(o=>o.d)];
  let dMax=Math.max(0.6, Math.min(2.5, Math.max(...dv.map(Math.abs))*1.15));
  const X=s=>PL+(s-sMin)/((sMax-sMin)||1)*(w-PL-PR);
  const Y=d=>PT+(dMax-d)/(2*dMax)*(h-PT-PB);
  x.clearRect(0,0,w,h);
  // grid
  x.strokeStyle=css('--line');x.fillStyle=css('--dim');x.font='9px var(--mono)';x.lineWidth=1;
  for(let dd=-Math.floor(dMax/0.25)*0.25; dd<=dMax; dd+=0.25){
    x.globalAlpha=Math.abs(dd)<1e-6?1:.4; x.beginPath();x.moveTo(PL,Y(dd));x.lineTo(w-PR,Y(dd));
    x.strokeStyle=Math.abs(dd)<1e-6?css('--line2'):css('--line'); x.stroke();
    x.globalAlpha=1; x.fillText(dd.toFixed(2),4,Y(dd)+3);
  }
  // raceline label
  x.fillStyle=css('--mut');x.fillText('raceline d=0',w-PR-84,Y(0)-4);
  // s ticks
  for(let ss=Math.ceil(sMin/5)*5; ss<=sMax; ss+=5){x.fillStyle=css('--dim');x.fillText(ss+'m',X(ss)-6,h-8);
    x.strokeStyle=css('--line');x.globalAlpha=.4;x.beginPath();x.moveTo(X(ss),PT);x.lineTo(X(ss),h-PB);x.stroke();x.globalAlpha=1;}
  // obstacles (translucent, cluster = real)
  obsPts.forEach(o=>{const r=Math.max(3,(o.size||0.35)/2/(2*dMax)*(h-PT-PB));
    x.fillStyle=css('--obs');x.globalAlpha=.16;x.beginPath();x.arc(X(o.s),Y(o.d),r,0,7);x.fill();x.globalAlpha=1;});
  // ego trace colored by state
  x.lineWidth=2.2;
  for(let i=1;i<S.length;i++){
    if(Math.abs(S[i].s-S[i-1].s)>TRACK*0.5)continue; // wrap guard
    x.strokeStyle=scol(S[i].st);x.beginPath();x.moveTo(X(S[i-1].s),Y(S[i-1].d));x.lineTo(X(S[i].s),Y(S[i].d));x.stroke();
  }
  // decision + near-stop markers
  S.forEach((e,i)=>{
    if(e.n>0 && e.soi>=0){const dec=DATA.sot[e.soi].decision;
      x.fillStyle=dec&&dec.startsWith('OVER')?css('--ot'):css('--tr');
      x.beginPath();const px=X(e.s),py=Y(e.d)-9;x.moveTo(px,py-4);x.lineTo(px-3.5,py+2);x.lineTo(px+3.5,py+2);x.closePath();x.fill();}
    if(e.vs<0.3 && e.n>0){x.fillStyle=css('--lost');x.beginPath();const px=X(e.s),py=Y(e.d);
      x.moveTo(px,py-5);x.lineTo(px+5,py);x.lineTo(px,py+5);x.lineTo(px-5,py);x.closePath();x.fill();}
  });
  // hover crosshair
  if(hoverIdx>=0&&hoverIdx<S.length){const e=S[hoverIdx];
    x.strokeStyle=css('--txt');x.globalAlpha=.35;x.setLineDash([3,3]);
    x.beginPath();x.moveTo(X(e.s),PT);x.lineTo(X(e.s),h-PB);x.stroke();x.setLineDash([]);x.globalAlpha=1;
    x.fillStyle=css('--accent');x.beginPath();x.arc(X(e.s),Y(e.d),4,0,7);x.fill();
    x.strokeStyle=css('--bg');x.lineWidth=1.5;x.stroke();}
  cv._map={X,Y,S,sMin,sMax};
}

function drawSpeed(){
  const cv=document.getElementById('speed');const {x,w,h}=fit(cv);
  const S=curSamples;if(!S.length)return;
  const PL=44,PR=14,PT=10,PB=22;
  let sMin=Math.min(...S.map(e=>e.s)),sMax=Math.max(...S.map(e=>e.s));
  let vMax=Math.max(1,Math.max(...S.map(e=>e.vs))*1.1);
  const X=s=>PL+(s-sMin)/((sMax-sMin)||1)*(w-PL-PR);
  const Y=v=>PT+(vMax-v)/vMax*(h-PT-PB);
  x.clearRect(0,0,w,h);
  x.strokeStyle=css('--line');x.fillStyle=css('--dim');x.font='9px var(--mono)';
  for(let v=0;v<=vMax;v+=1){x.globalAlpha=.4;x.beginPath();x.moveTo(PL,Y(v));x.lineTo(w-PR,Y(v));x.stroke();x.globalAlpha=1;x.fillText(v.toFixed(0),6,Y(v)+3);}
  // near-stop band
  x.fillStyle=css('--lost');x.globalAlpha=.08;x.fillRect(PL,Y(0.3),w-PL-PR,h-PB-Y(0.3));x.globalAlpha=1;
  x.strokeStyle=css('--accent');x.lineWidth=1.8;x.beginPath();let started=false;
  S.forEach((e,i)=>{if(i>0&&Math.abs(S[i].s-S[i-1].s)>TRACK*0.5){started=false;} const px=X(e.s),py=Y(e.vs);
    if(!started){x.moveTo(px,py);started=true;}else x.lineTo(px,py);});
  x.stroke();
  S.forEach(e=>{if(e.vs<0.3&&e.n>0){x.fillStyle=css('--lost');x.beginPath();x.arc(X(e.s),Y(e.vs),3,0,7);x.fill();}});
  if(hoverIdx>=0){const e=S[hoverIdx];x.strokeStyle=css('--txt');x.globalAlpha=.35;x.setLineDash([3,3]);
    x.beginPath();x.moveTo(X(e.s),PT);x.lineTo(X(e.s),h-PB);x.stroke();x.setLineDash([]);x.globalAlpha=1;
    x.fillStyle=css('--accent');x.beginPath();x.arc(X(e.s),Y(e.vs),3.5,0,7);x.fill();}
}

function drawRibbon(){
  const cv=document.getElementById('ribbon');const {x,w,h}=fit(cv);
  const S=curSamples;if(!S.length)return;
  const PL=44,PR=14;
  let sMin=Math.min(...S.map(e=>e.s)),sMax=Math.max(...S.map(e=>e.s));
  const X=s=>PL+(s-sMin)/((sMax-sMin)||1)*(w-PL-PR);
  x.clearRect(0,0,w,h);
  for(let i=0;i<S.length-1;i++){
    if(Math.abs(S[i+1].s-S[i].s)>TRACK*0.5)continue;
    x.fillStyle=scol(S[i].st);const x0=X(S[i].s),x1=X(S[i+1].s);
    x.fillRect(x0,4,Math.max(1,x1-x0)+.5,h-8);
  }
  if(hoverIdx>=0){const e=S[hoverIdx];x.strokeStyle=css('--txt');x.beginPath();x.moveTo(X(e.s),0);x.lineTo(X(e.s),h);x.stroke();}
}

function setPill(id,label,ok){const el=document.getElementById(id);}
function updateDetail(){
  const S=curSamples;const set=(id,v)=>document.getElementById(id).textContent=v;
  if(hoverIdx<0||hoverIdx>=S.length){return;}
  const e=S[hoverIdx];
  set('d_pos',`${(e.t-S[0].t).toFixed(2)}s · ${e.s.toFixed(2)} · ${e.d>=0?'+':''}${e.d.toFixed(2)}`);
  const vsEl=document.getElementById('d_vs');vsEl.textContent=e.vs.toFixed(2)+' m/s';
  vsEl.style.color=e.vs<0.3&&e.n>0?css('--lost'):css('--txt');
  const stEl=document.getElementById('d_st');stEl.textContent=e.st;stEl.style.color=scol(e.st);
  set('d_n',e.n+(e.n>0?' 개':''));
  // gates
  const g=document.getElementById('d_gates');
  if(e.soi>=0){const o=DATA.sot[e.soi];
    const gk=[['exists',o.path_exists],['ttl',o.path_ttl_ok],['on_spline',o.on_spline],['safe',o.path_safe],['locked',o.path_locked]];
    g.innerHTML=gk.map(([k,v])=>`<span class="pill ${v?'y':'n'}">${k} ${v?'✓':'✗'}</span>`).join('');
    const dec=o.decision||'—';const dc=dec.startsWith('OVER')?'dec-ot':'dec-tr';
    document.getElementById('d_dec').innerHTML=`<span class="pill ${dc}">${dec}</span>`;
    set('d_wp',o.raw_wpnts);
  }else{g.innerHTML='<span class="hint">이 지점엔 STATIC_OT 로그 없음 (장애물 미검출 구간)</span>';
    document.getElementById('d_dec').textContent='—';set('d_wp','—');}
  // static node
  if(e.sai>=0){const a=DATA.san[e.sai];
    set('d_oid',a.obs_id==='-'?'—':('id '+a.obs_id)+(a.obs_dist!=null?` · ${a.obs_dist.toFixed(2)}m`:''));
    set('d_side',(a.side&&a.side!=='-'?a.side:'—')+' / rung '+(a.rung&&a.rung!=='-'?a.rung:'—'));
    const lcEl=document.getElementById('d_lc'),rcEl=document.getElementById('d_rc');
    lcEl.textContent=a.lc!=null?a.lc.toFixed(2)+' m':'—'; rcEl.textContent=a.rc!=null?a.rc.toFixed(2)+' m':'—';
    lcEl.style.color=a.lc!=null&&a.lc<0.05?css('--lost'):css('--txt');
    rcEl.style.color=a.rc!=null&&a.rc<0.05?css('--lost'):css('--txt');
    const vEl=document.getElementById('d_valid');vEl.textContent=a.valid?'valid ✓':'invalid ✗';vEl.style.color=a.valid?css('--gb'):css('--lost');
  }else{['d_oid','d_side','d_lc','d_rc','d_valid'].forEach(i=>set(i,'—'));}
}

function drawStats(){
  const m=lapMeta();const S=curSamples;
  document.getElementById('lapsub').textContent=`L${m.i} · ${m.dur}s`;
  document.getElementById('frenetsub').textContent=`L${m.i} · s ${Math.min(...S.map(e=>e.s)).toFixed(1)}–${Math.max(...S.map(e=>e.s)).toFixed(1)}m`;
  const online=S.filter(e=>Math.abs(e.d)<0.05).length/S.length*100;
  const cells=[['obs 샘플',m.nobs,false],['near-stop',m.nstop,m.nstop>0],
    ['OVERTAKE 샘플',m.ov,false],['LOSTLINE 샘플',m.lost,m.lost>3],
    ['라인안착 <5cm',online.toFixed(0)+'%',false],['최저속도',Math.min(...S.map(e=>e.vs)).toFixed(2),false]];
  document.getElementById('lapstats').innerHTML=cells.map(([l,n,hot])=>
    `<div class="stat${hot?' hot':''}"><div class="n">${n}</div><div class="l">${l}</div></div>`).join('');
  // narrative read
  let msg;
  if(m.nstop>0) msg=`이 랩엔 near-stop ${m.nstop}회. 장애물 앞에서 추월 커밋이 취소되고 속도가 붕괴한 지점이야. 삼각형(핑크=trailing) 마커가 붙은 곳을 hover 하면 path_safe ✗ 를 확인할 수 있어.`;
  else if(m.lost>3) msg=`LOSTLINE 샘플이 ${m.lost}개 — 로컬라이제이션이 흔들린 랩. d 값이 크게 튀는 구간을 보면 돼.`;
  else if(m.nobs>0) msg=`장애물은 있었지만 near-stop 없이 통과한 랩. 정상적으로 raceline을 따라간 케이스와 비교해봐.`;
  else msg=`장애물 없이 깨끗하게 돈 랩.`;
  document.getElementById('lapread').textContent=msg;
}

function draw(){
  document.querySelectorAll('.chip').forEach(c=>c.classList.toggle('on',+c.dataset.i===sel));
  curSamples=lapSamples();hoverIdx=-1;
  drawCartesian();drawFrenet();drawSpeed();drawRibbon();drawStats();updateDetail();
}

// hover: map mouse x -> nearest sample by s on frenet, by s on speed, by t on ribbon
function bindHover(cv,mode){
  cv.addEventListener('mousemove',ev=>{
    const S=curSamples;if(!S.length)return;const r=cv.getBoundingClientRect();const mx=ev.clientX-r.left;
    let best=0,bd=1e9;
    {const map=document.getElementById('frenet')._map;if(!map)return;
      const sMin=map.sMin,sMax=map.sMax;const PL=44,PR=14;
      const frac=(mx-PL)/(r.width-PL-PR);const ss=sMin+frac*(sMax-sMin);
      S.forEach((e,i)=>{const dd=Math.abs(e.s-ss);if(dd<bd){bd=dd;best=i;}});}
    hoverIdx=best;drawCartesian();drawFrenet();drawSpeed();drawRibbon();updateDetail();
  });
  cv.addEventListener('mouseleave',()=>{hoverIdx=-1;drawCartesian();drawFrenet();drawSpeed();drawRibbon();});
}
bindHover(document.getElementById('frenet'),'frenet');
bindHover(document.getElementById('speed'),'speed');
bindHover(document.getElementById('ribbon'),'ribbon');

document.getElementById('foot').innerHTML =
  '읽는 법 — 가로축은 트랙 진행거리 s(트랙을 직선으로 펼친 것), 세로축은 raceline 기준 횡방향 오차 d. '+
  '주황 점이 뭉치면 실제 장애물 위치, 흩어지면 perception id churn. 경로 위 삼각형은 그 순간 state_machine의 추월 판단(주황=OVERTAKE 커밋, 핑크=TRAILING). '+
  '빨간 다이아몬드는 near-stop. 어느 점이든 hover 하면 우측에 그때의 추월 게이트(path_safe 등)와 static 노드의 좌/우 여유가 나와 — 여유가 음수인데 커밋을 취소한 지점이 문제의 핵심이야.';

window.addEventListener('resize',()=>{clearTimeout(window._rz);window._rz=setTimeout(draw,120);});
draw();
</script>
</body>
</html>'''


if __name__ == "__main__":
    main()
