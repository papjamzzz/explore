import os, time, socket, json
from pathlib import Path
from flask import Flask, request, jsonify, render_template_string
from dotenv import load_dotenv
import anthropic

load_dotenv()
app = Flask(__name__)
PORT = int(os.getenv("PORT", 5572))

ABLETON_HOST = "127.0.0.1"
ABLETON_PORT = 9877

# ── Ableton socket ─────────────────────────────────────────────────────────────

def ableton_send(command_type, params=None):
    payload = json.dumps({"type": command_type, "params": params or {}}) + "\n"
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(5)
            s.connect((ABLETON_HOST, ABLETON_PORT))
            s.sendall(payload.encode())
            time.sleep(0.02)
            chunks = []
            s.settimeout(2)
            while True:
                try:
                    chunk = s.recv(4096)
                    if not chunk:
                        break
                    chunks.append(chunk)
                except socket.timeout:
                    break
            raw = b"".join(chunks).decode()
            return json.loads(raw) if raw.strip() else {"error": "empty response"}
    except Exception as e:
        return {"error": str(e)}

def get_all_tracks():
    session = ableton_send("get_session_info")
    count = session.get("track_count", 0) if isinstance(session, dict) else 0
    tracks = []
    for i in range(min(count, 24)):
        t = ableton_send("get_track_info", {"track_index": i})
        if isinstance(t, dict) and "error" not in t:
            tracks.append(t)
        time.sleep(0.02)
    return session, tracks

# ── Audio analysis ─────────────────────────────────────────────────────────────

try:
    import librosa
    import numpy as np
    import pyloudnorm as pyln
    import soundfile as sf
    AUDIO_OK = True
except ImportError:
    AUDIO_OK = False
    np = None

def analyze_audio_file(path):
    if not AUDIO_OK:
        return {"error": "librosa not installed"}
    try:
        y, sr = librosa.load(str(path), sr=None, mono=False)
        if y.ndim == 2:
            stereo_width = float(
                np.mean(np.abs(y[0] - y[1])) / (np.mean(np.abs(y[0] + y[1])) + 1e-9)
            )
            y_mono = librosa.to_mono(y)
        else:
            stereo_width = 0.0
            y_mono = y
        peak_db  = float(20 * np.log10(np.max(np.abs(y_mono)) + 1e-9))
        rms_db   = float(20 * np.log10(np.sqrt(np.mean(y_mono**2)) + 1e-9))
        centroid = float(np.mean(librosa.feature.spectral_centroid(y=y_mono, sr=sr)))
        try:
            meter = pyln.Meter(sr)
            data, _ = sf.read(str(path))
            if data.ndim == 1:
                data = data[:, None]
            lufs = round(meter.integrated_loudness(data), 1)
        except Exception:
            lufs = None
        fft  = np.abs(np.fft.rfft(y_mono))
        freq = np.fft.rfftfreq(len(y_mono), d=1/sr)
        def band_energy(lo, hi):
            mask = (freq >= lo) & (freq < hi)
            total = np.sum(fft**2) + 1e-9
            return float(np.sum(fft[mask]**2) / total)
        return {
            "peak_db":              round(peak_db, 1),
            "rms_db":               round(rms_db, 1),
            "lufs":                 lufs,
            "dynamic_range_db":     round(peak_db - rms_db, 1),
            "stereo_width":         round(stereo_width, 3),
            "spectral_centroid_hz": round(centroid),
            "sub_energy":           round(band_energy(20,    80),    3),
            "bass_energy":          round(band_energy(80,    250),   3),
            "low_mid_energy":       round(band_energy(250,   500),   3),
            "mid_energy":           round(band_energy(500,   2000),  3),
            "high_mid_energy":      round(band_energy(2000,  6000),  3),
            "air_energy":           round(band_energy(6000,  20000), 3),
        }
    except Exception as e:
        return {"error": str(e)}

def find_project_audio(project_path):
    p = Path(project_path)
    files = []
    for ext in ("wav", "aif", "aiff", "mp3", "flac"):
        files.extend(p.rglob(f"*.{ext}"))
    return files[:24]

# ── Health scoring ─────────────────────────────────────────────────────────────

def compute_track_health(track, audio_data):
    name     = track.get("name", "")
    adat     = audio_data.get(name, {})
    is_audio = not track.get("is_midi_track", True)
    devices  = track.get("devices", [])
    if not adat or "peak_db" not in adat:
        return 72 if devices else 62
    score = 100
    peak    = adat.get("peak_db", -99)
    rms     = adat.get("rms_db", -99)
    lufs    = adat.get("lufs")
    width   = adat.get("stereo_width", 0)
    low_mid = adat.get("low_mid_energy", 0)
    sub     = adat.get("sub_energy", 0)
    bass    = adat.get("bass_energy", 0)
    if peak > -0.5:    score -= 35
    elif peak > -1.0:  score -= 20
    elif peak > -3.0:  score -= 8
    if lufs is not None:
        if lufs > -6:    score -= 20
        elif lufs > -9:  score -= 10
    if rms > -6:    score -= 10
    elif rms < -35: score -= 5
    if is_audio:
        if width < 0.02:   score -= 12
        elif width < 0.05: score -= 5
    if low_mid > 0.30:   score -= 12
    elif low_mid > 0.22: score -= 5
    if sub + bass > 0.55:   score -= 10
    elif sub + bass > 0.45: score -= 5
    if sub > 0.30: score -= 8
    return max(0, min(100, score))

def compute_overall_health(tracks, audio_data, problems):
    if not tracks:
        return 50
    scores = [compute_track_health(t, audio_data) for t in tracks]
    base   = sum(scores) / len(scores)
    for p in problems:
        sev = p.get("severity", "low")
        if sev == "high":    base -= 12
        elif sev == "medium": base -= 6
        elif sev == "low":   base -= 3
    return max(0, min(100, round(base)))

# ── Problem detection ──────────────────────────────────────────────────────────

def detect_problems(tracks, audio_data):
    problems = []
    muddy = [t["name"] for t in tracks
             if audio_data.get(t.get("name",""), {}).get("low_mid_energy", 0) > 0.25]
    if len(muddy) >= 2:
        problems.append({"severity": "high", "title": "Frequency Mud",
            "detail": f"{len(muddy)} tracks ({', '.join(muddy[:3])}) heavy in 250–500 Hz.",
            "fix": "High-pass or shelf cut ~200–350 Hz on non-bass elements."})
    low_heavy = [t["name"] for t in tracks
                 if (audio_data.get(t.get("name",""), {}).get("sub_energy", 0)
                     + audio_data.get(t.get("name",""), {}).get("bass_energy", 0)) > 0.45]
    if len(low_heavy) >= 2:
        problems.append({"severity": "high", "title": "Low-End Conflict",
            "detail": f"{len(low_heavy)} tracks competing in sub/bass region.",
            "fix": "Side-chain or high-pass everything except kick and bass below 80 Hz."})
    clipping = [t["name"] for t in tracks
                if audio_data.get(t.get("name",""), {}).get("peak_db", -99) > -1]
    if clipping:
        problems.append({"severity": "high", "title": "Clipping",
            "detail": f"Near or above 0 dBFS: {', '.join(clipping)}.",
            "fix": "Lower gain or add a limiter on these tracks."})
    narrow = [t["name"] for t in tracks
              if not t.get("is_midi_track")
              and audio_data.get(t.get("name",""), {}).get("stereo_width", 1) < 0.05]
    if len(narrow) >= 3:
        problems.append({"severity": "medium", "title": "Narrow Mix",
            "detail": f"{len(narrow)} audio tracks appear mostly mono.",
            "fix": "Add subtle stereo widening on pads or ambience."})
    if not problems:
        problems.append({"severity": "none", "title": "Clean Scan",
            "detail": "No critical issues detected.", "fix": ""})
    return problems

# ── Claude ─────────────────────────────────────────────────────────────────────

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

def ask_claude(system, user):
    try:
        msg = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=1024,
            system=system,
            messages=[{"role": "user", "content": user}]
        )
        text   = msg.content[0].text if msg.content else ""
        tokens = msg.usage.output_tokens if msg.usage else 0
        return text, tokens
    except Exception as e:
        return f"Error: {e}", 0

# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route("/api/session")
def api_session():
    session, tracks = get_all_tracks()
    return jsonify({"session": session, "tracks": tracks})

@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    d = request.get_json() or {}
    project_path = d.get("project_path", "")
    session, tracks = get_all_tracks()
    audio_data = {}
    if project_path and Path(project_path).exists():
        for f in find_project_audio(project_path):
            result = analyze_audio_file(f)
            if "error" not in result:
                audio_data[f.stem] = result
    problems       = detect_problems(tracks, audio_data)
    track_scores   = {t.get("name",""): compute_track_health(t, audio_data) for t in tracks}
    overall_health = compute_overall_health(tracks, audio_data, problems)
    track_lines = []
    for t in tracks:
        kind  = "MIDI" if t.get("is_midi_track") else "Audio"
        devs  = ", ".join(x["name"] for x in t.get("devices", []) if x.get("name"))
        adat  = audio_data.get(t.get("name",""), {})
        aline = ""
        if adat and "peak_db" in adat:
            aline = (f" [peak:{adat['peak_db']}dB rms:{adat['rms_db']}dB"
                     f" lufs:{adat.get('lufs','?')} width:{adat.get('stereo_width','?')}"
                     f" sub:{adat.get('sub_energy','?')} bass:{adat.get('bass_energy','?')}"
                     f" low_mid:{adat.get('low_mid_energy','?')} mid:{adat.get('mid_energy','?')}]")
        track_lines.append(f"  [{kind}] {t.get('name','?')}{': ' + devs if devs else ''}{aline}")
    session_ctx = (
        f"SESSION: {session.get('tempo','?')} BPM, "
        f"{session.get('signature_numerator',4)}/{session.get('signature_denominator',4)}, "
        f"{session.get('track_count',0)} tracks\n" + "\n".join(track_lines)
    )
    problem_ctx = "\n".join(
        f"- [{p['severity'].upper()}] {p['title']}: {p['detail']}" for p in problems
    )
    return jsonify({
        "session": session, "tracks": tracks,
        "audio_data": audio_data, "problems": problems,
        "session_ctx": session_ctx, "problem_ctx": problem_ctx,
        "track_scores": track_scores, "overall_health": overall_health,
    })

@app.route("/api/gain")
def api_gain():
    state_path = Path.home() / ".streamfader" / "state.json"
    try:
        return jsonify(json.loads(state_path.read_text()))
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route("/api/devices/<int:track_index>")
def api_devices(track_index):
    return jsonify(ableton_send("get_device_parameters", {"track_index": track_index}))

@app.route("/api/plugins")
def api_plugins():
    return jsonify(ableton_send("list_external_plugins"))

@app.route("/api/arrangement")
def api_arrangement():
    return jsonify(ableton_send("get_arrangement_info"))

@app.route("/api/cues")
def api_cues():
    return jsonify(ableton_send("get_cue_points"))

@app.route("/api/ask", methods=["POST"])
def api_ask():
    d = request.get_json() or {}
    system = """You are Explore — an expert mix engineer and producer mentor inside Ableton Live.

You have live session data: track names, devices, and audio analysis metrics.

Your job:
- Diagnose mix problems in plain language
- Explain WHY problems exist, not just that they exist
- Give specific, actionable fixes with exact parameters when possible
- Sound like an engineer sitting next to the producer — honest, direct, no fluff
- Never mix automatically. Explain first. Let the producer decide.

Audio metric guide:
- peak_db: peak level in dBFS (above -1 = clipping risk)
- rms_db: average loudness
- lufs: integrated loudness (-14 LUFS = streaming target)
- stereo_width: 0 = mono, 1 = fully wide
- spectral_centroid_hz: brightness center of sound
- sub/bass/low_mid/mid/high_mid/air: frequency band energy ratios (0–1)

Be specific. Reference actual track names. Sound like you heard the session."""
    user = (
        f"[SESSION]\n{d.get('session_ctx','')}\n\n"
        f"[DETECTED ISSUES]\n{d.get('problem_ctx','')}\n\n"
        f"[QUESTION]\n{d.get('prompt','')}"
    )
    text, tokens = ask_claude(system, user)
    return jsonify({"text": text, "tokens": tokens})

@app.route("/api/fix", methods=["POST"])
def api_fix():
    d = request.get_json() or {}
    action = d.get("action", "")
    if not action:
        return jsonify({"error": "No action"}), 400
    return jsonify(ableton_send(action, d.get("params", {})))

@app.route("/")
def index():
    return render_template_string(HTML)

# ── UI ─────────────────────────────────────────────────────────────────────────

HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Explore — AI Mix Engineer</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap" rel="stylesheet">
<style>
*{box-sizing:border-box;margin:0;padding:0;}
:root{
  --bg:#030507;--panel:#060A0F;--panel2:#0A1018;--panel3:#0D1520;
  --border:#162030;--border2:#1E2E40;--border3:#243848;
  --text:#D8EAF8;--dim:#486880;--dim2:#3A5268;
  --teal:#00A898;--teal2:#00D4C8;--purple:#7B2FD4;--gold:#C8A843;
  --red:#C84030;--green:#28B060;--orange:#C87030;--coral:#E05060;
  --teal-dim:rgba(0,168,152,.1);
}
body{background:var(--bg);color:var(--text);font-family:'Inter',system-ui,sans-serif;font-size:12px;height:100vh;display:flex;flex-direction:column;overflow:hidden;}

/* ── Header ── */
.hdr{display:flex;align-items:center;gap:12px;padding:0 16px;height:50px;border-bottom:1px solid var(--border);background:rgba(3,5,7,.97);flex-shrink:0;z-index:10;}
.brand{font-size:15px;font-weight:900;letter-spacing:5px;color:var(--teal);}
.brand-sub{font-size:7.5px;font-weight:700;letter-spacing:.22em;color:var(--dim);text-transform:uppercase;margin-top:1px;}
.status-dot{width:7px;height:7px;border-radius:50%;background:var(--red);flex-shrink:0;transition:background .3s;}
.status-dot.on{background:var(--green);}
.hdr-health{display:flex;align-items:center;gap:9px;padding:5px 12px;background:var(--panel2);border:1px solid var(--border2);border-radius:8px;}
.hdr-health-score{font-size:18px;font-weight:900;line-height:1;}
.hdr-health-sub{font-size:7px;font-weight:800;letter-spacing:.18em;text-transform:uppercase;color:var(--dim);margin-top:2px;}
.hdr-health-bar{width:64px;height:3px;background:var(--border2);border-radius:2px;margin-top:3px;overflow:hidden;}
.hdr-health-fill{height:100%;border-radius:2px;transition:width .5s;}
.hdr-right{margin-left:auto;display:flex;gap:5px;align-items:center;flex-wrap:nowrap;}
.pill{padding:4px 9px;border-radius:20px;font-size:8.5px;font-weight:800;letter-spacing:.1em;text-transform:uppercase;border:1px solid var(--border2);background:transparent;color:var(--dim);cursor:pointer;transition:all .15s;white-space:nowrap;}
.pill:hover{border-color:var(--teal);color:var(--teal);}
.pill.primary{background:var(--teal);color:#000;border-color:var(--teal);}
.pill.primary:hover{opacity:.85;}

/* ── Stats strip ── */
.stats-strip{display:flex;flex-shrink:0;border-bottom:1px solid var(--border);background:var(--panel);overflow:hidden;}
.stat-card{flex:1;display:flex;flex-direction:column;justify-content:center;padding:8px 14px;border-right:1px solid var(--border);position:relative;overflow:hidden;}
.stat-card:last-child{border-right:none;}
.stat-card::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;background:var(--stat-accent,var(--teal));opacity:.7;}
.stat-icon{font-size:9px;font-weight:800;letter-spacing:.15em;text-transform:uppercase;color:var(--dim);margin-bottom:2px;}
.stat-num{font-size:22px;font-weight:900;line-height:1;letter-spacing:-.02em;font-variant-numeric:tabular-nums;}
.stat-sub{font-size:8px;color:var(--dim);margin-top:2px;font-weight:500;}

/* ── Main layout ── */
.main{display:flex;flex:1;overflow:hidden;}

/* ── Sidebar ── */
.sidebar{width:200px;border-right:1px solid var(--border);display:flex;flex-direction:column;flex-shrink:0;}
.panel-hdr{padding:6px 11px;font-size:7.5px;font-weight:800;letter-spacing:.2em;color:var(--dim);text-transform:uppercase;border-bottom:1px solid var(--border);display:flex;align-items:center;justify-content:space-between;flex-shrink:0;}
.track-list{flex:1;overflow-y:auto;}
.track-row{padding:5px 9px;border-bottom:1px solid var(--border);display:flex;align-items:center;gap:6px;cursor:pointer;transition:background .1s;user-select:none;}
.track-row:hover{background:var(--panel2);}
.track-row.selected{background:var(--teal-dim);border-left:2px solid var(--teal);}
.track-num{font-size:8.5px;color:var(--dim);width:13px;flex-shrink:0;font-weight:700;font-variant-numeric:tabular-nums;}
.track-name{flex:1;font-size:10px;font-weight:600;overflow:hidden;white-space:nowrap;text-overflow:ellipsis;}
.track-type{font-size:7px;padding:1px 3px;border-radius:2px;font-weight:800;flex-shrink:0;}
.type-midi{background:rgba(0,168,152,.15);color:var(--teal);}
.type-audio{background:rgba(123,47,212,.15);color:var(--purple);}
.track-score{font-size:9.5px;font-weight:900;width:22px;text-align:right;flex-shrink:0;font-variant-numeric:tabular-nums;}

/* ── Center ── */
.center{flex:1;display:flex;flex-direction:column;overflow:hidden;border-right:1px solid var(--border);}

/* Charts strip */
.charts-strip{display:flex;flex-shrink:0;border-bottom:1px solid var(--border);height:190px;}
.chart-panel{flex:1;padding:10px 12px;border-right:1px solid var(--border);overflow:hidden;display:flex;flex-direction:column;}
.chart-panel:last-child{border-right:none;flex:0 0 200px;}
.chart-title{font-size:7.5px;font-weight:800;letter-spacing:.18em;text-transform:uppercase;color:var(--dim);margin-bottom:8px;flex-shrink:0;display:flex;align-items:center;gap:6px;}
.chart-title-track{color:var(--teal);font-weight:700;letter-spacing:.05em;text-transform:none;font-size:8px;}
.health-svg-wrap{flex:1;overflow:hidden;}
#health-svg{width:100%;display:block;}

/* Freq bands in chart area */
.freq-band{display:flex;align-items:center;gap:6px;margin-bottom:5px;}
.band-label{font-size:7.5px;font-weight:700;color:var(--dim);width:28px;flex-shrink:0;letter-spacing:.05em;}
.band-bar-wrap{flex:1;height:9px;background:var(--panel2);border-radius:3px;overflow:hidden;border:1px solid var(--border);}
.band-bar{height:100%;border-radius:3px;transition:width .4s ease;}
.band-pct{font-size:7.5px;font-weight:700;width:24px;text-align:right;flex-shrink:0;font-variant-numeric:tabular-nums;}

/* Problems strip */
.problems-strip{flex-shrink:0;border-bottom:1px solid var(--border);overflow-y:auto;max-height:100px;}
.prob-row{display:flex;align-items:flex-start;gap:8px;padding:6px 12px;border-bottom:1px solid var(--border);}
.prob-row:last-child{border-bottom:none;}
.prob-dot{width:5px;height:5px;border-radius:50%;flex-shrink:0;margin-top:4px;}
.prob-dot.high{background:var(--red);}
.prob-dot.medium{background:var(--orange);}
.prob-dot.low{background:var(--gold);}
.prob-dot.none{background:var(--green);}
.prob-body{flex:1;}
.prob-title-sm{font-weight:700;font-size:10.5px;line-height:1.4;}
.prob-detail-sm{color:var(--dim);font-size:9.5px;line-height:1.5;}
.prob-fix-sm{color:var(--teal);font-size:9.5px;line-height:1.5;margin-top:1px;}

/* Chat */
.chat-area{flex:1;overflow-y:auto;padding:14px 16px;display:flex;flex-direction:column;gap:10px;}
.msg{display:flex;flex-direction:column;gap:3px;max-width:90%;}
.msg.user{align-self:flex-end;}
.msg.ai{align-self:flex-start;}
.msg-bubble{padding:10px 14px;border-radius:12px;font-family:'Inter',system-ui,sans-serif;font-size:13px;font-weight:400;line-height:1.75;letter-spacing:.01em;}
.msg.user .msg-bubble{background:rgba(0,168,152,.1);border:1px solid rgba(0,168,152,.2);color:var(--text);}
.msg.ai .msg-bubble{background:var(--panel2);border:1px solid var(--border2);color:var(--text);}
.msg-meta{font-size:8.5px;color:var(--dim2);padding:0 3px;font-variant-numeric:tabular-nums;}
.input-row{padding:9px 12px;border-top:1px solid var(--border);display:flex;gap:6px;flex-shrink:0;background:var(--panel);}
.chat-input{flex:1;background:var(--panel2);border:1px solid var(--border2);border-radius:8px;padding:8px 12px;color:var(--text);font-size:12px;font-family:'Inter',system-ui,sans-serif;resize:none;height:38px;line-height:1.5;transition:border-color .15s;}
.chat-input:focus{outline:none;border-color:var(--teal);}
.send-btn{padding:0 14px;border-radius:8px;background:var(--teal);border:none;color:#000;font-size:9.5px;font-weight:800;letter-spacing:.1em;cursor:pointer;font-family:'Inter',system-ui,sans-serif;transition:opacity .15s;}
.send-btn:hover{opacity:.85;}
.send-btn:disabled{opacity:.35;cursor:not-allowed;}

/* ── Right panel ── */
.right{width:262px;display:flex;flex-direction:column;flex-shrink:0;overflow:hidden;}

/* Gain panel */
.gain-panel{flex-shrink:0;border-bottom:1px solid var(--border);}
.gain-mode-pill{font-size:7.5px;font-weight:800;letter-spacing:.15em;text-transform:uppercase;padding:2px 7px;border-radius:10px;background:rgba(0,168,152,.15);color:var(--teal);}
.gain-mode-pill.build{background:rgba(123,47,212,.15);color:var(--purple);}

/* Donut grid */
.donut-grid{display:grid;grid-template-columns:1fr 1fr;gap:1px;background:var(--border);border-top:1px solid var(--border);border-bottom:1px solid var(--border);}
.donut-cell{background:var(--panel2);display:flex;flex-direction:column;align-items:center;justify-content:center;padding:10px 6px;}
.donut-name{font-size:7.5px;font-weight:800;letter-spacing:.12em;text-transform:uppercase;color:var(--dim);margin-top:3px;}
.donut-svg{overflow:visible;}
.donut-bg{fill:none;stroke:#162030;stroke-width:7;}
.donut-arc{fill:none;stroke-width:7;stroke-linecap:round;stroke-dasharray:0 175.9;transition:stroke-dasharray .6s ease;}
.donut-pct{font-size:12.5px;font-weight:900;fill:var(--text);text-anchor:middle;font-family:'Inter',sans-serif;}
.donut-sub{font-size:6.5px;fill:#486880;text-anchor:middle;font-family:'Inter',sans-serif;letter-spacing:.06em;}

/* Knob bars */
.knob-bars{padding:8px 12px;}
.knob-row{display:flex;align-items:center;gap:7px;margin-bottom:6px;}
.knob-row:last-child{margin-bottom:0;}
.knob-lbl{font-size:7.5px;font-weight:700;color:var(--dim);width:50px;flex-shrink:0;letter-spacing:.05em;}
.knob-bar-wrap{flex:1;height:5px;background:var(--border2);border-radius:3px;overflow:hidden;}
.knob-bar-fill{height:100%;border-radius:3px;transition:width .4s ease;}
.knob-val{font-size:7.5px;font-weight:700;width:24px;text-align:right;flex-shrink:0;font-variant-numeric:tabular-nums;}

/* Inspector */
.inspector{flex:1;overflow-y:auto;border-bottom:1px solid var(--border);}
.inspector-empty{display:flex;align-items:center;justify-content:center;height:100%;color:var(--dim2);font-size:10px;text-align:center;padding:16px;}
.track-health-banner{padding:9px 11px;border-bottom:1px solid var(--border);}
.th-name{font-size:11px;font-weight:800;margin-bottom:1px;overflow:hidden;white-space:nowrap;text-overflow:ellipsis;}
.th-meta{font-size:8.5px;color:var(--dim);margin-bottom:6px;}
.th-score-row{display:flex;align-items:center;gap:8px;}
.th-score{font-size:24px;font-weight:900;line-height:1;}
.th-score-lbl{font-size:7px;font-weight:800;letter-spacing:.15em;text-transform:uppercase;color:var(--dim);}
.th-bar-wrap{flex:1;height:4px;background:var(--border2);border-radius:2px;overflow:hidden;}
.th-bar-fill{height:100%;border-radius:2px;transition:width .5s;}
.device-block{border-bottom:1px solid var(--border);padding:8px 11px;}
.device-name{font-size:10.5px;font-weight:700;margin-bottom:5px;}
.param-row{display:flex;justify-content:space-between;padding:2px 0;border-bottom:1px solid var(--border);}
.param-name{font-size:8.5px;color:var(--dim);overflow:hidden;white-space:nowrap;text-overflow:ellipsis;max-width:110px;}
.param-val{font-size:8.5px;font-weight:700;color:var(--text);font-variant-numeric:tabular-nums;}

/* Scan box */
.scan-box{flex-shrink:0;padding:9px 11px;border-top:1px solid var(--border);}
.scan-label{font-size:7.5px;font-weight:800;letter-spacing:.15em;color:var(--dim);text-transform:uppercase;margin-bottom:5px;display:block;}
.path-input{width:100%;background:var(--panel2);border:1px solid var(--border2);border-radius:5px;padding:5px 7px;color:var(--text);font-size:9.5px;font-family:'Inter',system-ui,sans-serif;margin-bottom:5px;}
.path-input:focus{outline:none;border-color:var(--teal);}
.scan-btn{width:100%;padding:6px;border-radius:5px;background:var(--purple);border:none;color:#fff;font-size:9px;font-weight:800;letter-spacing:.1em;cursor:pointer;font-family:'Inter',system-ui,sans-serif;text-transform:uppercase;transition:opacity .15s;}
.scan-btn:hover{opacity:.85;}
.scan-btn:disabled{opacity:.35;}

/* Misc */
::-webkit-scrollbar{width:3px;}
::-webkit-scrollbar-track{background:transparent;}
::-webkit-scrollbar-thumb{background:var(--border2);border-radius:2px;}
.empty{display:flex;flex-direction:column;align-items:center;justify-content:center;height:100%;color:var(--dim2);gap:6px;padding:16px;text-align:center;font-size:10.5px;}
.empty-icon{font-size:22px;opacity:.2;}
.toast{position:fixed;bottom:14px;left:50%;transform:translateX(-50%);background:var(--teal);color:#000;padding:7px 16px;border-radius:6px;font-size:10.5px;font-weight:800;opacity:0;transition:opacity .2s;pointer-events:none;z-index:999;white-space:nowrap;}
.toast.show{opacity:1;}
.loading{color:var(--dim);font-style:italic;}
</style>
</head>
<body>

<!-- ── Header ──────────────────────────────────────────────────────── -->
<div class="hdr">
  <div>
    <div class="brand">EXPLORE</div>
    <div class="brand-sub">AI Mix Engineer</div>
  </div>
  <div id="ableton-dot" class="status-dot" title="Ableton Live"></div>

  <div class="hdr-health" id="hdr-health" style="display:none">
    <div>
      <div id="hdr-health-score" class="hdr-health-score">—</div>
      <div id="hdr-health-lbl" class="hdr-health-sub">Mix Health</div>
      <div class="hdr-health-bar"><div id="hdr-health-fill" class="hdr-health-fill" style="width:0%"></div></div>
    </div>
  </div>

  <div class="hdr-right">
    <button class="pill primary" onclick="runScan()">⟳ Scan</button>
    <button class="pill" onclick="clearChat()" title="Clear chat history">✕ Chat</button>
    <button class="pill" onclick="quickPrompt('Why does this mix sound muddy?')">Mud?</button>
    <button class="pill" onclick="quickPrompt('What is taking up the most space in the mix?')">Space?</button>
    <button class="pill" onclick="quickPrompt('What should I work on first?')">Priority?</button>
    <button class="pill" onclick="quickPrompt('How is the low end balance?')">Low End?</button>
    <button class="pill" onclick="quickPrompt('Where is the vocal sitting in the mix?')">Vocal?</button>
    <button class="pill" onclick="loadArrangement()">Arrange</button>
  </div>
</div>

<!-- ── Stats strip ─────────────────────────────────────────────────── -->
<div class="stats-strip">
  <div class="stat-card" style="--stat-accent:var(--teal)">
    <div class="stat-icon">♩ BPM</div>
    <div class="stat-num" id="stat-bpm">—</div>
    <div class="stat-sub" id="stat-timesig">—/—</div>
  </div>
  <div class="stat-card" style="--stat-accent:var(--purple)">
    <div class="stat-icon">▤ TRACKS</div>
    <div class="stat-num" id="stat-tracks">—</div>
    <div class="stat-sub" id="stat-track-types">—</div>
  </div>
  <div class="stat-card" style="--stat-accent:var(--green)">
    <div class="stat-icon">◈ MIX HEALTH</div>
    <div class="stat-num" id="stat-health" style="color:var(--green)">—</div>
    <div class="stat-sub" id="stat-health-lbl">—</div>
  </div>
  <div class="stat-card" style="--stat-accent:var(--teal)">
    <div class="stat-icon">◎ GAIN MODE</div>
    <div class="stat-num" id="stat-gain-mode" style="font-size:14px;padding-top:4px">—</div>
    <div class="stat-sub" id="stat-gain-voice">—</div>
  </div>
  <div class="stat-card" style="--stat-accent:var(--orange)">
    <div class="stat-icon">⬆ INTENSITY</div>
    <div class="stat-num" id="stat-intensity">—</div>
    <div class="stat-sub" id="stat-certainty">certainty —</div>
  </div>
  <div class="stat-card" style="--stat-accent:var(--gold)">
    <div class="stat-icon">⊞ SCOPE</div>
    <div class="stat-num" id="stat-scope">—</div>
    <div class="stat-sub" id="stat-room">room —</div>
  </div>
</div>

<!-- ── Main ────────────────────────────────────────────────────────── -->
<div class="main">

  <!-- Sidebar: track list -->
  <div class="sidebar">
    <div class="panel-hdr">
      Tracks
      <span id="track-count-lbl" style="color:var(--text);font-size:9px;font-weight:900;letter-spacing:0"></span>
    </div>
    <div class="track-list" id="track-list">
      <div class="empty"><div class="empty-icon">🎚</div><div>Scan to load tracks</div></div>
    </div>
  </div>

  <!-- Center -->
  <div class="center">

    <!-- Charts strip -->
    <div class="charts-strip">
      <!-- Health bar chart -->
      <div class="chart-panel">
        <div class="chart-title">Track Health</div>
        <div class="health-svg-wrap">
          <svg id="health-svg" xmlns="http://www.w3.org/2000/svg" height="170"></svg>
        </div>
      </div>
      <!-- Freq map -->
      <div class="chart-panel" id="freq-chart-panel">
        <div class="chart-title">
          Frequency Map
          <span class="chart-title-track" id="freq-track-name"></span>
        </div>
        <div id="freq-bands" style="flex:1;display:flex;flex-direction:column;justify-content:center">
          <div style="color:var(--dim2);font-size:9.5px;text-align:center">Click a track</div>
        </div>
      </div>
    </div>

    <!-- Problems strip -->
    <div class="problems-strip" id="problems-strip">
      <div class="prob-row">
        <div class="prob-dot none"></div>
        <div class="prob-body"><div class="prob-detail-sm">Run a scan to detect mix problems</div></div>
      </div>
    </div>

    <!-- Chat -->
    <div class="chat-area" id="chat-area">
      <div class="msg ai">
        <div class="msg-bubble">I'm Explore — your AI mix engineer. The dashboard is loading your session and Gain state now.<br><br>Hit <strong>Scan</strong> to refresh tracks, or just ask me anything about your mix.</div>
      </div>
    </div>

    <!-- Input -->
    <div class="input-row">
      <textarea class="chat-input" id="chat-input" placeholder="Ask about your mix..."></textarea>
      <button class="send-btn" id="send-btn" onclick="sendMessage()">Ask</button>
    </div>
  </div>

  <!-- Right panel -->
  <div class="right">

    <!-- Gain panel -->
    <div class="gain-panel">
      <div class="panel-hdr">
        Gain
        <span class="gain-mode-pill" id="gain-mode-pill">—</span>
      </div>

      <!-- 2x2 donut grid -->
      <div class="donut-grid">
        <!-- INTENSITY (Track 1 fader) -->
        <div class="donut-cell">
          <svg class="donut-svg" viewBox="0 0 80 80" width="76" height="76">
            <defs>
              <linearGradient id="dg-intens" x1="0" y1="0" x2="1" y2="1">
                <stop offset="0%" stop-color="#00A898"/>
                <stop offset="100%" stop-color="#00D4C8"/>
              </linearGradient>
            </defs>
            <circle class="donut-bg" cx="40" cy="40" r="28"/>
            <circle class="donut-arc" id="arc-intensity" cx="40" cy="40" r="28"
              stroke="url(#dg-intens)" transform="rotate(-90 40 40)"/>
            <text class="donut-pct" id="pct-intensity" x="40" y="36">—</text>
            <text class="donut-sub" x="40" y="50">INTENS</text>
          </svg>
          <div class="donut-name">Mode</div>
        </div>
        <!-- CERTAINTY (Track 2 fader) -->
        <div class="donut-cell">
          <svg class="donut-svg" viewBox="0 0 80 80" width="76" height="76">
            <defs>
              <linearGradient id="dg-cert" x1="0" y1="0" x2="1" y2="1">
                <stop offset="0%" stop-color="#7B2FD4"/>
                <stop offset="100%" stop-color="#B060FF"/>
              </linearGradient>
            </defs>
            <circle class="donut-bg" cx="40" cy="40" r="28"/>
            <circle class="donut-arc" id="arc-certainty" cx="40" cy="40" r="28"
              stroke="url(#dg-cert)" transform="rotate(-90 40 40)"/>
            <text class="donut-pct" id="pct-certainty" x="40" y="36">—</text>
            <text class="donut-sub" x="40" y="50">CERTAINT</text>
          </svg>
          <div class="donut-name">Confidence</div>
        </div>
        <!-- SCOPE (Track 3 fader) -->
        <div class="donut-cell">
          <svg class="donut-svg" viewBox="0 0 80 80" width="76" height="76">
            <defs>
              <linearGradient id="dg-scope" x1="0" y1="0" x2="1" y2="1">
                <stop offset="0%" stop-color="#C8A843"/>
                <stop offset="100%" stop-color="#F0D060"/>
              </linearGradient>
            </defs>
            <circle class="donut-bg" cx="40" cy="40" r="28"/>
            <circle class="donut-arc" id="arc-scope" cx="40" cy="40" r="28"
              stroke="url(#dg-scope)" transform="rotate(-90 40 40)"/>
            <text class="donut-pct" id="pct-scope" x="40" y="36">—</text>
            <text class="donut-sub" x="40" y="50">SCOPE</text>
          </svg>
          <div class="donut-name">Scope</div>
        </div>
        <!-- ROOM (Track 4 fader) -->
        <div class="donut-cell">
          <svg class="donut-svg" viewBox="0 0 80 80" width="76" height="76">
            <defs>
              <linearGradient id="dg-room" x1="0" y1="0" x2="1" y2="1">
                <stop offset="0%" stop-color="#C87030"/>
                <stop offset="100%" stop-color="#E05060"/>
              </linearGradient>
            </defs>
            <circle class="donut-bg" cx="40" cy="40" r="28"/>
            <circle class="donut-arc" id="arc-room" cx="40" cy="40" r="28"
              stroke="url(#dg-room)" transform="rotate(-90 40 40)"/>
            <text class="donut-pct" id="pct-room" x="40" y="36">—</text>
            <text class="donut-sub" x="40" y="50">ROOM</text>
          </svg>
          <div class="donut-name">Voice</div>
        </div>
      </div>

      <!-- Knob bars -->
      <div class="knob-bars">
        <div class="knob-row">
          <span class="knob-lbl">DEPTH</span>
          <div class="knob-bar-wrap"><div class="knob-bar-fill" id="knob-depth" style="width:0%;background:var(--teal)"></div></div>
          <span class="knob-val" id="knob-depth-val">—</span>
        </div>
        <div class="knob-row">
          <span class="knob-lbl">RISK</span>
          <div class="knob-bar-wrap"><div class="knob-bar-fill" id="knob-risk" style="width:0%;background:var(--purple)"></div></div>
          <span class="knob-val" id="knob-risk-val">—</span>
        </div>
        <div class="knob-row">
          <span class="knob-lbl">BANDWDTH</span>
          <div class="knob-bar-wrap"><div class="knob-bar-fill" id="knob-bandwidth" style="width:0%;background:var(--gold)"></div></div>
          <span class="knob-val" id="knob-bandwidth-val">—</span>
        </div>
        <div class="knob-row">
          <span class="knob-lbl">DECAY</span>
          <div class="knob-bar-wrap"><div class="knob-bar-fill" id="knob-decay" style="width:0%;background:var(--orange)"></div></div>
          <span class="knob-val" id="knob-decay-val">—</span>
        </div>
      </div>
    </div>

    <!-- Track inspector -->
    <div class="inspector" id="inspector">
      <div class="inspector-empty">Click a track to inspect devices &amp; parameters</div>
    </div>

    <!-- Scan box -->
    <div class="scan-box">
      <label class="scan-label">Project Folder</label>
      <input class="path-input" id="project-path" placeholder="/Users/you/Music/Project" type="text">
      <button class="scan-btn" id="scan-btn" onclick="scanAudio()">Scan Audio Files</button>
    </div>
  </div>

</div>

<div class="toast" id="toast"></div>

<script>
// ── State ──────────────────────────────────────────────────────────────────────
var sessionCtx = '';
var problemCtx = '';
var allTracks = [];
var allAudioData = {};
var trackScores = {};
var overallHealth = 0;
var selectedTrackIdx = -1;
var gainData = {};

// ── Persistence ───────────────────────────────────────────────────────────────
var chatHistory = [];
var STORAGE_KEY = 'explore_v1';

function saveState() {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify({
      projectPath:   document.getElementById('project-path').value,
      chatHistory:   chatHistory.slice(-60),
      sessionCtx:    sessionCtx,
      problemCtx:    problemCtx,
      trackScores:   trackScores,
      overallHealth: overallHealth,
      allTracks:     allTracks,
    }));
  } catch(e) {}
}

function loadState() {
  try {
    var raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return;
    var d = JSON.parse(raw);
    if (d.projectPath) document.getElementById('project-path').value = d.projectPath;
    if (d.sessionCtx)    sessionCtx    = d.sessionCtx;
    if (d.problemCtx)    problemCtx    = d.problemCtx;
    if (d.trackScores)   trackScores   = d.trackScores;
    if (d.overallHealth) overallHealth = d.overallHealth;
    if (d.allTracks)     allTracks     = d.allTracks;
    if (d.allTracks && d.allTracks.length) {
      renderTracks(d.allTracks);
      renderHealthChart(d.allTracks, d.trackScores || {});
      document.getElementById('ableton-dot').classList.add('on');
    }
    if (d.overallHealth) {
      renderOverallHealth(d.overallHealth, d.allTracks || [], []);
      updateStatCards(null, d.allTracks || [], d.overallHealth);
    }
    if (d.chatHistory && d.chatHistory.length) {
      chatHistory = d.chatHistory;
      var area = document.getElementById('chat-area');
      area.innerHTML = '';
      chatHistory.forEach(function(m) {
        var div = document.createElement('div');
        div.className = 'msg ' + m.role;
        div.innerHTML = '<div class="msg-bubble">' + renderText(m.text) + '</div>'
          + '<div class="msg-meta">' + esc(m.meta || '') + '</div>';
        area.appendChild(div);
      });
      scrollChat();
    }
  } catch(e) {}
}

function clearChat() {
  chatHistory = [];
  var area = document.getElementById('chat-area');
  area.innerHTML = '<div class="msg ai"><div class="msg-bubble">Chat cleared. Session data is still loaded.</div></div>';
  saveState();
}

// ── Helpers ───────────────────────────────────────────────────────────────────
function healthColor(s) {
  return s >= 80 ? 'var(--green)' : s >= 60 ? 'var(--gold)' : 'var(--red)';
}
function healthLabel(s) {
  if (s >= 85) return 'EXCELLENT';
  if (s >= 70) return 'GOOD';
  if (s >= 55) return 'FAIR';
  if (s >= 40) return 'WEAK';
  return 'CRITICAL';
}
function esc(s) {
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}
function renderText(s) {
  return esc(s).replace(/\n/g,'<br>');
}
function pct(v) { return Math.round((v||0) * 100) + '%'; }

// ── Stat cards ────────────────────────────────────────────────────────────────
function updateStatCards(session, tracks, health) {
  if (session) {
    document.getElementById('stat-bpm').textContent = session.tempo ? Math.round(session.tempo) : '—';
    document.getElementById('stat-timesig').textContent =
      (session.signature_numerator || '?') + '/' + (session.signature_denominator || '?');
  }
  if (tracks && tracks.length) {
    document.getElementById('stat-tracks').textContent = tracks.length;
    var midi = tracks.filter(function(t) { return t.is_midi_track; }).length;
    var audio = tracks.length - midi;
    document.getElementById('stat-track-types').textContent = midi + ' MIDI · ' + audio + ' Audio';
  }
  if (health !== undefined) {
    var el = document.getElementById('stat-health');
    el.textContent = health;
    el.style.color = healthColor(health);
    document.getElementById('stat-health-lbl').textContent = healthLabel(health);
  }
}

// ── Gain ──────────────────────────────────────────────────────────────────────
function updateDonut(id, value) {
  var circ = 175.9;
  var dash = (Math.min(1, Math.max(0, value || 0)) * circ).toFixed(1);
  var arc = document.getElementById('arc-' + id);
  if (arc) arc.setAttribute('stroke-dasharray', dash + ' ' + circ);
  var pctEl = document.getElementById('pct-' + id);
  if (pctEl) pctEl.textContent = Math.round((value || 0) * 100) + '%';
}

function updateKnob(id, value) {
  var fill = document.getElementById('knob-' + id);
  if (fill) fill.style.width = Math.round((value || 0) * 100) + '%';
  var val = document.getElementById('knob-' + id + '-val');
  if (val) val.textContent = Math.round((value || 0) * 100) + '%';
}

function renderGain(g) {
  if (!g || g.error) return;
  gainData = g;

  // Mode pill
  var mode = g.mode || '';
  var pill = document.getElementById('gain-mode-pill');
  pill.textContent = mode || '—';
  pill.className = 'gain-mode-pill' + (mode === 'BUILD' ? ' build' : '');

  // Stat cards
  var voice = g.voice || g.filter || '';
  document.getElementById('stat-gain-mode').textContent = mode || '—';
  document.getElementById('stat-gain-voice').textContent = voice ? 'voice: ' + voice : '—';
  document.getElementById('stat-intensity').textContent = pct(g.intensity);
  document.getElementById('stat-certainty').textContent = 'certainty ' + pct(g.certainty);
  document.getElementById('stat-scope').textContent = pct(g.scope);
  document.getElementById('stat-room').textContent = 'room ' + pct(g.room);

  // Donuts
  updateDonut('intensity', g.intensity);
  updateDonut('certainty', g.certainty);
  updateDonut('scope',     g.scope);
  updateDonut('room',      g.room);

  // Knob bars
  updateKnob('depth',     g.depth);
  updateKnob('risk',      g.risk);
  updateKnob('bandwidth', g.bandwidth);
  updateKnob('decay',     g.decay);
}

async function fetchGain() {
  try {
    var r = await fetch('/api/gain');
    var d = await r.json();
    renderGain(d);
  } catch(e) {}
}

// ── Scan ──────────────────────────────────────────────────────────────────────
var scanPromise = null;

function ensureScanned() {
  if (sessionCtx) return Promise.resolve();
  if (scanPromise) return scanPromise;
  scanPromise = runScan().then(function() { scanPromise = null; });
  return scanPromise;
}

async function runScan() {
  toast('Scanning session...');
  try {
    var r = await fetch('/api/analyze', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({project_path: document.getElementById('project-path').value.trim()})
    });
    var d = await r.json();
    document.getElementById('ableton-dot').classList.add('on');
    sessionCtx    = d.session_ctx || '';
    problemCtx    = d.problem_ctx || '';
    allTracks     = d.tracks || [];
    allAudioData  = d.audio_data || {};
    trackScores   = d.track_scores || {};
    overallHealth = d.overall_health || 0;

    renderOverallHealth(overallHealth, allTracks, d.problems || []);
    updateStatCards(d.session, allTracks, overallHealth);
    renderTracks(allTracks);
    renderProblems(d.problems || []);
    renderHealthChart(allTracks, trackScores);
    saveState();
    toast('Loaded ' + allTracks.length + ' tracks');
  } catch(e) {
    document.getElementById('ableton-dot').classList.remove('on');
    toast('Cannot reach Ableton', true);
  }
}

async function scanAudio() {
  var path = document.getElementById('project-path').value.trim();
  if (!path) { toast('Enter project folder path first', true); return; }
  var btn = document.getElementById('scan-btn');
  btn.disabled = true; btn.textContent = 'Scanning...';
  await runScan();
  btn.disabled = false; btn.textContent = 'Scan Audio Files';
}

// ── Render ────────────────────────────────────────────────────────────────────
function renderOverallHealth(score, tracks, problems) {
  var el = document.getElementById('hdr-health');
  el.style.display = 'flex';
  var col = healthColor(score);
  document.getElementById('hdr-health-score').textContent = score;
  document.getElementById('hdr-health-score').style.color = col;
  document.getElementById('hdr-health-lbl').textContent = healthLabel(score);
  var fill = document.getElementById('hdr-health-fill');
  fill.style.width = score + '%';
  fill.style.background = col;
}

function renderTracks(tracks) {
  document.getElementById('track-count-lbl').textContent = tracks.length;
  var el = document.getElementById('track-list');
  if (!tracks.length) {
    el.innerHTML = '<div class="empty"><div>No tracks</div></div>';
    return;
  }
  var html = '';
  for (var i = 0; i < tracks.length; i++) {
    var t = tracks[i];
    var name = t.name || ('Track ' + (i+1));
    var badge = t.is_midi_track
      ? '<span class="track-type type-midi">M</span>'
      : '<span class="track-type type-audio">A</span>';
    var score = trackScores[name];
    var scoreHtml = score !== undefined
      ? '<span class="track-score" style="color:' + healthColor(score) + '">' + score + '</span>'
      : '';
    html += '<div class="track-row" id="trow-' + i + '" onclick="selectTrack(' + i + ')">'
      + '<span class="track-num">' + (i+1) + '</span>'
      + '<span class="track-name" title="' + esc(name) + '">' + esc(name) + '</span>'
      + badge + scoreHtml + '</div>';
  }
  el.innerHTML = html;
}

function renderProblems(problems) {
  var el = document.getElementById('problems-strip');
  if (!problems.length) {
    el.innerHTML = '<div class="prob-row"><div class="prob-dot none"></div>'
      + '<div class="prob-body"><div class="prob-detail-sm">No issues detected</div></div></div>';
    return;
  }
  var html = '';
  for (var i = 0; i < problems.length; i++) {
    var p = problems[i];
    html += '<div class="prob-row"><div class="prob-dot ' + esc(p.severity) + '"></div>'
      + '<div class="prob-body"><div class="prob-title-sm">' + esc(p.title) + '</div>'
      + '<div class="prob-detail-sm">' + esc(p.detail) + '</div>'
      + (p.fix ? '<div class="prob-fix-sm">→ ' + esc(p.fix) + '</div>' : '')
      + '</div></div>';
  }
  el.innerHTML = html;
}

function renderHealthChart(tracks, scores) {
  var svgEl = document.getElementById('health-svg');
  if (!svgEl || !tracks.length) { if (svgEl) svgEl.innerHTML = ''; return; }
  var rowH = 20;
  var barX = 108;
  var maxW = 155;
  var totalH = tracks.length * rowH + 4;
  svgEl.setAttribute('viewBox', '0 0 280 ' + totalH);
  svgEl.setAttribute('height', Math.min(totalH, 170));
  var html = '<defs>'
    + '<linearGradient id="hg-g" x1="0" y1="0" x2="1" y2="0">'
    + '<stop offset="0%" stop-color="#009690"/><stop offset="100%" stop-color="#28B060"/>'
    + '</linearGradient>'
    + '<linearGradient id="hg-y" x1="0" y1="0" x2="1" y2="0">'
    + '<stop offset="0%" stop-color="#C87030"/><stop offset="100%" stop-color="#C8A843"/>'
    + '</linearGradient>'
    + '<linearGradient id="hg-r" x1="0" y1="0" x2="1" y2="0">'
    + '<stop offset="0%" stop-color="#C84030"/><stop offset="100%" stop-color="#C87030"/>'
    + '</linearGradient>'
    + '</defs>';
  for (var i = 0; i < tracks.length; i++) {
    var t = tracks[i];
    var name = (t.name || ('Track ' + (i+1))).substring(0, 14);
    var score = scores[t.name] || 72;
    var barW = Math.round((score / 100) * maxW);
    var grad = score >= 80 ? 'url(#hg-g)' : score >= 60 ? 'url(#hg-y)' : 'url(#hg-r)';
    var col  = score >= 80 ? '#28B060'    : score >= 60 ? '#C8A843'    : '#C84030';
    var y = i * rowH + 2;
    html += '<g transform="translate(0,' + y + ')">'
      + '<text x="0" y="12" font-size="8.5" fill="#486880" font-family="Inter,sans-serif">' + esc(name) + '</text>'
      + '<rect x="' + barX + '" y="3" width="' + barW + '" height="10" rx="3" fill="' + grad + '" opacity=".85"/>'
      + '<text x="' + (barX + barW + 5) + '" y="12" font-size="8.5" fill="' + col + '" font-family="Inter,sans-serif" font-weight="700">' + score + '</text>'
      + '</g>';
  }
  svgEl.innerHTML = html;
}

function renderFreqMap(adat, trackName) {
  var el = document.getElementById('freq-bands');
  var label = document.getElementById('freq-track-name');
  if (!adat || adat.peak_db === undefined) {
    el.innerHTML = '<div style="color:var(--dim2);font-size:9.5px;text-align:center">No audio data</div>';
    label.textContent = '';
    return;
  }
  label.textContent = trackName || '';
  var bands = [
    {label:'SUB',  key:'sub_energy'},
    {label:'BASS', key:'bass_energy'},
    {label:'LMID', key:'low_mid_energy'},
    {label:'MID',  key:'mid_energy'},
    {label:'HMID', key:'high_mid_energy'},
    {label:'AIR',  key:'air_energy'},
  ];
  var html = '';
  for (var i = 0; i < bands.length; i++) {
    var b = bands[i];
    var raw = adat[b.key] || 0;
    var p = Math.round(raw * 100);
    var col = (b.label === 'LMID' && p > 25) ? 'var(--red)'
            : (b.label === 'LMID' && p > 18) ? 'var(--orange)'
            : (b.label === 'SUB'  && p > 25) ? 'var(--orange)'
            : (b.label === 'BASS' && p > 35) ? 'var(--orange)'
            : 'var(--teal)';
    var barW = Math.min(100, Math.round(p * 2.5));
    html += '<div class="freq-band">'
      + '<span class="band-label">' + b.label + '</span>'
      + '<div class="band-bar-wrap"><div class="band-bar" style="width:' + barW + '%;background:' + col + '"></div></div>'
      + '<span class="band-pct" style="color:' + col + '">' + p + '%</span>'
      + '</div>';
  }
  el.innerHTML = html;
}

// ── Track selection + Device inspector ────────────────────────────────────────
async function selectTrack(i) {
  selectedTrackIdx = i;
  document.querySelectorAll('.track-row').forEach(function(r, j) {
    r.classList.toggle('selected', i === j);
  });
  var t = allTracks[i];
  if (!t) return;
  var name = t.name || ('Track ' + (i+1));
  var score = trackScores[name];
  var adat = allAudioData[name] || {};

  // Freq map in charts area
  renderFreqMap(adat, name);

  // Inspector
  var col = score !== undefined ? healthColor(score) : 'var(--dim)';
  var lbl = score !== undefined ? healthLabel(score) : '';
  var scoreNum = score !== undefined ? score : '—';
  var barW = score !== undefined ? score : 0;
  var devices = t.devices || [];
  var metaStr = (t.is_midi_track ? 'MIDI' : 'Audio') + ' · ' + devices.length + ' device' + (devices.length !== 1 ? 's' : '');
  if (adat.peak_db !== undefined) metaStr += ' · peak ' + adat.peak_db + 'dB';

  var html = '<div class="track-health-banner">'
    + '<div class="th-name">' + esc(name) + '</div>'
    + '<div class="th-meta">' + esc(metaStr) + '</div>'
    + '<div class="th-score-row">'
    + '<div><div class="th-score" style="color:' + col + '">' + scoreNum + '</div>'
    + '<div class="th-score-lbl">' + esc(lbl) + '</div></div>'
    + '<div class="th-bar-wrap"><div class="th-bar-fill" style="width:' + barW + '%;background:' + col + '"></div></div>'
    + '</div></div>';

  for (var d = 0; d < devices.length; d++) {
    var dev = devices[d];
    if (dev && dev.name) {
      html += '<div class="device-block"><div class="device-name">' + esc(dev.name) + '</div></div>';
    }
  }
  html += '<div id="device-params-area" style="padding:8px 11px;font-size:9px;color:var(--dim)">Loading params...</div>';
  document.getElementById('inspector').innerHTML = html;
  loadDeviceParams(i);
}

async function loadDeviceParams(trackIdx) {
  try {
    var r = await fetch('/api/devices/' + trackIdx);
    var d = await r.json();
    var el = document.getElementById('device-params-area');
    if (!el) return;
    if (d.error) { el.textContent = 'No params: ' + d.error; return; }
    var devs = Array.isArray(d) ? d : (d.devices || []);
    if (!devs.length) { el.textContent = 'No parameter data.'; return; }
    var html = '';
    for (var i = 0; i < devs.length; i++) {
      var dev = devs[i];
      var params = dev.parameters || [];
      if (!params.length) continue;
      html += '<div style="margin-bottom:8px"><div style="font-size:9px;font-weight:800;color:var(--text);margin-bottom:4px">' + esc(dev.name || '') + '</div>';
      var shown = params.slice(0, 10);
      for (var j = 0; j < shown.length; j++) {
        var p = shown[j];
        html += '<div class="param-row">'
          + '<span class="param-name">' + esc(p.name || '') + '</span>'
          + '<span class="param-val">' + esc(String(p.value !== undefined ? p.value : '—').substring(0,10)) + '</span>'
          + '</div>';
      }
      if (params.length > 10) html += '<div style="font-size:7.5px;color:var(--dim2);margin-top:2px">+' + (params.length-10) + ' more</div>';
      html += '</div>';
    }
    el.innerHTML = html || '<span style="color:var(--dim2)">No parameters found.</span>';
  } catch(e) {
    var el2 = document.getElementById('device-params-area');
    if (el2) el2.textContent = 'Could not load parameters.';
  }
}

// ── Chat ──────────────────────────────────────────────────────────────────────
async function quickPrompt(p) {
  document.getElementById('chat-input').value = p;
  if (!sessionCtx) { toast('Scanning session first...'); await ensureScanned(); }
  sendMessage();
}

async function sendMessage() {
  var input = document.getElementById('chat-input');
  var prompt = input.value.trim();
  if (!prompt) return;
  if (!sessionCtx) { toast('Scanning session first...'); await ensureScanned(); }
  input.value = '';
  chatHistory.push({role:'user', text:prompt, meta:''});
  addMessage('user', prompt);
  var btn = document.getElementById('send-btn');
  btn.disabled = true;
  var thinking = addThinking();
  try {
    var r = await fetch('/api/ask', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({prompt:prompt, session_ctx:sessionCtx, problem_ctx:problemCtx})
    });
    var d = await r.json();
    var aiText = d.text || 'No response.';
    var aiMeta = (d.tokens||'') + ' tok';
    thinking.querySelector('.msg-bubble').innerHTML = renderText(aiText);
    thinking.querySelector('.msg-meta').textContent = aiMeta;
    chatHistory.push({role:'ai', text:aiText, meta:aiMeta});
    saveState();
  } catch(e) {
    thinking.querySelector('.msg-bubble').textContent = 'Error connecting to server.';
  }
  btn.disabled = false;
  scrollChat();
}

function addMessage(role, text) {
  var area = document.getElementById('chat-area');
  var div = document.createElement('div');
  div.className = 'msg ' + role;
  div.innerHTML = '<div class="msg-bubble">' + renderText(text) + '</div><div class="msg-meta"></div>';
  area.appendChild(div);
  scrollChat();
  return div;
}

function addThinking() {
  var area = document.getElementById('chat-area');
  var div = document.createElement('div');
  div.className = 'msg ai';
  div.innerHTML = '<div class="msg-bubble loading">Thinking...</div><div class="msg-meta"></div>';
  area.appendChild(div);
  scrollChat();
  return div;
}

function scrollChat() {
  var area = document.getElementById('chat-area');
  area.scrollTop = area.scrollHeight;
}

// ── Arrangement ───────────────────────────────────────────────────────────────
async function loadArrangement() {
  toast('Loading arrangement...');
  try {
    var r = await fetch('/api/arrangement');
    var d = await r.json();
    if (d.error) { toast('Arrangement: ' + d.error, true); return; }
    addMessage('ai', 'Arrangement info:\n\n' + JSON.stringify(d, null, 2).substring(0, 600));
    toast('Arrangement loaded');
  } catch(e) { toast('Could not load arrangement', true); }
}

// ── Input ─────────────────────────────────────────────────────────────────────
document.getElementById('chat-input').addEventListener('keydown', function(e) {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
});
document.getElementById('project-path').addEventListener('change', saveState);

// ── Toast ─────────────────────────────────────────────────────────────────────
var toastTimer;
function toast(msg, err) {
  var el = document.getElementById('toast');
  el.textContent = msg;
  el.style.background = err ? 'var(--red)' : 'var(--teal)';
  el.style.color = err ? '#fff' : '#000';
  el.classList.add('show');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(function() { el.classList.remove('show'); }, 2800);
}

// ── Init ──────────────────────────────────────────────────────────────────────
loadState();
fetchGain();
setInterval(fetchGain, 2000);
ensureScanned();
</script>
</body>
</html>"""

if __name__ == "__main__":
    print(f"Explore running at http://127.0.0.1:{PORT}")
    app.run(host="127.0.0.1", port=PORT, debug=False, threaded=True)
