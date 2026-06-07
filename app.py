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
        score = 72
        if not devices:
            score -= 10
        return score

    score   = 100
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
             if audio_data.get(t.get("name", ""), {}).get("low_mid_energy", 0) > 0.25]
    if len(muddy) >= 2:
        problems.append({
            "severity": "high", "title": "Frequency Mud",
            "detail": f"{len(muddy)} tracks ({', '.join(muddy[:3])}) heavy in 250–500 Hz.",
            "fix": "High-pass or shelf cut ~200–350 Hz on non-bass elements."
        })

    low_heavy = [t["name"] for t in tracks
                 if (audio_data.get(t.get("name", ""), {}).get("sub_energy", 0)
                     + audio_data.get(t.get("name", ""), {}).get("bass_energy", 0)) > 0.45]
    if len(low_heavy) >= 2:
        problems.append({
            "severity": "high", "title": "Low-End Conflict",
            "detail": f"{len(low_heavy)} tracks competing in sub/bass region.",
            "fix": "Side-chain or high-pass everything except kick and bass below 80 Hz."
        })

    clipping = [t["name"] for t in tracks
                if audio_data.get(t.get("name", ""), {}).get("peak_db", -99) > -1]
    if clipping:
        problems.append({
            "severity": "high", "title": "Clipping",
            "detail": f"Near or above 0 dBFS: {', '.join(clipping)}.",
            "fix": "Lower gain or add a limiter on these tracks."
        })

    narrow = [t["name"] for t in tracks
              if not t.get("is_midi_track")
              and audio_data.get(t.get("name", ""), {}).get("stereo_width", 1) < 0.05]
    if len(narrow) >= 3:
        problems.append({
            "severity": "medium", "title": "Narrow Mix",
            "detail": f"{len(narrow)} audio tracks appear mostly mono.",
            "fix": "Add subtle stereo widening on pads or ambience."
        })

    if not problems:
        problems.append({
            "severity": "none", "title": "Clean Scan",
            "detail": "No critical issues detected.", "fix": ""
        })

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

    problems      = detect_problems(tracks, audio_data)
    track_scores  = {t.get("name", ""): compute_track_health(t, audio_data) for t in tracks}
    overall_health = compute_overall_health(tracks, audio_data, problems)

    track_lines = []
    for t in tracks:
        kind  = "MIDI" if t.get("is_midi_track") else "Audio"
        devs  = ", ".join(x["name"] for x in t.get("devices", []) if x.get("name"))
        adat  = audio_data.get(t.get("name", ""), {})
        aline = ""
        if adat and "peak_db" in adat:
            aline = (
                f" [peak:{adat['peak_db']}dB rms:{adat['rms_db']}dB"
                f" lufs:{adat.get('lufs','?')} width:{adat.get('stereo_width','?')}"
                f" sub:{adat.get('sub_energy','?')} bass:{adat.get('bass_energy','?')}"
                f" low_mid:{adat.get('low_mid_energy','?')} mid:{adat.get('mid_energy','?')}]"
            )
        track_lines.append(
            f"  [{kind}] {t.get('name','?')}{': ' + devs if devs else ''}{aline}"
        )

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
  --teal:#00A898;--purple:#7B2FD4;--gold:#C8A843;
  --red:#C84030;--green:#28B060;--orange:#C87030;
  --teal-dim:rgba(0,168,152,.12);--red-dim:rgba(200,64,48,.12);
  --green-dim:rgba(40,176,96,.12);--gold-dim:rgba(200,168,67,.12);
}
body{background:var(--bg);color:var(--text);font-family:'Inter',system-ui,sans-serif;font-size:12px;min-height:100vh;display:flex;flex-direction:column;}

/* ── Header ── */
.hdr{display:flex;align-items:center;gap:14px;padding:0 18px;height:52px;border-bottom:1px solid var(--border);background:rgba(3,5,7,.97);flex-shrink:0;position:sticky;top:0;z-index:10;}
.brand{font-size:15px;font-weight:900;letter-spacing:5px;color:var(--teal);}
.brand-sub{font-size:8px;font-weight:700;letter-spacing:.22em;color:var(--dim);text-transform:uppercase;margin-top:1px;}
.status-dot{width:7px;height:7px;border-radius:50%;background:var(--red);flex-shrink:0;transition:background .3s;}
.status-dot.on{background:var(--green);}

/* Overall health in header */
.hdr-health{display:flex;align-items:center;gap:10px;padding:6px 14px;background:var(--panel2);border:1px solid var(--border2);border-radius:8px;margin-left:4px;}
.hdr-health-score{font-size:20px;font-weight:900;line-height:1;}
.hdr-health-label{font-size:8px;font-weight:800;letter-spacing:.2em;text-transform:uppercase;color:var(--dim);}
.hdr-health-bar{width:80px;height:4px;background:var(--border2);border-radius:2px;overflow:hidden;margin-top:3px;}
.hdr-health-fill{height:100%;border-radius:2px;transition:width .5s ease;}

.hdr-right{margin-left:auto;display:flex;gap:6px;align-items:center;flex-wrap:wrap;}
.pill{padding:5px 10px;border-radius:20px;font-size:9px;font-weight:800;letter-spacing:.1em;text-transform:uppercase;border:1px solid var(--border2);background:transparent;color:var(--dim);cursor:pointer;transition:all .15s;white-space:nowrap;}
.pill:hover{border-color:var(--teal);color:var(--teal);}
.pill.primary{background:var(--teal);color:#000;border-color:var(--teal);}
.pill.primary:hover{opacity:.85;}

/* ── Layout ── */
.main{display:flex;flex:1;overflow:hidden;height:calc(100vh - 52px);}

/* ── Sidebar (tracks) ── */
.sidebar{width:210px;border-right:1px solid var(--border);display:flex;flex-direction:column;flex-shrink:0;}
.panel-hdr{padding:7px 12px;font-size:8px;font-weight:800;letter-spacing:.2em;color:var(--dim);text-transform:uppercase;border-bottom:1px solid var(--border);display:flex;align-items:center;justify-content:space-between;flex-shrink:0;}
.track-list{flex:1;overflow-y:auto;}
.track-row{padding:6px 10px;border-bottom:1px solid var(--border);display:flex;align-items:center;gap:7px;cursor:pointer;transition:background .1s;user-select:none;}
.track-row:hover{background:var(--panel2);}
.track-row.selected{background:var(--teal-dim);border-left:2px solid var(--teal);}
.track-num{font-size:9px;color:var(--dim);width:14px;flex-shrink:0;font-weight:700;font-variant-numeric:tabular-nums;}
.track-name{flex:1;font-size:10.5px;font-weight:600;overflow:hidden;white-space:nowrap;text-overflow:ellipsis;}
.track-type{font-size:7.5px;padding:1px 4px;border-radius:2px;font-weight:800;flex-shrink:0;}
.type-midi{background:rgba(0,168,152,.15);color:var(--teal);}
.type-audio{background:rgba(123,47,212,.15);color:var(--purple);}
.track-score{font-size:10px;font-weight:900;width:24px;text-align:right;flex-shrink:0;font-variant-numeric:tabular-nums;}

/* ── Center ── */
.center{flex:1;display:flex;flex-direction:column;overflow:hidden;border-right:1px solid var(--border);}

/* Problems strip */
.problems-strip{flex-shrink:0;border-bottom:1px solid var(--border);overflow-y:auto;max-height:140px;}
.prob-row{display:flex;align-items:flex-start;gap:10px;padding:8px 14px;border-bottom:1px solid var(--border);font-size:11px;}
.prob-row:last-child{border-bottom:none;}
.prob-dot{width:6px;height:6px;border-radius:50%;flex-shrink:0;margin-top:4px;}
.prob-dot.high{background:var(--red);}
.prob-dot.medium{background:var(--orange);}
.prob-dot.low{background:var(--gold);}
.prob-dot.none{background:var(--green);}
.prob-body{flex:1;}
.prob-title-sm{font-weight:700;font-size:11px;line-height:1.4;}
.prob-detail-sm{color:var(--dim);font-size:10px;line-height:1.5;margin-top:1px;}
.prob-fix-sm{color:var(--teal);font-size:10px;line-height:1.5;margin-top:2px;}

/* Chat area */
.chat-area{flex:1;overflow-y:auto;padding:16px 18px;display:flex;flex-direction:column;gap:12px;}
.msg{display:flex;flex-direction:column;gap:4px;max-width:88%;}
.msg.user{align-self:flex-end;}
.msg.ai{align-self:flex-start;}
.msg-bubble{
  padding:11px 15px;border-radius:12px;
  font-family:'Inter',system-ui,sans-serif;
  font-size:13px;font-weight:400;
  line-height:1.75;
  letter-spacing:.01em;
}
.msg.user .msg-bubble{background:var(--teal-dim);border:1px solid rgba(0,168,152,.25);color:var(--text);}
.msg.ai .msg-bubble{background:var(--panel2);border:1px solid var(--border2);color:var(--text);}
.msg-meta{font-size:9px;color:var(--dim2);padding:0 4px;font-variant-numeric:tabular-nums;}

/* Input row */
.input-row{padding:10px 14px;border-top:1px solid var(--border);display:flex;gap:7px;flex-shrink:0;background:var(--panel);}
.chat-input{flex:1;background:var(--panel2);border:1px solid var(--border2);border-radius:8px;padding:9px 13px;color:var(--text);font-size:12.5px;font-family:'Inter',system-ui,sans-serif;font-weight:400;resize:none;height:40px;transition:border-color .15s;line-height:1.5;}
.chat-input:focus{outline:none;border-color:var(--teal);}
.send-btn{padding:0 16px;border-radius:8px;background:var(--teal);border:none;color:#000;font-size:10px;font-weight:800;letter-spacing:.1em;cursor:pointer;font-family:'Inter',system-ui,sans-serif;transition:opacity .15s;}
.send-btn:hover{opacity:.85;}
.send-btn:disabled{opacity:.35;cursor:not-allowed;}

/* ── Right panel ── */
.right{width:265px;display:flex;flex-direction:column;flex-shrink:0;overflow:hidden;}

/* Device inspector */
.inspector{flex:1;overflow-y:auto;border-bottom:1px solid var(--border);}
.inspector-empty{display:flex;flex-direction:column;align-items:center;justify-content:center;height:100%;color:var(--dim2);gap:8px;text-align:center;padding:20px;font-size:11px;}
.inspector-empty-icon{font-size:24px;opacity:.25;}

/* Freq map */
.freq-section{flex-shrink:0;border-bottom:1px solid var(--border);padding:10px 12px;}
.freq-title{font-size:8px;font-weight:800;letter-spacing:.2em;color:var(--dim);text-transform:uppercase;margin-bottom:8px;}
.freq-band{display:flex;align-items:center;gap:7px;margin-bottom:5px;}
.freq-band:last-child{margin-bottom:0;}
.band-label{font-size:8px;font-weight:700;color:var(--dim);width:30px;flex-shrink:0;letter-spacing:.05em;}
.band-bar-wrap{flex:1;height:8px;background:var(--panel2);border-radius:3px;overflow:hidden;border:1px solid var(--border);}
.band-bar{height:100%;border-radius:3px;transition:width .4s ease;}
.band-pct{font-size:8px;font-weight:700;width:26px;text-align:right;flex-shrink:0;font-variant-numeric:tabular-nums;}

/* Device list */
.device-block{border-bottom:1px solid var(--border);padding:10px 12px;}
.device-name-row{display:flex;align-items:center;gap:6px;margin-bottom:6px;}
.device-name{font-size:11px;font-weight:700;color:var(--text);}
.device-type-tag{font-size:7.5px;font-weight:800;letter-spacing:.1em;text-transform:uppercase;padding:1px 5px;border-radius:2px;background:rgba(0,168,152,.1);color:var(--teal);}
.param-grid{display:grid;grid-template-columns:1fr 1fr;gap:3px;}
.param-row{display:flex;justify-content:space-between;padding:2px 0;}
.param-name{font-size:9px;color:var(--dim);overflow:hidden;white-space:nowrap;text-overflow:ellipsis;max-width:80px;}
.param-val{font-size:9px;font-weight:700;color:var(--text);font-variant-numeric:tabular-nums;}

/* Audio scan box */
.scan-box{flex-shrink:0;padding:10px 12px;}
.scan-label{font-size:8px;font-weight:800;letter-spacing:.15em;color:var(--dim);text-transform:uppercase;margin-bottom:6px;display:block;}
.path-input{width:100%;background:var(--panel2);border:1px solid var(--border2);border-radius:5px;padding:6px 8px;color:var(--text);font-size:10px;font-family:'Inter',system-ui,sans-serif;margin-bottom:6px;}
.path-input:focus{outline:none;border-color:var(--teal);}
.scan-btn{width:100%;padding:7px;border-radius:5px;background:var(--purple);border:none;color:#fff;font-size:9.5px;font-weight:800;letter-spacing:.1em;cursor:pointer;font-family:'Inter',system-ui,sans-serif;text-transform:uppercase;transition:opacity .15s;}
.scan-btn:hover{opacity:.85;}
.scan-btn:disabled{opacity:.35;}

/* Track health detail */
.track-health-banner{padding:10px 12px;border-bottom:1px solid var(--border);}
.th-name{font-size:12px;font-weight:800;color:var(--text);margin-bottom:2px;overflow:hidden;white-space:nowrap;text-overflow:ellipsis;}
.th-meta{font-size:9px;color:var(--dim);margin-bottom:8px;}
.th-score-row{display:flex;align-items:center;gap:10px;}
.th-score{font-size:28px;font-weight:900;line-height:1;}
.th-score-label{font-size:8px;font-weight:800;letter-spacing:.2em;text-transform:uppercase;color:var(--dim);}
.th-bar-wrap{flex:1;height:5px;background:var(--border2);border-radius:3px;overflow:hidden;}
.th-bar-fill{height:100%;border-radius:3px;transition:width .5s ease;}

/* Misc */
.empty{display:flex;flex-direction:column;align-items:center;justify-content:center;height:100%;color:var(--dim2);gap:8px;padding:20px;text-align:center;font-size:11px;}
.empty-icon{font-size:26px;opacity:.25;}
::-webkit-scrollbar{width:3px;}
::-webkit-scrollbar-track{background:transparent;}
::-webkit-scrollbar-thumb{background:var(--border2);border-radius:2px;}
.toast{position:fixed;bottom:16px;left:50%;transform:translateX(-50%);background:var(--teal);color:#000;padding:8px 18px;border-radius:6px;font-size:11px;font-weight:800;opacity:0;transition:opacity .2s;pointer-events:none;z-index:999;white-space:nowrap;}
.toast.show{opacity:1;}
.loading{color:var(--dim);font-style:italic;}
</style>
</head>
<body>

<div class="hdr">
  <div>
    <div class="brand">EXPLORE</div>
    <div class="brand-sub">AI Mix Engineer</div>
  </div>
  <div id="ableton-dot" class="status-dot" title="Ableton Live"></div>

  <div class="hdr-health" id="hdr-health" style="display:none">
    <div>
      <div id="hdr-health-score" class="hdr-health-score">—</div>
      <div id="hdr-health-lbl" class="hdr-health-label">Mix Health</div>
      <div class="hdr-health-bar"><div id="hdr-health-fill" class="hdr-health-fill" style="width:0%"></div></div>
    </div>
  </div>

  <div class="hdr-right">
    <button class="pill primary" onclick="runScan()">⟳ Scan</button>
    <button class="pill" onclick="quickPrompt('Why does this mix sound muddy?')">Mud?</button>
    <button class="pill" onclick="quickPrompt('What is taking up the most space in the mix?')">Space?</button>
    <button class="pill" onclick="quickPrompt('What should I work on first?')">Priority?</button>
    <button class="pill" onclick="quickPrompt('Why does the chorus feel weak?')">Chorus?</button>
    <button class="pill" onclick="quickPrompt('How is the low end balance?')">Low End?</button>
    <button class="pill" onclick="quickPrompt('Where is the vocal sitting in the mix?')">Vocal?</button>
    <button class="pill" onclick="loadArrangement()">Arrangement</button>
    <button class="pill" onclick="loadCues()">Cue Points</button>
  </div>
</div>

<div class="main">

  <!-- Sidebar: track list -->
  <div class="sidebar">
    <div class="panel-hdr">
      Tracks
      <span id="track-count-lbl" style="color:var(--text);font-size:10px;font-weight:900;letter-spacing:0"></span>
    </div>
    <div class="track-list" id="track-list">
      <div class="empty"><div class="empty-icon">🎚</div><div>Scan to load tracks</div></div>
    </div>
  </div>

  <!-- Center: problems + chat -->
  <div class="center">
    <div class="problems-strip" id="problems-strip">
      <div class="prob-row">
        <div class="prob-dot none"></div>
        <div class="prob-body"><div class="prob-detail-sm">Run a scan to detect mix problems</div></div>
      </div>
    </div>
    <div class="chat-area" id="chat-area">
      <div class="msg ai">
        <div class="msg-bubble">I'm Explore — your AI mix engineer.<br><br>Hit <strong>Scan</strong> to load your Ableton session, then ask me anything about your mix. Point me at your project folder (right panel) for deep audio analysis including clipping, frequency energy, and stereo width.<br><br>Try: "What should I work on first?" or "Why does this sound muddy?"</div>
      </div>
    </div>
    <div class="input-row">
      <textarea class="chat-input" id="chat-input" placeholder="Ask about your mix..."></textarea>
      <button class="send-btn" id="send-btn" onclick="sendMessage()">Ask</button>
    </div>
  </div>

  <!-- Right panel: inspector + freq map + scan -->
  <div class="right">
    <div class="panel-hdr">Inspector</div>
    <div class="inspector" id="inspector">
      <div class="inspector-empty">
        <div class="inspector-empty-icon">🔍</div>
        <div>Click a track to inspect devices and frequency energy</div>
      </div>
    </div>
    <div class="freq-section" id="freq-section" style="display:none">
      <div class="freq-title">Frequency Map</div>
      <div id="freq-map"></div>
    </div>
    <div class="scan-box">
      <label class="scan-label">Project Folder</label>
      <input class="path-input" id="project-path" placeholder="/Users/you/Music/MyProject" type="text">
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

// ── Helpers ────────────────────────────────────────────────────────────────────
function healthColor(score) {
  if (score >= 80) return 'var(--green)';
  if (score >= 60) return 'var(--gold)';
  return 'var(--red)';
}

function healthLabel(score) {
  if (score >= 85) return 'EXCELLENT';
  if (score >= 70) return 'GOOD';
  if (score >= 55) return 'FAIR';
  if (score >= 40) return 'WEAK';
  return 'CRITICAL';
}

function esc(s) {
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

function renderText(s) {
  return esc(s).replace(/\n/g,'<br>');
}

function bandColor(pct, band) {
  if (band === 'LMID') {
    if (pct > 25) return 'var(--red)';
    if (pct > 18) return 'var(--orange)';
    return 'var(--teal)';
  }
  if (band === 'SUB') {
    if (pct > 25) return 'var(--orange)';
    return 'var(--teal)';
  }
  if (band === 'BASS') {
    if (pct > 35) return 'var(--orange)';
    return 'var(--green)';
  }
  return 'var(--teal)';
}

// ── Scan ───────────────────────────────────────────────────────────────────────
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
    renderTracks(allTracks);
    renderProblems(d.problems || []);
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

// ── Render functions ───────────────────────────────────────────────────────────
function renderOverallHealth(score, tracks, problems) {
  var el = document.getElementById('hdr-health');
  el.style.display = 'flex';
  var col = healthColor(score);
  var lbl = healthLabel(score);
  document.getElementById('hdr-health-score').textContent = score;
  document.getElementById('hdr-health-score').style.color = col;
  document.getElementById('hdr-health-lbl').textContent = lbl;
  var fill = document.getElementById('hdr-health-fill');
  fill.style.width = score + '%';
  fill.style.background = col;
}

function renderTracks(tracks) {
  document.getElementById('track-count-lbl').textContent = tracks.length;
  var el = document.getElementById('track-list');
  if (!tracks.length) {
    el.innerHTML = '<div class="empty"><div>No tracks found</div></div>';
    return;
  }
  var html = '';
  for (var i = 0; i < tracks.length; i++) {
    var t = tracks[i];
    var name = t.name || ('Track ' + (i+1));
    var isMidi = t.is_midi_track;
    var badge = isMidi
      ? '<span class="track-type type-midi">M</span>'
      : '<span class="track-type type-audio">A</span>';
    var score = trackScores[name];
    var scoreHtml = '';
    if (score !== undefined) {
      scoreHtml = '<span class="track-score" style="color:' + healthColor(score) + '">' + score + '</span>';
    }
    html += '<div class="track-row" id="trow-' + i + '" onclick="selectTrack(' + i + ')">'
      + '<span class="track-num">' + (i+1) + '</span>'
      + '<span class="track-name" title="' + esc(name) + '">' + esc(name) + '</span>'
      + badge + scoreHtml
      + '</div>';
  }
  el.innerHTML = html;
}

function renderProblems(problems) {
  var el = document.getElementById('problems-strip');
  if (!problems.length) {
    el.innerHTML = '<div class="prob-row"><div class="prob-dot none"></div><div class="prob-body"><div class="prob-detail-sm">No issues detected</div></div></div>';
    return;
  }
  var html = '';
  for (var i = 0; i < problems.length; i++) {
    var p = problems[i];
    html += '<div class="prob-row">'
      + '<div class="prob-dot ' + esc(p.severity) + '"></div>'
      + '<div class="prob-body">'
      + '<div class="prob-title-sm">' + esc(p.title) + '</div>'
      + '<div class="prob-detail-sm">' + esc(p.detail) + '</div>'
      + (p.fix ? '<div class="prob-fix-sm">→ ' + esc(p.fix) + '</div>' : '')
      + '</div></div>';
  }
  el.innerHTML = html;
}

function renderFreqMap(adat) {
  var section = document.getElementById('freq-section');
  var el = document.getElementById('freq-map');
  if (!adat || adat.peak_db === undefined) {
    section.style.display = 'none';
    return;
  }
  section.style.display = 'block';
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
    var pct = Math.round(raw * 100);
    var col = bandColor(pct, b.label);
    // bar width: scale so 40% = full width (typical band won't exceed 40%)
    var barW = Math.min(100, Math.round(pct * 2.5));
    html += '<div class="freq-band">'
      + '<span class="band-label">' + b.label + '</span>'
      + '<div class="band-bar-wrap"><div class="band-bar" style="width:' + barW + '%;background:' + col + '"></div></div>'
      + '<span class="band-pct" style="color:' + col + '">' + pct + '%</span>'
      + '</div>';
  }
  el.innerHTML = html;
}

// ── Track selection + Device Inspector ────────────────────────────────────────
async function selectTrack(i) {
  selectedTrackIdx = i;
  document.querySelectorAll('.track-row').forEach(function(r, j) {
    r.classList.toggle('selected', i === j);
  });

  var t = allTracks[i];
  if (!t) return;

  var name = t.name || ('Track ' + (i+1));
  var isMidi = t.is_midi_track;
  var score = trackScores[name];
  var adat = allAudioData[name] || {};

  // Track health banner
  var col = score !== undefined ? healthColor(score) : 'var(--dim)';
  var lbl = score !== undefined ? healthLabel(score) : '';
  var scoreNum = score !== undefined ? score : '—';
  var barW = score !== undefined ? score : 0;

  var devices = t.devices || [];
  var devCount = devices.length;
  var metaStr = (isMidi ? 'MIDI' : 'Audio') + ' · ' + devCount + ' device' + (devCount !== 1 ? 's' : '');
  if (adat.peak_db !== undefined) {
    metaStr += ' · peak ' + adat.peak_db + 'dB';
    if (adat.lufs !== undefined && adat.lufs !== null) metaStr += ' · ' + adat.lufs + ' LUFS';
  }

  var inspHtml = '<div class="track-health-banner">'
    + '<div class="th-name">' + esc(name) + '</div>'
    + '<div class="th-meta">' + esc(metaStr) + '</div>'
    + '<div class="th-score-row">'
    + '<div><div class="th-score" style="color:' + col + '">' + scoreNum + '</div>'
    + '<div class="th-score-label">' + esc(lbl) + '</div></div>'
    + '<div class="th-bar-wrap"><div class="th-bar-fill" style="width:' + barW + '%;background:' + col + '"></div></div>'
    + '</div></div>';

  // Devices from track data
  if (devices.length) {
    for (var d = 0; d < devices.length; d++) {
      var dev = devices[d];
      if (dev && dev.name) {
        inspHtml += '<div class="device-block">'
          + '<div class="device-name-row">'
          + '<span class="device-name">' + esc(dev.name) + '</span>'
          + '</div></div>';
      }
    }
  } else {
    inspHtml += '<div style="padding:10px 12px;font-size:10px;color:var(--dim)">No devices on this track</div>';
  }

  inspHtml += '<div id="device-params-area" style="padding:8px 12px;font-size:9px;color:var(--dim)">Loading device parameters...</div>';

  document.getElementById('inspector').innerHTML = inspHtml;

  // Freq map
  renderFreqMap(adat);

  // Fetch device parameters async
  loadDeviceParams(i);
}

async function loadDeviceParams(trackIdx) {
  try {
    var r = await fetch('/api/devices/' + trackIdx);
    var d = await r.json();
    var el = document.getElementById('device-params-area');
    if (!el) return;

    if (d.error) {
      el.textContent = 'No parameter data: ' + d.error;
      return;
    }

    // d may be an array of device objects or {devices: [...]}
    var devs = Array.isArray(d) ? d : (d.devices || []);
    if (!devs.length) {
      el.textContent = 'No parameter data available.';
      return;
    }

    var html = '';
    for (var i = 0; i < devs.length; i++) {
      var dev = devs[i];
      var params = dev.parameters || [];
      if (!params.length) continue;
      html += '<div style="margin-bottom:10px;">'
        + '<div style="font-size:9px;font-weight:800;color:var(--text);margin-bottom:5px;letter-spacing:.05em;">'
        + esc(dev.name || ('Device ' + (i+1))) + '</div>';
      // Show up to 10 key parameters
      var shown = params.slice(0, 10);
      for (var j = 0; j < shown.length; j++) {
        var p = shown[j];
        var val = p.value !== undefined ? String(p.value).substring(0, 10) : '—';
        html += '<div style="display:flex;justify-content:space-between;padding:2px 0;border-bottom:1px solid var(--border);">'
          + '<span style="font-size:9px;color:var(--dim);overflow:hidden;white-space:nowrap;text-overflow:ellipsis;max-width:120px">' + esc(p.name || '') + '</span>'
          + '<span style="font-size:9px;font-weight:700;color:var(--text)">' + esc(val) + '</span>'
          + '</div>';
      }
      if (params.length > 10) {
        html += '<div style="font-size:8px;color:var(--dim2);margin-top:3px">+ ' + (params.length - 10) + ' more</div>';
      }
      html += '</div>';
    }
    el.innerHTML = html || '<span style="color:var(--dim2)">No parameters found.</span>';
  } catch(e) {
    var el2 = document.getElementById('device-params-area');
    if (el2) el2.textContent = 'Could not load parameters.';
  }
}

// ── Arrangement / Cues ────────────────────────────────────────────────────────
async function loadArrangement() {
  toast('Loading arrangement...');
  try {
    var r = await fetch('/api/arrangement');
    var d = await r.json();
    if (d.error) { toast('Arrangement: ' + d.error, true); return; }
    var info = JSON.stringify(d, null, 2);
    addMessage('ai', 'Arrangement info:\n\n' + info.substring(0, 800));
    toast('Arrangement loaded');
  } catch(e) {
    toast('Could not load arrangement', true);
  }
}

async function loadCues() {
  toast('Loading cue points...');
  try {
    var r = await fetch('/api/cues');
    var d = await r.json();
    if (d.error) { toast('Cues: ' + d.error, true); return; }
    var cues = Array.isArray(d) ? d : (d.cue_points || d.cues || []);
    if (!cues.length) { toast('No cue points found'); return; }
    var lines = cues.map(function(c) {
      return (c.name || 'Cue') + ' @ ' + (c.time !== undefined ? c.time.toFixed(2) : '?') + 's';
    });
    addMessage('ai', 'Cue points:\n\n' + lines.join('\n'));
    toast(cues.length + ' cue points loaded');
  } catch(e) {
    toast('Could not load cues', true);
  }
}

// ── Chat ───────────────────────────────────────────────────────────────────────
function quickPrompt(p) {
  document.getElementById('chat-input').value = p;
  sendMessage();
}

async function sendMessage() {
  var input = document.getElementById('chat-input');
  var prompt = input.value.trim();
  if (!prompt) return;
  input.value = '';
  addMessage('user', prompt);
  var btn = document.getElementById('send-btn');
  btn.disabled = true;
  var thinking = addThinking();
  try {
    var r = await fetch('/api/ask', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({prompt: prompt, session_ctx: sessionCtx, problem_ctx: problemCtx})
    });
    var d = await r.json();
    var bubble = thinking.querySelector('.msg-bubble');
    bubble.innerHTML = renderText(d.text || 'No response.');
    thinking.querySelector('.msg-meta').textContent = (d.tokens || '') + ' tok';
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

// ── Input key handling ────────────────────────────────────────────────────────
document.getElementById('chat-input').addEventListener('keydown', function(e) {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
});

// ── Toast ──────────────────────────────────────────────────────────────────────
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

// ── Auto-check Ableton on load ────────────────────────────────────────────────
fetch('/api/session').then(function(r) { return r.json(); }).then(function(d) {
  if (d.tracks && d.tracks.length) {
    document.getElementById('ableton-dot').classList.add('on');
  }
}).catch(function() {});
</script>
</body>
</html>"""

if __name__ == "__main__":
    print(f"Explore running at http://127.0.0.1:{PORT}")
    app.run(host="127.0.0.1", port=PORT, debug=False, threaded=True)
