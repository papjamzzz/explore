import os, time, socket, json, tempfile
from pathlib import Path
from flask import Flask, request, jsonify, render_template_string
from dotenv import load_dotenv
import anthropic
import numpy as np
import librosa
from collections import Counter

load_dotenv()
app = Flask(__name__)
PORT     = int(os.getenv("PORT", 5572))
DATA_FILE = Path(__file__).parent / "data" / "state.json"

# ── Server-side persistence ────────────────────────────────────────────────────

def load_server_state():
    try:
        if DATA_FILE.exists():
            return json.loads(DATA_FILE.read_text())
    except Exception:
        pass
    return {}

def save_server_state(data):
    try:
        DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
        # Trim chat to last 500 messages
        if "chat_history" in data:
            data["chat_history"] = data["chat_history"][-500:]
        DATA_FILE.write_text(json.dumps(data, indent=2))
    except Exception:
        pass

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
    raw_session = ableton_send("get_session_info")
    # AbletonMCP wraps responses: {"status":"success","result":{...}}
    session = raw_session.get("result", raw_session) if isinstance(raw_session, dict) else {}
    count = session.get("track_count", 0) if isinstance(session, dict) else 0
    tracks = []
    for i in range(min(count, 24)):
        raw_t = ableton_send("get_track_info", {"track_index": i})
        t = raw_t.get("result", raw_t) if isinstance(raw_t, dict) else raw_t
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
            model="claude-sonnet-4-6",
            max_tokens=2048,
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

@app.route("/api/state", methods=["GET"])
def api_state_get():
    return jsonify(load_server_state())

@app.route("/api/state", methods=["POST"])
def api_state_post():
    d = request.get_json() or {}
    state = load_server_state()
    state.update(d)
    state["saved_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    save_server_state(state)
    return jsonify({"ok": True})

@app.route("/api/gain/run", methods=["POST"])
def api_gain_run():
    d = request.get_json() or {}
    task = (d.get("task") or "").strip()
    if not task:
        return jsonify({"error": "No task provided"}), 400
    state_path = Path.home() / ".streamfader" / "state.json"
    try:
        state = {}
        if state_path.exists():
            state = json.loads(state_path.read_text())
        mode      = state.get("mode", "")
        intensity = float(state.get("intensity", 0.5))
        room      = float(state.get("room", 0.5))
        t1_on     = state.get("t1_on", True)
        if not t1_on:
            return jsonify({"output": "[Track 1 is muted — unmute to run]", "mode": mode})
        if mode == "BUILD":
            system = ("BUILD mode: execute immediately. One approach only, no alternatives. "
                      "Intensity " + str(round(intensity, 2)) + ". Verbosity " + str(round(room, 2)) + ". "
                      "No preamble. Start doing it.")
        elif mode == "EXPLORE":
            system = ("EXPLORE mode: think broadly, surface multiple angles and tradeoffs, ask open questions. "
                      "Intensity " + str(round(intensity, 2)) + ". Verbosity " + str(round(room, 2)) + ". "
                      "Be thorough.")
        else:
            system = ("Helpful AI assistant. "
                      "Intensity " + str(round(intensity, 2)) + ". Verbosity " + str(round(room, 2)) + ".")
        client = anthropic.Anthropic()
        resp = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=int(150 + room * 900),
            system=system,
            messages=[{"role": "user", "content": task}]
        )
        return jsonify({"output": resp.content[0].text, "mode": mode})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/gain/set", methods=["POST"])
def api_gain_set():
    d = request.get_json() or {}
    state_path = Path.home() / ".streamfader" / "state.json"
    try:
        current = {}
        if state_path.exists():
            current = json.loads(state_path.read_text())
        current.update(d)
        tmp = state_path.parent / (state_path.name + ".tmp")
        tmp.write_text(json.dumps(current, indent=2) + "\n")
        tmp.rename(state_path)
        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/chords", methods=["POST"])
def api_chords():
    f = request.files.get("file")
    if not f:
        return jsonify({"error": "No file uploaded"}), 400

    pitch_classes = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']

    # Krumhansl-Schmuckler key profiles
    major_profile = np.array([6.35,2.23,3.48,2.33,4.38,4.09,2.52,5.19,2.39,3.66,2.29,2.88])
    minor_profile = np.array([6.33,2.68,3.52,5.38,2.60,3.53,2.54,4.75,3.98,2.69,3.34,3.17])

    def make_template(root, intervals):
        t = np.zeros(12)
        for iv in intervals:
            t[(root + iv) % 12] = 1.0
        return t / t.sum()

    templates = {}
    for i, pc in enumerate(pitch_classes):
        templates[pc]        = make_template(i, [0,4,7])
        templates[pc+'m']    = make_template(i, [0,3,7])
        templates[pc+'7']    = make_template(i, [0,4,7,10])
        templates[pc+'m7']   = make_template(i, [0,3,7,10])
        templates[pc+'maj7'] = make_template(i, [0,4,7,11])
        templates[pc+'sus2'] = make_template(i, [0,2,7])
        templates[pc+'sus4'] = make_template(i, [0,5,7])

    suffix = Path(f.filename).suffix or '.wav'
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        f.save(tmp.name)
        tmp_path = tmp.name

    try:
        y, sr = librosa.load(tmp_path, mono=True, duration=300)
        hop = 2048  # ~0.09s per frame at 22050Hz — good resolution without explosion
        chroma = librosa.feature.chroma_cens(y=y, sr=sr, hop_length=hop)

        # Detect key
        chroma_mean = chroma.mean(axis=1)
        best_key, best_mode, best_corr = 'C', 'major', -999
        for i, pc in enumerate(pitch_classes):
            for mode, profile in [('major', major_profile), ('minor', minor_profile)]:
                corr = float(np.corrcoef(np.roll(profile, i), chroma_mean)[0, 1])
                if corr > best_corr:
                    best_corr, best_key, best_mode = corr, pc, mode

        # Detect chord per frame
        times = librosa.frames_to_time(np.arange(chroma.shape[1]), sr=sr, hop_length=hop)
        frame_chords = []
        for fi in range(chroma.shape[1]):
            frame = chroma[:, fi]
            s = frame.sum()
            fn = frame / (s + 1e-9)
            best_c, best_s = 'N', -1
            for cn, tmpl in templates.items():
                sc = float(np.dot(fn, tmpl))
                if sc > best_s:
                    best_s, best_c = sc, cn
            frame_chords.append(best_c)

        # Smooth with majority-vote window (~1s)
        win = max(4, int(sr / hop))
        smoothed = []
        for i in range(len(frame_chords)):
            sl = frame_chords[max(0, i-win//2): i+win//2+1]
            smoothed.append(Counter(sl).most_common(1)[0][0])

        # Segment into chord changes
        segments, cur_chord, cur_start = [], smoothed[0], 0
        for i, ch in enumerate(smoothed[1:], 1):
            if ch != cur_chord:
                dur = float(times[i-1]) - float(times[cur_start])
                if dur >= 0.8:
                    segments.append({"chord": cur_chord,
                                     "start": round(float(times[cur_start]), 1),
                                     "end":   round(float(times[i-1]), 1),
                                     "dur":   round(dur, 1)})
                cur_chord, cur_start = ch, i
        dur = float(times[-1]) - float(times[cur_start])
        if dur >= 0.8:
            segments.append({"chord": cur_chord,
                             "start": round(float(times[cur_start]), 1),
                             "end":   round(float(times[-1]), 1),
                             "dur":   round(dur, 1)})

        # Unique chord list in order
        seen, chord_list = set(), []
        for s in segments:
            if s['chord'] not in seen:
                seen.add(s['chord']); chord_list.append(s['chord'])

        return jsonify({
            "key":        best_key + " " + best_mode,
            "root":       best_key,
            "mode":       best_mode,
            "segments":   segments,
            "chord_list": chord_list,
            "duration":   round(float(times[-1]), 1)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        try: os.unlink(tmp_path)
        except: pass

@app.route("/")
def index():
    from flask import Response
    resp = Response(HTML, mimetype='text/html; charset=utf-8')
    resp.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate'
    return resp

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
  --bg:#07050F;--panel:#0C0A18;--panel2:#110E22;--panel3:#161230;
  --border:#1A1535;--border2:#241E48;--border3:#2E2860;
  --text:#E0D8F8;--dim:#5A5080;--dim2:#3A3260;
  --teal:#00B8A8;--teal2:#00E0D0;--purple:#8B3FE4;--gold:#D4B050;
  --red:#D84840;--green:#30C070;--orange:#D88040;--coral:#E85870;
  --teal-dim:rgba(0,184,168,.1);
  --glow-teal:rgba(0,224,208,.18);--glow-purple:rgba(139,63,228,.18);--glow-coral:rgba(232,88,112,.18);
}
body{background:var(--bg);color:var(--text);font-family:'Inter',system-ui,sans-serif;font-size:12px;height:100vh;display:flex;flex-direction:column;overflow:hidden;background-image:radial-gradient(ellipse 80% 50% at 20% 0%,rgba(139,63,228,.06) 0%,transparent 60%),radial-gradient(ellipse 60% 40% at 80% 100%,rgba(0,184,168,.05) 0%,transparent 60%);transition:background .25s,color .25s;}

/* ── Light theme ── */
body.light{
  --bg:#EDF0F8;--panel:#F4F6FC;--panel2:#FFFFFF;--panel3:#EAEEf8;
  --border:#C8D4E8;--border2:#B0C0D8;--border3:#98AABE;
  --text:#0E1422;--dim:#2A3A50;--dim2:#4A5A72;
  --teal:#007868;--teal2:#009888;--purple:#5A18A8;--gold:#906800;
  --red:#B82020;--green:#0F7840;--orange:#984810;--coral:#B82040;
  --teal-dim:rgba(0,120,104,.08);
  --glow-teal:rgba(0,152,136,.12);--glow-purple:rgba(90,24,168,.12);--glow-coral:rgba(184,32,64,.12);
  background-image:radial-gradient(ellipse 80% 50% at 20% 0%,rgba(90,24,168,.03) 0%,transparent 60%),radial-gradient(ellipse 60% 40% at 80% 100%,rgba(0,120,104,.03) 0%,transparent 60%);
  font-size:13px;
}
body.light .hdr{background:rgba(244,246,252,.97);box-shadow:0 1px 8px rgba(0,0,0,.06);}
body.light .track-row:hover{background:#EAEFF8;}
body.light .track-row.selected{background:rgba(0,136,120,.07);border-left-color:var(--teal);}
body.light .type-midi{background:rgba(0,136,120,.1);color:#009888;}
body.light .type-audio{background:rgba(104,32,184,.08);color:#6820B8;}
body.light .msg.user .msg-bubble{background:rgba(0,136,120,.07);border-color:rgba(0,136,120,.18);}
body.light .msg.ai .msg-bubble{background:#FFFFFF;border-color:#D4DCEE;box-shadow:0 1px 6px rgba(0,0,0,.05);}
body.light .chat-input{background:#FFFFFF;border-color:#BCC8E0;color:#1A1E2E;}
body.light .donut-bg{stroke:#DDE5F5;}
body.light .chart-panel{background:linear-gradient(180deg,rgba(255,255,255,.7) 0%,rgba(255,255,255,.3) 100%);}
body.light .stats-strip{background:var(--panel);}
body.light .stat-card{background:var(--panel);}
/* Light mode — size + contrast upgrades */
body.light .panel-hdr{font-size:10px;color:var(--text);letter-spacing:.14em;}
body.light .track-name{font-size:13px;color:var(--text);}
body.light .track-num{font-size:10px;color:var(--dim);}
body.light .track-score{font-size:11px;}
body.light .stat-icon{font-size:11px;color:var(--text);font-weight:900;}
body.light .stat-val{font-size:16px;font-weight:900;}
body.light .stat-sub{font-size:10px;color:var(--dim);}
body.light .stat-label{font-size:11px;color:var(--dim);}
body.light .gb-label{font-size:10px;color:var(--text);}
body.light .gb-val{font-size:11px;color:var(--text);}
body.light .gb-btn{font-size:11px;}
body.light .chart-title{font-size:10px;color:var(--text);}
body.light .donut-name{font-size:10px;color:var(--dim);}
body.light .knob-lbl{font-size:9.5px;color:var(--dim);}
body.light .knob-val{font-size:9.5px;color:var(--text);}
body.light .prob-detail-sm{font-size:11px;color:var(--dim);}
body.light .prob-fix-sm{font-size:11px;}
body.light .prob-title{font-size:12px;font-weight:800;}
body.light .msg-bubble{font-size:12px;color:var(--text);}
body.light .chat-input{font-size:12px;color:var(--text);}
body.light .hdr-health-sub{font-size:9px;color:var(--dim);}
body.light .brand-sub{font-size:9px;color:var(--dim);}
body.light .bk-mode-name{color:var(--teal2);}
body.light .scan-run-btn{background:var(--teal2);}
body.light .bk-station{background:var(--panel);}
body.light .bottom-bar{background:var(--bg);}
body.light .scan-label{font-size:10px;color:var(--dim);}
body.light .th-meta{font-size:10px;color:var(--dim);}
body.light .th-score-lbl{font-size:9px;color:var(--dim);}
body.light .empty{font-size:12px;color:var(--dim);}
body.light .msg-meta{font-size:10px;color:var(--dim2);}
body.light .ableton-status-lbl{font-size:10px;color:var(--dim);}
body.light .sidebar{background:var(--panel);}
body.light .right{background:var(--panel);}
body.light .gain-bridge{background:var(--panel);}
body.light .gain-mode-pill{background:rgba(0,136,120,.12);}
body.light .gain-mode-pill.build{background:rgba(104,32,184,.1);}
/* GB sliders on light */
body.light input[type=range].gb-slider::-webkit-slider-runnable-track{background:#D0DAF0;box-shadow:inset 0 1px 3px rgba(0,0,0,.12),inset 0 -1px 1px rgba(0,0,0,.06);border:1px solid rgba(0,0,0,.09);}
body.light input[type=range].gb-slider.effort::-webkit-slider-thumb{background:linear-gradient(90deg,#D8F2F0 0%,#A8E4E0 22%,#80D8D4 44%,rgba(0,180,168,.3) 50%,#80D8D4 56%,#A8E4E0 78%,#D8F2F0 100%);border:1px solid #009888;box-shadow:0 2px 8px rgba(0,152,136,.2),inset 0 1px 0 rgba(255,255,255,.9),inset 0 -1px 0 rgba(0,0,0,.08);}
body.light input[type=range].gb-slider.effort::-webkit-slider-thumb:hover{border-color:var(--teal2);box-shadow:0 0 8px rgba(0,168,152,.35),0 2px 8px rgba(0,0,0,.1);}
body.light input[type=range].gb-slider.verbosity::-webkit-slider-thumb{background:linear-gradient(90deg,#EEE8F8 0%,#D4C0F0 22%,#BCA0E8 44%,rgba(104,32,184,.25) 50%,#BCA0E8 56%,#D4C0F0 78%,#EEE8F8 100%);border:1px solid #6820B8;box-shadow:0 2px 8px rgba(104,32,184,.2),inset 0 1px 0 rgba(255,255,255,.9),inset 0 -1px 0 rgba(0,0,0,.08);}
body.light input[type=range].gb-slider.verbosity::-webkit-slider-thumb:hover{border-color:#9B6FE0;box-shadow:0 0 8px rgba(104,32,184,.35),0 2px 8px rgba(0,0,0,.1);}
body.light .gb-label.effort{color:#009888;}
body.light .gb-label.verbosity{color:#6820B8;}
body.light .gb-val.effort{color:#009888;}
body.light .gb-val.verbosity{color:#6820B8;}
body.light .gb-btn.build{background:#E8EEF8;color:#4A6890;}
body.light .gb-btn.explore{background:#E8EEF8;color:#4A6890;}
body.light .gb-btn.mute{border-color:#C0CADF;color:#8898B8;}
body.light .gb-slider-wrap{background:radial-gradient(ellipse 7px 100% at 50% 50%,#C8D4E8 0%,transparent 100%);}

/* ── Header ── */
.hdr{display:flex;align-items:center;gap:12px;padding:0 20px;height:64px;border-bottom:1px solid var(--border);background:rgba(7,5,15,.97);flex-shrink:0;z-index:10;transition:background .25s;}
.brand{font-size:15px;font-weight:900;letter-spacing:5px;color:var(--teal);}
.brand-sub{font-size:7.5px;font-weight:700;letter-spacing:.22em;color:var(--dim);text-transform:uppercase;margin-top:1px;}
.status-dot{width:7px;height:7px;border-radius:50%;background:var(--red);flex-shrink:0;transition:background .3s;}
.status-dot.on{background:var(--green);}
.hdr-health{display:flex;align-items:center;gap:9px;padding:5px 12px;background:var(--panel2);border:1px solid var(--border2);border-radius:8px;}
.hdr-health-score{font-size:18px;font-weight:900;line-height:1;}
.hdr-health-sub{font-size:7px;font-weight:800;letter-spacing:.18em;text-transform:uppercase;color:var(--dim);margin-top:2px;}
.hdr-health-bar{width:64px;height:3px;background:var(--border2);border-radius:2px;margin-top:3px;overflow:hidden;}
.hdr-health-fill{height:100%;border-radius:2px;transition:width .5s;}
.hdr-right{margin-left:auto;display:flex;gap:8px;align-items:center;flex-wrap:nowrap;z-index:1;}
.theme-btn{background:transparent;border:1px solid var(--border2);border-radius:8px;color:var(--dim);font-size:15px;padding:3px 8px;cursor:pointer;transition:all .15s;line-height:1.2;flex-shrink:0;}
.theme-btn:hover{color:var(--text);border-color:var(--border3);}
/* ── Bottom Bar + Big Knob (OneKnob style) ── */
.bottom-bar{display:flex;flex-direction:row;flex-shrink:0;border-top:1px solid var(--border);background:var(--bg);overflow:hidden;}
.bk-station{display:flex;flex-direction:column;align-items:center;justify-content:center;padding:16px 20px 12px 16px;gap:10px;flex-shrink:0;border-right:1px solid var(--border);background:var(--panel);}
.bk-wrap{position:relative;width:360px;height:360px;flex-shrink:0;}
.bk-svg{position:absolute;inset:0;width:100%;height:100%;}
.bk-face-outer{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;pointer-events:none;}
.bk-face-shell{position:relative;width:176px;height:176px;border-radius:50%;cursor:pointer;user-select:none;-webkit-user-select:none;pointer-events:all;background:radial-gradient(circle at 38% 32%,#2C3E52 0%,#0D1E2E 50%,#060E18 100%);box-shadow:0 4px 16px rgba(0,0,0,.7),inset 0 2px 0 rgba(255,255,255,.06),inset 0 -2px 0 rgba(0,0,0,.5);}
.bk-knurl{position:absolute;inset:0;border-radius:50%;background:repeating-conic-gradient(rgba(255,255,255,.022) 0deg,transparent 2deg,transparent 12deg,rgba(255,255,255,.022) 14deg);}
.bk-inner-face{position:absolute;inset:18px;border-radius:50%;background:radial-gradient(circle at 38% 32%,#1E2E3E 0%,#080F18 60%,#040A10 100%);box-shadow:inset 0 2px 0 rgba(255,255,255,.04),inset 0 1px 4px rgba(0,0,0,.8);}
.bk-ptr{position:absolute;left:50%;top:50%;width:4px;height:58px;margin-left:-2px;margin-top:-58px;transform-origin:bottom center;transform:rotate(-135deg);border-radius:4px 4px 0 0;background:linear-gradient(to top,rgba(0,200,188,.45),#00C8BE);transition:transform .22s cubic-bezier(.4,0,.2,1);}
.bk-ptr::after{content:'';position:absolute;top:-2px;left:50%;transform:translateX(-50%);width:8px;height:8px;border-radius:50%;background:#00C8BE;box-shadow:0 0 12px 4px rgba(0,200,188,.8);}
.bk-cap{position:absolute;inset:0;margin:auto;width:22px;height:22px;border-radius:50%;background:radial-gradient(circle at 40% 35%,#1A2E40,#060E18);border:1px solid rgba(0,200,188,.3);box-shadow:0 2px 8px rgba(0,0,0,.7);}
.bk-label-row{display:flex;align-items:center;gap:16px;}
.bk-mode-name{font-size:18px;font-weight:900;letter-spacing:.08em;color:var(--teal);text-transform:uppercase;}
.bk-hint{font-size:7px;font-weight:600;letter-spacing:.14em;color:var(--dim2);text-transform:uppercase;}
.scan-run-btn{padding:10px 22px;background:var(--teal);color:#000;border:none;border-radius:7px;font-size:11px;font-weight:900;letter-spacing:.14em;cursor:pointer;font-family:'Inter',system-ui,sans-serif;text-transform:uppercase;transition:opacity .15s,transform .1s;white-space:nowrap;}
.scan-run-btn:hover{opacity:.85;}
.scan-run-btn:active{transform:scale(.96);}
.bottom-chat{flex:1;display:flex;flex-direction:column;padding:16px;gap:8px;justify-content:flex-end;min-width:0;overflow:hidden;}
/* SVG mode labels on the knob ring */
.bk-lbl{font-size:11px;font-weight:700;font-family:'Inter',system-ui,sans-serif;letter-spacing:.05em;fill:rgba(220,235,245,.55);cursor:pointer;}
.bk-lbl:hover{fill:rgba(255,255,255,.9);}
.bk-lbl.active{font-size:12px;font-weight:900;fill:#00C8BE;}
/* LED indicator dots on the arc ring */
.bk-dot{r:4px;fill:rgba(0,180,165,.22);cursor:pointer;transition:fill .15s,r .15s;}
.bk-dot:hover{fill:rgba(0,200,188,.45);}
.bk-dot.active{r:6px;fill:#00C8BE;}
.bk-hit{fill:transparent;cursor:pointer;}

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
.charts-strip{display:flex;flex-shrink:0;border-bottom:1px solid var(--border);height:240px;}
.chart-panel{flex:1;padding:12px 14px;border-right:1px solid var(--border);overflow:hidden;display:flex;flex-direction:column;background:linear-gradient(180deg,rgba(255,255,255,.015) 0%,transparent 100%);}
.chart-panel:last-child{border-right:none;flex:0 0 210px;}
.chart-title{font-size:7.5px;font-weight:800;letter-spacing:.2em;text-transform:uppercase;color:var(--dim);margin-bottom:10px;flex-shrink:0;display:flex;align-items:center;gap:6px;}
.chart-title-track{color:var(--teal2);font-weight:700;letter-spacing:.05em;text-transform:none;font-size:8px;opacity:.85;}
.health-svg-wrap{flex:1;overflow:hidden;position:relative;}
#health-svg{width:100%;height:100%;display:block;}
/* Cylinder freq chart */
#freq-bands{flex:1;overflow:hidden;display:flex;align-items:stretch;}
#freq-bands svg{width:100%;height:100%;}

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
.msg-meta{font-size:8px;color:var(--dim2);padding:1px 3px 0;font-variant-numeric:tabular-nums;text-align:right;opacity:.5;}
.input-row{padding:9px 12px;border-top:1px solid var(--border);display:flex;gap:6px;flex-shrink:0;background:var(--panel);}
.chat-input{flex:1;background:var(--panel2);border:1px solid var(--border2);border-radius:8px;padding:8px 12px;color:var(--text);font-size:12px;font-family:'Inter',system-ui,sans-serif;resize:none;height:38px;line-height:1.5;transition:border-color .15s;}
.chat-input:focus{outline:none;border-color:var(--teal);}
.send-btn{padding:0 14px;border-radius:8px;background:var(--teal);border:none;color:#000;font-size:9.5px;font-weight:800;letter-spacing:.1em;cursor:pointer;font-family:'Inter',system-ui,sans-serif;transition:opacity .15s;}
.send-btn:hover{opacity:.85;}
.send-btn:disabled{opacity:.35;cursor:not-allowed;}

/* ── Gain Bridge (leftmost panel) ── */
.gain-bridge{width:150px;border-right:1px solid var(--border);display:flex;flex-direction:column;flex-shrink:0;background:var(--panel);}
.gb-faders{flex:1;min-height:0;display:flex;border-bottom:1px solid var(--border);overflow:hidden;}
/* Soft divider between fader columns */
.gb-fader{flex:1;min-width:0;display:flex;flex-direction:column;align-items:center;padding:12px 0 5px;border-right:1px solid rgba(16,28,42,.9);overflow:hidden;}
.gb-fader:last-child{border-right:none;}
.gb-label{font-size:8px;font-weight:900;letter-spacing:.1em;text-transform:uppercase;margin-bottom:6px;flex-shrink:0;}
.gb-label.effort{color:#C08030;}
.gb-label.verbosity{color:#8B5520;}
/* Groove track background */
.gb-slider-wrap{flex:1;min-height:0;display:flex;align-items:center;justify-content:center;width:100%;position:relative;overflow:hidden;}
.gb-val{font-size:11px;font-weight:900;font-variant-numeric:tabular-nums;margin-top:4px;flex-shrink:0;height:15px;line-height:15px;}
.gb-val.effort{color:#C08030;}
.gb-val.verbosity{color:#8B5520;}
/* Vertical fader — rotated; JS sets width = wrapper height for full-height fill */
input[type=range].gb-slider{-webkit-appearance:none;appearance:none;transform:rotate(-90deg);height:20px;cursor:pointer;outline:none;background:transparent;margin:0;padding:0;flex-shrink:0;}
/* Deep wood groove channel */
input[type=range].gb-slider::-webkit-slider-runnable-track{height:8px;border-radius:4px;background:linear-gradient(90deg,#1A0A04,#050C14 40%,#1A0A04);box-shadow:inset 0 3px 8px rgba(0,0,0,.98),inset 0 -1px 2px rgba(0,0,0,.6);border:1px solid rgba(0,0,0,.9);}
/* Wood fader cap */
input[type=range].gb-slider::-webkit-slider-thumb{-webkit-appearance:none;appearance:none;width:14px;height:52px;border-radius:5px;margin-top:-22px;box-shadow:0 3px 12px rgba(0,0,0,.9),inset 0 1px 0 rgba(255,255,255,.22),inset 0 -1px 0 rgba(0,0,0,.5);}
input[type=range].gb-slider.effort::-webkit-slider-thumb{background:linear-gradient(90deg,#2A1508 0%,#7A4520 16%,#C07828 34%,#E09840 50%,#C07828 66%,#7A4520 84%,#2A1508 100%);border:1px solid #1A0A04;}
input[type=range].gb-slider.effort::-webkit-slider-thumb:hover{background:linear-gradient(90deg,#351A0A 0%,#9B5A28 16%,#D88A30 34%,#F0AC4A 50%,#D88A30 66%,#9B5A28 84%,#351A0A 100%);box-shadow:0 0 14px rgba(200,120,40,.5),0 3px 12px rgba(0,0,0,.9);}
input[type=range].gb-slider.verbosity::-webkit-slider-thumb{background:linear-gradient(90deg,#1A0A04 0%,#4A2810 16%,#7B4020 34%,#9B5830 50%,#7B4020 66%,#4A2810 84%,#1A0A04 100%);border:1px solid #0A0402;}
input[type=range].gb-slider.verbosity::-webkit-slider-thumb:hover{background:linear-gradient(90deg,#221008 0%,#5A3418 16%,#8B4F28 34%,#AB6840 50%,#8B4F28 66%,#5A3418 84%,#221008 100%);box-shadow:0 0 14px rgba(160,90,40,.5),0 3px 12px rgba(0,0,0,.9);}
/* Firefox */
input[type=range].gb-slider::-moz-range-track{height:8px;border-radius:4px;background:#1A0A04;box-shadow:inset 0 2px 5px rgba(0,0,0,.95);}
input[type=range].gb-slider.effort::-moz-range-thumb{width:14px;height:52px;border-radius:5px;background:#C07828;border:1px solid #1A0A04;}
input[type=range].gb-slider.verbosity::-moz-range-thumb{width:14px;height:52px;border-radius:5px;background:#7B4020;border:1px solid #0A0402;}
/* Buttons — tight */
.gb-btns{display:flex;flex-direction:column;gap:4px;padding:6px 7px 7px;flex-shrink:0;}
.gb-btn{width:100%;padding:10px 0;font-size:9.5px;font-weight:900;letter-spacing:.1em;border-radius:6px;border:none;cursor:pointer;font-family:'Inter',system-ui,sans-serif;text-transform:uppercase;transition:opacity .12s,background .15s,color .15s,box-shadow .15s;}
.gb-btn:hover{opacity:.82;}
.gb-btn:active{opacity:.65;}
.gb-btn.build{background:#0C1622;color:#426080;}
.gb-btn.build.active{background:var(--purple);color:#fff;box-shadow:0 0 16px rgba(123,47,212,.6);}
.gb-btn.explore{background:#0C1622;color:#426080;}
.gb-btn.explore.active{background:var(--teal);color:#000;font-weight:900;box-shadow:0 0 16px rgba(0,168,152,.6);}
.gb-btn.mute{background:transparent;color:var(--dim2);border:1px solid rgba(22,46,64,.7);font-size:8.5px;padding:6px 0;}
.gb-btn.mute.active{background:rgba(200,64,48,.12);color:var(--red);border-color:rgba(200,64,48,.45);}
/* Prompt + output */
.gb-prompt-area{display:flex;flex-direction:column;gap:5px;padding:7px 7px 8px;border-top:1px solid var(--border);flex-shrink:0;}
.gb-prompt{width:100%;box-sizing:border-box;background:#050D1A;border:1px solid rgba(22,46,64,.9);border-radius:6px;color:var(--text);font-size:10.5px;font-family:'Inter',system-ui,sans-serif;line-height:1.4;padding:7px 8px;resize:none;outline:none;min-height:52px;max-height:100px;}
.gb-prompt:focus{border-color:rgba(0,168,152,.45);box-shadow:0 0 0 2px rgba(0,168,152,.08);}
.gb-prompt::placeholder{color:var(--dim2);}
.gb-run-btn{width:100%;padding:9px 0;font-size:9px;font-weight:900;letter-spacing:.12em;text-transform:uppercase;border:none;border-radius:6px;cursor:pointer;font-family:'Inter',system-ui,sans-serif;background:linear-gradient(135deg,#0E2235 0%,#132D46 100%);color:#4A90A4;transition:background .15s,box-shadow .15s;}
.gb-run-btn:hover:not(:disabled){background:linear-gradient(135deg,#1A3A58 0%,#1C4060 100%);color:var(--teal);box-shadow:0 0 12px rgba(0,168,152,.3);}
.gb-run-btn:disabled{opacity:.45;cursor:not-allowed;}
.gb-output{display:none;background:#040B14;border:1px solid rgba(22,46,64,.8);border-radius:6px;padding:7px 8px;font-size:10px;line-height:1.5;color:var(--text);white-space:pre-wrap;word-break:break-word;max-height:160px;overflow-y:auto;}
/* Light mode overrides — prompt area */
body.light .gb-prompt{background:#F4F7FB;border-color:#C8D8E8;color:#1A2A3A;}
body.light .gb-prompt:focus{border-color:#009888;box-shadow:0 0 0 2px rgba(0,152,136,.08);}
body.light .gb-prompt::placeholder{color:#8BA0B0;}
body.light .gb-run-btn{background:linear-gradient(135deg,#E0EEF8 0%,#D4E8F4 100%);color:#2A6080;}
body.light .gb-run-btn:hover:not(:disabled){background:linear-gradient(135deg,#C8E4F4 0%,#B8D8EC 100%);color:#007868;box-shadow:0 0 10px rgba(0,152,136,.2);}
body.light .gb-output{background:#F8FAFB;border-color:#C8D8E8;color:#1A2A3A;}

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
/* ── Chord ID panel ── */
.chord-panel{flex-shrink:0;border-top:1px solid var(--border);}
.chord-drop-zone{margin:8px 10px 0;border:1.5px dashed var(--border2);border-radius:8px;padding:12px 10px;text-align:center;cursor:pointer;transition:border-color .15s,background .15s;}
.chord-drop-zone:hover,.chord-drop-zone.drag-over{border-color:var(--teal);background:var(--teal-dim);}
.chord-drop-lbl{font-size:9.5px;font-weight:700;color:var(--dim);letter-spacing:.06em;}
.chord-drop-lbl span{color:var(--teal);font-weight:800;}
.chord-key-row{display:flex;align-items:baseline;gap:8px;padding:10px 12px 4px;}
.chord-key-big{font-size:22px;font-weight:900;color:var(--text);letter-spacing:-.01em;}
.chord-key-mode{font-size:10px;font-weight:700;color:var(--dim);text-transform:uppercase;letter-spacing:.1em;}
.chord-key-conf{font-size:9px;color:var(--dim2);margin-left:auto;}
.chord-list{display:flex;flex-wrap:wrap;gap:5px;padding:4px 12px 10px;}
.chord-chip{padding:4px 9px;border-radius:5px;font-size:11px;font-weight:800;letter-spacing:.02em;background:var(--panel2);border:1px solid var(--border2);color:var(--text);cursor:default;transition:background .12s;}
.chord-chip.tonic{border-color:var(--teal);color:var(--teal);}
.chord-chip:hover{background:var(--border2);}
.chord-timeline{max-height:130px;overflow-y:auto;border-top:1px solid var(--border);margin-top:2px;}
.chord-row{display:flex;align-items:center;gap:0;padding:4px 12px;border-bottom:1px solid var(--border);transition:background .1s;}
.chord-row:hover{background:var(--panel2);}
.chord-row:last-child{border-bottom:none;}
.chord-row-name{font-size:12px;font-weight:800;color:var(--text);width:54px;flex-shrink:0;}
.chord-row-bar{flex:1;height:4px;border-radius:2px;background:var(--teal-dim);position:relative;overflow:hidden;margin:0 8px;}
.chord-row-fill{height:100%;border-radius:2px;background:var(--teal);opacity:.6;}
.chord-row-time{font-size:9px;color:var(--dim2);width:38px;text-align:right;flex-shrink:0;font-variant-numeric:tabular-nums;}
.chord-processing{padding:14px 12px;font-size:10px;color:var(--dim);text-align:center;}
/* Light mode */
body.light .chord-chip{background:#F4F7FB;border-color:#C0D0E0;color:#1A2A3A;}
body.light .chord-chip.tonic{border-color:var(--teal);color:var(--teal);}
body.light .chord-key-big{color:#0E1422;}
body.light .chord-row-name{color:#0E1422;font-size:13px;}
body.light .chord-drop-lbl{font-size:11px;}

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
<body class="light">

<!-- ── Header ──────────────────────────────────────────────────────── -->
<div class="hdr">
  <!-- Left: brand -->
  <div style="display:flex;align-items:center;gap:10px;z-index:1">
    <div>
      <div class="brand">EXPLORE</div>
      <div class="brand-sub">AI Mix Engineer</div>
    </div>
    <div id="ableton-dot" class="status-dot" title="Ableton Live"></div>
  </div>

  <!-- Right: health + theme -->
  <div class="hdr-right">
    <div class="hdr-health" id="hdr-health" style="display:none">
      <div>
        <div id="hdr-health-score" class="hdr-health-score">—</div>
        <div id="hdr-health-lbl" class="hdr-health-sub">Mix Health</div>
        <div class="hdr-health-bar"><div id="hdr-health-fill" class="hdr-health-fill" style="width:0%"></div></div>
      </div>
    </div>
    <button class="theme-btn" id="theme-btn" onclick="toggleTheme()" title="Toggle light/dark">◑</button>
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

  <!-- Gain Bridge (leftmost) -->
  <div class="gain-bridge">
    <div class="panel-hdr">Gain Bridge</div>
    <div class="gb-faders">
      <div class="gb-fader">
        <div class="gb-label effort">EFFORT</div>
        <div class="gb-slider-wrap">
          <input type="range" class="gb-slider effort" id="gb-effort" min="0" max="100" value="50"
            oninput="gbFaderInput('intensity', this.value)"
            ondblclick="this.value=50; gbFaderInput('intensity', 50)"
            onmousedown="this._drag=true" onmouseup="this._drag=false"
            ontouchstart="this._drag=true" ontouchend="this._drag=false">
        </div>
        <div class="gb-val effort" id="gb-effort-val">0.50</div>
      </div>
      <div class="gb-fader">
        <div class="gb-label verbosity">VERBOSITY</div>
        <div class="gb-slider-wrap">
          <input type="range" class="gb-slider verbosity" id="gb-verbosity" min="0" max="100" value="50"
            oninput="gbFaderInput('room', this.value)"
            ondblclick="this.value=50; gbFaderInput('room', 50)"
            onmousedown="this._drag=true" onmouseup="this._drag=false"
            ontouchstart="this._drag=true" ontouchend="this._drag=false">
        </div>
        <div class="gb-val verbosity" id="gb-verbosity-val">0.50</div>
      </div>
    </div>
    <div class="gb-btns">
      <button class="gb-btn build"   id="gb-build-btn"   onclick="gbMode('BUILD')">BUILD</button>
      <button class="gb-btn explore" id="gb-explore-btn" onclick="gbMode('EXPLORE')">EXPLORE</button>
      <button class="gb-btn mute"    id="gb-mute-btn"    onclick="gbMute()">MUTE</button>
    </div>
    <div class="gb-prompt-area">
      <textarea class="gb-prompt" id="gb-prompt" rows="3" placeholder="Describe a task…"></textarea>
      <button class="gb-run-btn" id="gb-run-btn" onclick="gbRun()">RUN</button>
      <div class="gb-output" id="gb-output"></div>
    </div>
  </div>

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
        <div id="freq-bands" style="flex:1;overflow:hidden;display:flex;align-items:stretch;">
          <div style="color:var(--dim2);font-size:9px;text-align:center;padding-top:40px;width:100%">Click a track</div>
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

    <!-- Chord ID panel -->
    <div class="chord-panel" id="chord-panel">
      <div class="panel-hdr">
        Chord ID
        <span style="font-size:8px;color:var(--dim2);font-weight:500;letter-spacing:.04em">drop a stem</span>
      </div>
      <input type="file" id="chord-file-input" accept=".mp3,.wav,.flac,.aiff,.aif,.m4a,.ogg" style="display:none">
      <div class="chord-drop-zone" id="chord-drop-zone"
           ondragover="event.preventDefault();this.classList.add('drag-over')"
           ondragleave="this.classList.remove('drag-over')"
           ondrop="event.preventDefault();this.classList.remove('drag-over');runChordID(event.dataTransfer.files[0])">
        <div class="chord-drop-lbl" id="chord-drop-lbl">↑ Upload or drag a stem · <span>click to browse</span></div>
      </div>
      <div id="chord-results" style="display:none">
        <div class="chord-key-row">
          <div class="chord-key-big" id="chord-key-big">—</div>
          <div class="chord-key-mode" id="chord-key-mode"></div>
          <div class="chord-key-conf" id="chord-key-conf"></div>
        </div>
        <div class="chord-list" id="chord-list"></div>
        <div class="chord-timeline" id="chord-timeline"></div>
      </div>
      <div class="chord-processing" id="chord-processing" style="display:none">Analyzing… this takes ~10s</div>
    </div>

    <!-- Scan box -->
    <div class="scan-box">
      <label class="scan-label">Project Folder</label>
      <input class="path-input" id="project-path" placeholder="/Users/you/Music/Project" type="text">
      <button class="scan-btn" id="scan-btn" onclick="scanAudio()">Scan Audio Files</button>
    </div>
  </div>

</div>

<!-- ── Bottom Bar: Big Knob + Chat Input ──────────────────────────────────── -->
<div class="bottom-bar">

  <!-- Big OneKnob-style scan knob -->
  <div class="bk-station">
    <div class="bk-wrap" id="bk-wrap">
      <svg class="bk-svg" id="bk-svg" viewBox="0 0 360 360" xmlns="http://www.w3.org/2000/svg">
        <defs>
          <linearGradient id="bk-wood" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%"   stop-color="#1A0A04"/>
            <stop offset="14%"  stop-color="#7A4B20"/>
            <stop offset="28%"  stop-color="#B07030"/>
            <stop offset="42%"  stop-color="#8A5422"/>
            <stop offset="56%"  stop-color="#5C3412"/>
            <stop offset="70%"  stop-color="#8A5422"/>
            <stop offset="84%"  stop-color="#B07030"/>
            <stop offset="100%" stop-color="#1A0A04"/>
          </linearGradient>
          <filter id="bk-glow" x="-60%" y="-60%" width="220%" height="220%">
            <feGaussianBlur stdDeviation="3.5" result="b"/>
            <feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>
          </filter>
          <filter id="bk-lbl-glow" x="-80%" y="-80%" width="260%" height="260%">
            <feGaussianBlur stdDeviation="1.5" result="b"/>
            <feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>
          </filter>
        </defs>
        <!-- Dark base -->
        <circle cx="180" cy="180" r="178" fill="#040810"/>
        <!-- Outer label zone bg (subtle dark ring) -->
        <circle cx="180" cy="180" r="178" fill="none" stroke="rgba(0,0,0,.3)" stroke-width="36"/>
        <!-- Walnut wood ring — sits just outside inner face -->
        <circle cx="180" cy="180" r="108" fill="none" stroke="url(#bk-wood)" stroke-width="22"/>
        <!-- Wood ring inner shadow -->
        <circle cx="180" cy="180" r="97" fill="none" stroke="rgba(0,0,0,.55)" stroke-width="2"/>
        <!-- Wood ring outer sheen -->
        <circle cx="180" cy="180" r="119" fill="none" stroke="rgba(255,200,120,.08)" stroke-width="1"/>
        <!-- Arc track background (outside wood ring) -->
        <path id="bk-arc-bg" fill="none" stroke="rgba(0,180,165,.38)" stroke-width="7" stroke-linecap="round"/>
        <!-- Arc fill (active green) -->
        <path id="bk-arc-fill" fill="none" stroke="#00C8BE" stroke-width="7" stroke-linecap="round" filter="url(#bk-glow)"/>
        <!-- Tick marks (JS-updated) -->
        <g id="bk-marks"></g>
        <!-- LED position dots on arc ring -->
        <circle class="bk-dot" id="bk-dot-0" cx="86.0"  cy="274.0" onclick="knobJump(0)"/>
        <circle class="bk-dot" id="bk-dot-1" cx="47.7"  cy="193.9" onclick="knobJump(1)"/>
        <circle class="bk-dot" id="bk-dot-2" cx="67.2"  cy="109.5" onclick="knobJump(2)"/>
        <circle class="bk-dot" id="bk-dot-3" cx="136.7" cy="54.3"  onclick="knobJump(3)"/>
        <circle class="bk-dot" id="bk-dot-4" cx="223.3" cy="54.3"  onclick="knobJump(4)"/>
        <circle class="bk-dot" id="bk-dot-5" cx="292.8" cy="109.5" onclick="knobJump(5)"/>
        <circle class="bk-dot" id="bk-dot-6" cx="312.3" cy="193.9" onclick="knobJump(6)"/>
        <circle class="bk-dot" id="bk-dot-7" cx="274.0" cy="274.0" onclick="knobJump(7)"/>
        <!-- Invisible hit targets (click to jump to that mode) -->
        <circle class="bk-hit" cx="73.9"  cy="287.1" r="22" onclick="knobJump(0)"/>
        <circle class="bk-hit" cx="30.8"  cy="196.7" r="22" onclick="knobJump(1)"/>
        <circle class="bk-hit" cx="52.8"  cy="101.5" r="22" onclick="knobJump(2)"/>
        <circle class="bk-hit" cx="131.2" cy="39.2"  r="22" onclick="knobJump(3)"/>
        <circle class="bk-hit" cx="228.8" cy="39.2"  r="22" onclick="knobJump(4)"/>
        <circle class="bk-hit" cx="307.2" cy="101.5" r="22" onclick="knobJump(5)"/>
        <circle class="bk-hit" cx="329.2" cy="196.7" r="22" onclick="knobJump(6)"/>
        <circle class="bk-hit" cx="286.1" cy="287.1" r="22" onclick="knobJump(7)"/>
        <!-- Static mode labels — JS toggles .active class for highlight -->
        <text id="bk-lbl-0" class="bk-lbl"  x="73.9"  y="290.6" text-anchor="middle" onclick="knobJump(0)">SCAN</text>
        <text id="bk-lbl-1" class="bk-lbl"  x="30.8"  y="200.2" text-anchor="middle" onclick="knobJump(1)">MUD</text>
        <text id="bk-lbl-2" class="bk-lbl"  x="52.8"  y="105.0" text-anchor="middle" onclick="knobJump(2)">VOCAL</text>
        <text id="bk-lbl-3" class="bk-lbl"  x="131.2" y="42.7"  text-anchor="middle" onclick="knobJump(3)">SPACE</text>
        <text id="bk-lbl-4" class="bk-lbl"  x="228.8" y="42.7"  text-anchor="middle" onclick="knobJump(4)">LOW</text>
        <text id="bk-lbl-5" class="bk-lbl"  x="307.2" y="105.0" text-anchor="middle" onclick="knobJump(5)">DYN</text>
        <text id="bk-lbl-6" class="bk-lbl"  x="329.2" y="200.2" text-anchor="middle" onclick="knobJump(6)">PRI</text>
        <text id="bk-lbl-7" class="bk-lbl"  x="286.1" y="290.6" text-anchor="middle" onclick="knobJump(7)">ARR</text>
      </svg>
      <!-- Inner knob face (CSS — pointer rotates via JS) -->
      <div class="bk-face-outer">
        <div class="bk-face-shell" id="bk-face-shell"
             onclick="knobClick(event)"
             oncontextmenu="knobAdvance(-1);event.preventDefault()"
             title="Left: back · Right: forward · Scroll: scroll modes">
          <div class="bk-knurl"></div>
          <div class="bk-inner-face">
            <div class="bk-ptr" id="bk-ptr"></div>
            <div class="bk-cap"></div>
          </div>
        </div>
      </div>
    </div>
    <!-- Mode name + RUN -->
    <div class="bk-label-row">
      <div class="bk-mode-name" id="knob-mode-lbl">SCAN</div>
      <div class="bk-hint">← → scroll · click knob</div>
      <button class="scan-run-btn" onclick="knobRun()" id="knob-run-btn">RUN</button>
    </div>
  </div>

  <!-- Chat input (fills remaining width) -->
  <div class="bottom-chat">
    <textarea class="chat-input" id="chat-input" placeholder="Ask about your mix..." style="flex:1;height:auto;min-height:80px;max-height:220px;resize:none;"></textarea>
    <button class="send-btn" id="send-btn" onclick="sendMessage()">Ask</button>
  </div>

</div>

<div class="toast" id="toast"></div>

<script>
// ── State ──────────────────────────────────────────────────────────────────────
var sessionCtx = '';
var problemCtx = '';
var allTracks = [];
var allAudioData = {};
var sessionData = {};
var currentProblems = [];
var trackScores = {};
var overallHealth = 0;
var selectedTrackIdx = -1;
var gainData = {};

// ── Persistence ───────────────────────────────────────────────────────────────
var chatHistory = [];
var STORAGE_KEY = 'explore_v1';

function buildStateObj() {
  return {
    projectPath:   document.getElementById('project-path').value,
    chatHistory:   chatHistory.slice(-500),
    sessionCtx:    sessionCtx,
    problemCtx:    problemCtx,
    trackScores:   trackScores,
    overallHealth: overallHealth,
    allTracks:     allTracks,
    session:       sessionData,
    problems:      currentProblems,
  };
}

function applyState(d) {
  if (!d) return;
  try {
    if (d.projectPath) document.getElementById('project-path').value = d.projectPath;
    if (d.sessionCtx)    sessionCtx    = d.sessionCtx;
    if (d.problemCtx)    problemCtx    = d.problemCtx;
    if (d.trackScores)   trackScores      = d.trackScores;
    if (d.overallHealth) overallHealth    = d.overallHealth;
    if (d.allTracks)     allTracks        = d.allTracks;
    if (d.session)       sessionData      = d.session;
    if (d.problems)      currentProblems  = d.problems;
    if (d.allTracks && d.allTracks.length) {
      renderTracks(d.allTracks);
      renderHealthChart(d.allTracks, d.trackScores || {});
      document.getElementById('ableton-dot').classList.add('on');
    }
    if (d.overallHealth) {
      renderOverallHealth(d.overallHealth, d.allTracks || [], d.problems || []);
      updateStatCards(d.session || null, d.allTracks || [], d.overallHealth);
    }
    if (d.problems && d.problems.length) renderProblems(d.problems);
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

// Write to server file (durable) + localStorage (instant cache)
function saveState() {
  var obj = buildStateObj();
  // localStorage: instant
  try { localStorage.setItem(STORAGE_KEY, JSON.stringify(obj)); } catch(e) {}
  // Server file: durable
  fetch('/api/state', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(obj)
  }).catch(function() {});
}

// Load: try localStorage first (instant paint), then server (truth)
function loadState() {
  // 1. Instant render from localStorage cache
  try {
    var raw = localStorage.getItem(STORAGE_KEY);
    if (raw) applyState(JSON.parse(raw));
  } catch(e) {}

  // 2. Authoritative load from server file (may have data from other browsers/sessions)
  fetch('/api/state', {cache: 'no-store'})
    .then(function(r) { return r.json(); })
    .then(function(d) {
      if (!d || d.error) return;
      applyState(d);
      // Sync localStorage with server truth
      try { localStorage.setItem(STORAGE_KEY, JSON.stringify(d)); } catch(e) {}
    })
    .catch(function() {});
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
  var html = esc(s);
  // Headers
  html = html.replace(/^### (.+)$/gm, '<div style="font-weight:800;font-size:11px;letter-spacing:.06em;text-transform:uppercase;color:var(--teal);margin:10px 0 3px">$1</div>');
  html = html.replace(/^## (.+)$/gm,  '<div style="font-weight:800;font-size:12px;color:var(--text);margin:10px 0 4px">$1</div>');
  html = html.replace(/^# (.+)$/gm,   '<div style="font-weight:900;font-size:13px;color:var(--text);margin:10px 0 4px">$1</div>');
  // Bold / italic
  html = html.replace(/\*\*\*(.+?)\*\*\*/g, '<strong><em>$1</em></strong>');
  html = html.replace(/\*\*(.+?)\*\*/g,     '<strong>$1</strong>');
  html = html.replace(/\*(.+?)\*/g,         '<em>$1</em>');
  // Inline code
  html = html.replace(/`([^`]+)`/g, '<code style="background:rgba(0,168,152,.12);color:var(--teal);padding:1px 5px;border-radius:3px;font-size:11px;font-family:monospace">$1</code>');
  // Bullet lists — convert leading "- " on a line
  html = html.replace(/^[-•] (.+)$/gm, '<div style="display:flex;gap:6px;margin:2px 0"><span style="color:var(--teal);flex-shrink:0">·</span><span>$1</span></div>');
  // Horizontal rules
  html = html.replace(/^---+$/gm, '<hr style="border:none;border-top:1px solid var(--border);margin:8px 0">');
  // Newlines
  html = html.replace(/\\n/g, '<br>').replace(/\n/g, '<br>');
  return html;
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
    var hasAudio = Object.keys(allAudioData).length > 0;
    document.getElementById('stat-health-lbl').textContent =
      healthLabel(health) + (hasAudio ? '' : ' · no audio');
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

  // Gain Bridge
  gbSyncFromGain(g);
}

// ── Gain Bridge ───────────────────────────────────────────────────────────────
var _gbMuted = false;
var _gbLastMode = '';

function gbPost(fields) {
  fetch('/api/gain/set', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(fields)
  }).catch(function() {});
}

function gbFaderInput(field, rawVal) {
  var val = Math.round(rawVal * 10) / 1000;  // 0-100 → 0.000-1.000
  val = parseFloat(val.toFixed(3));
  var elId = field === 'intensity' ? 'gb-effort-val' : 'gb-verbosity-val';
  document.getElementById(elId).textContent = val.toFixed(2);
  var payload = {};
  payload[field] = val;
  gbPost(payload);
}

function gbMode(mode) {
  var buildBtn   = document.getElementById('gb-build-btn');
  var exploreBtn = document.getElementById('gb-explore-btn');
  var cur = buildBtn.classList.contains('active') ? 'BUILD'
          : exploreBtn.classList.contains('active') ? 'EXPLORE' : '';
  var next = (cur === mode) ? '' : mode;
  buildBtn.classList.toggle('active', next === 'BUILD');
  exploreBtn.classList.toggle('active', next === 'EXPLORE');
  gbPost({mode: next});
  // If BUILD is activated with a prompt, run immediately
  if (next === 'BUILD') {
    var promptEl = document.getElementById('gb-prompt');
    if (promptEl && promptEl.value.trim()) gbRun();
  }
}

function gbMute() {
  _gbMuted = !_gbMuted;
  var btn = document.getElementById('gb-mute-btn');
  if (btn) btn.classList.toggle('active', _gbMuted);
  gbPost({t1_on: !_gbMuted});
}

async function gbRun() {
  var promptEl = document.getElementById('gb-prompt');
  var btn      = document.getElementById('gb-run-btn');
  var out      = document.getElementById('gb-output');
  var task = (promptEl.value || '').trim();
  if (!task) return;
  btn.disabled = true; btn.textContent = '···';
  out.style.display = 'block';
  out.textContent = 'Running…';
  try {
    var r = await fetch('/api/gain/run', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({task: task})
    });
    var d = await r.json();
    if (d.error) {
      out.textContent = '⚠ ' + d.error;
    } else {
      out.textContent = d.output || '';
      // Flash mode badge if response included mode
      if (d.mode) {
        var modeHint = document.createElement('div');
        modeHint.style.cssText = 'font-size:8px;opacity:.5;margin-top:4px;letter-spacing:.08em;text-transform:uppercase;';
        modeHint.textContent = '— ' + d.mode + ' mode';
        out.appendChild(modeHint);
      }
    }
  } catch(e) {
    out.textContent = '⚠ ' + e.message;
  }
  btn.disabled = false; btn.textContent = 'RUN';
}

function gbSyncFromGain(g) {
  // Update faders (skip if user is actively dragging)
  var effortEl = document.getElementById('gb-effort');
  if (effortEl && !effortEl._drag && g.intensity !== undefined) {
    effortEl.value = Math.round((g.intensity || 0) * 100);
    document.getElementById('gb-effort-val').textContent = (g.intensity || 0).toFixed(2);
  }
  var verbEl = document.getElementById('gb-verbosity');
  if (verbEl && !verbEl._drag && g.room !== undefined) {
    verbEl.value = Math.round((g.room || 0) * 100);
    document.getElementById('gb-verbosity-val').textContent = (g.room || 0).toFixed(2);
  }
  // Mode buttons
  var mode = g.mode || '';
  var buildBtn   = document.getElementById('gb-build-btn');
  var exploreBtn = document.getElementById('gb-explore-btn');
  if (buildBtn)   buildBtn.classList.toggle('active', mode === 'BUILD');
  if (exploreBtn) exploreBtn.classList.toggle('active', mode === 'EXPLORE');
  _gbLastMode = mode;
  // Mute
  _gbMuted = (g.t1_on === false);
  var muteBtn = document.getElementById('gb-mute-btn');
  if (muteBtn) muteBtn.classList.toggle('active', _gbMuted);
}

async function fetchGain() {
  try {
    var r = await fetch('/api/gain', {cache: 'no-store'});
    var d = await r.json();
    renderGain(d);
  } catch(e) {}
}

// ── Rotary Knob ────────────────────────────────────────────────────────────────
var KNOB_MODES = [
  { name:'SCAN',     fn: function(){ runScan(); } },
  { name:'MUD',      fn: function(){ quickPrompt('Why does this mix sound muddy? Identify every track contributing to low-mid buildup (200-500 Hz). Give me specific EQ cuts with frequencies and amounts.'); } },
  { name:'VOCAL',    fn: function(){ quickPrompt('Analyse the vocal tracks in this session. Is the vocal sitting clearly in the mix? What is competing with it in the 1-4 kHz presence range? Give me exact fixes.'); } },
  { name:'SPACE',    fn: function(){ quickPrompt('Map the frequency space in this mix. Which tracks are occupying the same zones? Where is it congested and where is it thin? Be specific about Hz ranges and which tracks.'); } },
  { name:'LOW END',  fn: function(){ quickPrompt('Analyse the low end in this mix. Which track should own 20-60 Hz sub? What is happening 60-200 Hz? Are the bass sources conflicting? Give me a concrete hierarchy and fixes.'); } },
  { name:'DYNAMICS', fn: function(){ quickPrompt('How is the dynamic balance in this mix? Is anything over-compressed or squashed? What are the loudness relationships between elements? What should be levelled or automated?'); } },
  { name:'PRIORITY', fn: function(){ quickPrompt('Given everything you know about this session, what is the single highest-impact change I can make right now? Rank the top 3 issues by the improvement they would create.'); } },
  { name:'ARRANGE',  fn: function(){ loadArrangement(); } },
];
var KNOB_ANGLES = [-135, -96, -58, -19, 19, 58, 96, 135];
var knobPos = 0;

function knobClick(e) {
  var rect = e.currentTarget.getBoundingClientRect();
  var cx = rect.left + rect.width / 2;
  knobAdvance(e.clientX >= cx ? 1 : -1);
}

function knobAdvance(dir) {
  knobPos = (knobPos + dir + KNOB_MODES.length) % KNOB_MODES.length;
  drawKnob();
}

function knobJump(pos) {
  knobPos = pos;
  drawKnob();
}

function drawKnob() {
  var angle = KNOB_ANGLES[knobPos];
  // Rotate CSS pointer on inner face
  var ptr = document.getElementById('bk-ptr');
  if (ptr) ptr.style.transform = 'rotate(' + angle + 'deg)';

  // SVG arc helpers — center (180,180), angles measured from top, clockwise
  function angXY(deg, r) {
    var rad = (deg - 90) * Math.PI / 180;
    return [180 + r * Math.cos(rad), 180 + r * Math.sin(rad)];
  }
  function arcPath(a1, a2, r) {
    var p1 = angXY(a1, r);
    var p2 = angXY(a2, r);
    var diff = a2 - a1;
    if (diff <= 0) diff += 360;
    var large = diff > 180 ? 1 : 0;
    return 'M ' + p1[0].toFixed(2) + ' ' + p1[1].toFixed(2) +
           ' A ' + r + ' ' + r + ' 0 ' + large + ' 1 ' +
           p2[0].toFixed(2) + ' ' + p2[1].toFixed(2);
  }

  // Full background arc at r=133 (outside wood ring outer edge r=119)
  var bgArc = document.getElementById('bk-arc-bg');
  if (bgArc) bgArc.setAttribute('d', arcPath(-135, 135, 133));

  // Active fill arc: from start to current
  var fillArc = document.getElementById('bk-arc-fill');
  if (fillArc) {
    if (knobPos === 0) {
      fillArc.setAttribute('d', '');  // at start, no fill
    } else {
      fillArc.setAttribute('d', arcPath(-135, angle, 133));
    }
  }

  // Tick marks (one per mode, dynamic)
  var marks = document.getElementById('bk-marks');
  if (marks) {
    marks.innerHTML = '';
    var ns = 'http://www.w3.org/2000/svg';
    for (var i = 0; i < KNOB_MODES.length; i++) {
      var a = KNOB_ANGLES[i];
      var active = (i === knobPos);
      var t1 = angXY(a, 122); var t2 = angXY(a, 128);
      var tick = document.createElementNS(ns, 'line');
      tick.setAttribute('x1', t1[0].toFixed(1)); tick.setAttribute('y1', t1[1].toFixed(1));
      tick.setAttribute('x2', t2[0].toFixed(1)); tick.setAttribute('y2', t2[1].toFixed(1));
      tick.setAttribute('stroke', active ? '#00C8BE' : 'rgba(200,230,240,.3)');
      tick.setAttribute('stroke-width', active ? '3' : '1.5');
      tick.setAttribute('stroke-linecap', 'round');
      marks.appendChild(tick);
    }
  }

  // Toggle active class on static label text elements and LED dots
  for (var j = 0; j < KNOB_MODES.length; j++) {
    var lel = document.getElementById('bk-lbl-' + j);
    if (lel) lel.setAttribute('class', j === knobPos ? 'bk-lbl active' : 'bk-lbl');
    var dot = document.getElementById('bk-dot-' + j);
    if (dot) dot.setAttribute('class', j === knobPos ? 'bk-dot active' : 'bk-dot');
  }

  // Mode name label below the knob
  var ml = document.getElementById('knob-mode-lbl');
  if (ml) ml.textContent = KNOB_MODES[knobPos].name;
}

function knobRun() {
  var btn = document.getElementById('knob-run-btn');
  if (btn) { btn.textContent = '···'; btn.disabled = true; }
  var restore = function() { if (btn) { btn.textContent = 'RUN'; btn.disabled = false; } };
  try {
    KNOB_MODES[knobPos].fn();
  } catch(e) {}
  setTimeout(restore, 2000);
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
    sessionCtx      = d.session_ctx || '';
    problemCtx      = d.problem_ctx || '';
    allTracks       = d.tracks || [];
    allAudioData    = d.audio_data || {};
    trackScores     = d.track_scores || {};
    overallHealth   = d.overall_health || 0;
    sessionData     = d.session || {};
    currentProblems = d.problems || [];

    if (!allTracks.length) {
      document.getElementById('ableton-dot').classList.remove('on');
      document.getElementById('track-list').innerHTML =
        '<div style="padding:14px 10px;font-size:10px;color:var(--dim);line-height:1.6">'
        + '<div style="font-weight:800;color:var(--orange);margin-bottom:6px">⚠ Ableton Not Connected</div>'
        + 'Make sure:<br>'
        + '1. Ableton is open<br>'
        + '2. AbletonMCP is enabled in Ableton → Settings → MIDI<br>'
        + '3. The AbletonMCP server script is running'
        + '</div>';
      toast('No tracks — is AbletonMCP running?', true);
      return;
    }
    document.getElementById('ableton-dot').classList.add('on');
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
    if (!t || !t.name) continue;
    var name = t.name;
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
  if (!svgEl) return;
  if (!tracks.length) { svgEl.innerHTML = ''; return; }

  var W = 400, H = 200;
  svgEl.setAttribute('viewBox', '0 0 ' + W + ' ' + H);
  svgEl.setAttribute('preserveAspectRatio', 'none');

  var n = tracks.length;
  var padL = 24, padR = 12, padT = 12, padB = 24;
  var chartW = W - padL - padR;
  var chartH = H - padT - padB;

  // Build data points
  var pts = [];
  for (var i = 0; i < n; i++) {
    var sc = scores[tracks[i].name] !== undefined ? scores[tracks[i].name] : 72;
    var x = padL + (n === 1 ? chartW / 2 : (i / (n - 1)) * chartW);
    var y = padT + chartH - (sc / 100) * chartH;
    pts.push({x: x, y: y, score: sc, name: tracks[i].name || ('T' + (i+1))});
  }

  // Smooth bezier path
  function bezierPath(points) {
    if (!points.length) return '';
    if (points.length === 1) return 'M' + points[0].x + ',' + points[0].y;
    var d = 'M' + points[0].x + ',' + points[0].y;
    for (var k = 1; k < points.length; k++) {
      var prev = points[k-1], curr = points[k];
      var cpx = (prev.x + curr.x) / 2;
      d += ' C' + cpx + ',' + prev.y + ' ' + cpx + ',' + curr.y + ' ' + curr.x + ',' + curr.y;
    }
    return d;
  }

  var linePath = bezierPath(pts);
  var areaPath = linePath
    + ' L' + pts[pts.length-1].x + ',' + (padT + chartH)
    + ' L' + pts[0].x + ',' + (padT + chartH) + ' Z';

  // Color by avg score
  var avg = 0;
  for (var i = 0; i < pts.length; i++) avg += pts[i].score;
  avg = avg / pts.length;
  var c1 = avg >= 75 ? '#00E0D0' : avg >= 55 ? '#D88040' : '#E85870';
  var c2 = avg >= 75 ? '#008880' : avg >= 55 ? '#905020' : '#901830';
  var cMid = avg >= 75 ? '#00B8A8' : avg >= 55 ? '#B86030' : '#C03050';

  var html = '<defs>'
    + '<linearGradient id="wf" x1="0" y1="0" x2="0" y2="1">'
    + '<stop offset="0%" stop-color="' + c1 + '" stop-opacity=".5"/>'
    + '<stop offset="70%" stop-color="' + c2 + '" stop-opacity=".12"/>'
    + '<stop offset="100%" stop-color="' + c2 + '" stop-opacity=".02"/>'
    + '</linearGradient>'
    + '<linearGradient id="wl" x1="0" y1="0" x2="1" y2="0">'
    + '<stop offset="0%" stop-color="' + c1 + '" stop-opacity=".35"/>'
    + '<stop offset="30%" stop-color="' + c1 + '"/>'
    + '<stop offset="70%" stop-color="' + c1 + '"/>'
    + '<stop offset="100%" stop-color="' + c1 + '" stop-opacity=".35"/>'
    + '</linearGradient>'
    + '<filter id="glow"><feGaussianBlur stdDeviation="3" result="blur"/>'
    + '<feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge></filter>'
    + '</defs>';

  // Grid lines
  var lt = isLight();
  var gridStroke = lt ? 'rgba(0,0,0,.07)' : 'rgba(255,255,255,.04)';
  var gridLabel  = lt ? 'rgba(0,0,0,.3)'  : 'rgba(255,255,255,.18)';
  var dropStroke = lt ? 'rgba(0,0,0,.07)' : 'rgba(255,255,255,.06)';
  var nameColor  = lt ? 'rgba(0,0,0,.4)'  : 'rgba(255,255,255,.22)';
  var gridVals = [25, 50, 75, 100];
  for (var g = 0; g < gridVals.length; g++) {
    var gv = gridVals[g];
    var gy = padT + chartH - (gv / 100) * chartH;
    html += '<line x1="' + padL + '" y1="' + gy + '" x2="' + (W - padR) + '" y2="' + gy
      + '" stroke="' + gridStroke + '" stroke-width="1" stroke-dasharray="3,4"/>';
    html += '<text x="' + (padL - 4) + '" y="' + (gy + 3) + '" font-size="7" fill="' + gridLabel + '"'
      + ' text-anchor="end" font-family="Inter,sans-serif">' + gv + '</text>';
  }

  // Area fill
  html += '<path d="' + areaPath + '" fill="url(#wf)"/>';

  // Glow blur line
  html += '<path d="' + linePath + '" fill="none" stroke="' + c1 + '" stroke-width="6"'
    + ' stroke-opacity=".2" stroke-linecap="round" stroke-linejoin="round"/>';

  // Main line
  html += '<path d="' + linePath + '" fill="none" stroke="url(#wl)" stroke-width="2.5"'
    + ' stroke-linecap="round" stroke-linejoin="round"/>';

  // Dots + labels
  for (var j = 0; j < pts.length; j++) {
    var p = pts[j];
    var dc = p.score >= 80 ? '#00E0D0' : p.score >= 60 ? '#D88040' : '#E85870';
    var tname = p.name.substring(0, 9);
    // Drop line
    html += '<line x1="' + p.x + '" y1="' + p.y + '" x2="' + p.x + '" y2="' + (padT + chartH)
      + '" stroke="' + dropStroke + '" stroke-width="1" stroke-dasharray="2,3"/>';
    // Outer ring
    html += '<circle cx="' + p.x + '" cy="' + p.y + '" r="5" fill="' + dc + '" opacity=".2"/>';
    // Dot
    html += '<circle cx="' + p.x + '" cy="' + p.y + '" r="3" fill="' + dc
      + '" stroke="rgba(0,0,0,.6)" stroke-width="1"/>';
    // Score label above dot
    html += '<text x="' + p.x + '" y="' + (p.y - 8) + '" font-size="8" fill="' + dc
      + '" text-anchor="middle" font-family="Inter,sans-serif" font-weight="800">' + p.score + '</text>';
    // Track name at bottom
    html += '<text x="' + p.x + '" y="' + (H - 4) + '" font-size="6.5" fill="' + nameColor + '"'
      + ' text-anchor="middle" font-family="Inter,sans-serif">' + esc(tname) + '</text>';
  }

  svgEl.innerHTML = html;
}

function renderFreqMap(adat, trackName) {
  var el = document.getElementById('freq-bands');
  var label = document.getElementById('freq-track-name');
  if (!adat || adat.peak_db === undefined) {
    el.innerHTML = '<div style="color:var(--dim2);font-size:9px;text-align:center;padding-top:40px">Click a track</div>';
    if (label) label.textContent = '';
    return;
  }
  if (label) label.textContent = trackName || '';

  var bands = [
    {label:'SUB',  key:'sub_energy',      crit: function(p){return p>25;}, warn: function(p){return p>18;}},
    {label:'BASS', key:'bass_energy',     crit: function(p){return p>40;}, warn: function(p){return p>30;}},
    {label:'LMID', key:'low_mid_energy',  crit: function(p){return p>28;}, warn: function(p){return p>18;}},
    {label:'MID',  key:'mid_energy',      crit: function(p){return false;},warn: function(p){return false;}},
    {label:'HMID', key:'high_mid_energy', crit: function(p){return false;},warn: function(p){return false;}},
    {label:'AIR',  key:'air_energy',      crit: function(p){return false;},warn: function(p){return false;}},
  ];

  // SVG cylinder chart
  var W = 186, H = 200;
  var n = bands.length;
  var colW = W / n;
  var cylW = Math.floor(colW * 0.62);
  var rx = cylW / 2;
  var ry = Math.max(3, Math.floor(rx * 0.32));
  var baseY = H - 22;
  var maxBarH = baseY - 18;

  var svg = '<svg viewBox="0 0 ' + W + ' ' + H + '" xmlns="http://www.w3.org/2000/svg" preserveAspectRatio="xMidYMid meet"><defs>';

  // Gradients per band
  for (var i = 0; i < n; i++) {
    var b = bands[i];
    var raw = adat[b.key] || 0;
    var p = Math.round(raw * 100);
    var isCrit = b.crit(p), isWarn = b.warn(p);
    var ca = isCrit ? '#E85870' : isWarn ? '#D88040' : '#00E0D0';
    var cb = isCrit ? '#701020' : isWarn ? '#804010' : '#006858';
    var cc = isCrit ? '#B03050' : isWarn ? '#A05828' : '#00A898';

    svg += '<linearGradient id="cbody' + i + '" x1="0" y1="0" x2="1" y2="0">'
      + '<stop offset="0%" stop-color="' + cb + '"/>'
      + '<stop offset="35%" stop-color="' + cc + '"/>'
      + '<stop offset="65%" stop-color="' + ca + '"/>'
      + '<stop offset="100%" stop-color="' + cb + '"/>'
      + '</linearGradient>';
    svg += '<radialGradient id="ctop' + i + '" cx="42%" cy="38%" r="58%">'
      + '<stop offset="0%" stop-color="#ffffff" stop-opacity=".55"/>'
      + '<stop offset="60%" stop-color="' + ca + '" stop-opacity=".9"/>'
      + '<stop offset="100%" stop-color="' + cb + '" stop-opacity=".7"/>'
      + '</radialGradient>';
    svg += '<radialGradient id="cbot' + i + '" cx="50%" cy="50%" r="50%">'
      + '<stop offset="0%" stop-color="' + cc + '" stop-opacity=".6"/>'
      + '<stop offset="100%" stop-color="' + cb + '" stop-opacity=".3"/>'
      + '</radialGradient>';
  }

  // Glow filter
  svg += '<filter id="cglow" x="-40%" y="-40%" width="180%" height="180%">'
    + '<feGaussianBlur stdDeviation="2.5" result="b"/>'
    + '<feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>'
    + '</filter>';
  svg += '</defs>';

  // Floor line
  var ltf = isLight();
  var floorStroke = ltf ? 'rgba(0,0,0,.14)' : 'rgba(255,255,255,.08)';
  var bandLabelColor = ltf ? 'rgba(0,0,0,.5)' : 'rgba(255,255,255,.35)';
  svg += '<line x1="4" y1="' + baseY + '" x2="' + (W-4) + '" y2="' + baseY
    + '" stroke="' + floorStroke + '" stroke-width="1"/>';

  // Cylinders
  for (var i = 0; i < n; i++) {
    var b = bands[i];
    var raw = adat[b.key] || 0;
    var p = Math.round(raw * 100);
    var isCrit = b.crit(p), isWarn = b.warn(p);
    var ca = isCrit ? '#E85870' : isWarn ? '#D88040' : '#00E0D0';
    var cb = isCrit ? '#701020' : isWarn ? '#804010' : '#006858';

    var barH = Math.max(ry * 2 + 2, Math.round((Math.min(p, 100) / 100) * maxBarH));
    var cx = i * colW + colW / 2;
    var topY = baseY - barH;
    var bodyH = barH - ry;

    // Shadow glow under cylinder
    svg += '<ellipse cx="' + cx + '" cy="' + baseY + '" rx="' + (rx * 1.3) + '" ry="' + (ry * 0.7)
      + '" fill="' + ca + '" opacity=".08"/>';

    // Body
    svg += '<rect x="' + (cx - rx) + '" y="' + topY + '" width="' + (rx*2) + '" height="' + bodyH
      + '" fill="url(#cbody' + i + ')"/>';

    // Bottom cap
    svg += '<ellipse cx="' + cx + '" cy="' + (topY + bodyH) + '" rx="' + rx + '" ry="' + ry
      + '" fill="url(#cbot' + i + ')"/>';

    // Top cap (highlight)
    svg += '<ellipse cx="' + cx + '" cy="' + topY + '" rx="' + rx + '" ry="' + ry
      + '" fill="url(#ctop' + i + ')" filter="url(#cglow)"/>';

    // Percentage above
    svg += '<text x="' + cx + '" y="' + (topY - ry - 4) + '" font-size="8.5" fill="' + ca
      + '" text-anchor="middle" font-family="Inter,sans-serif" font-weight="900">' + p + '%</text>';

    // Band label below floor
    svg += '<text x="' + cx + '" y="' + (H - 5) + '" font-size="7" fill="' + bandLabelColor + '"'
      + ' text-anchor="middle" font-family="Inter,sans-serif" font-weight="800">' + b.label + '</text>';
  }

  el.innerHTML = svg + '</svg>';
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
    var aiMeta = d.tokens ? d.tokens + ' tok' : '';
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
    addMessage('ai', 'Arrangement info:\\n\\n' + JSON.stringify(d, null, 2).substring(0, 600));
    toast('Arrangement loaded');
  } catch(e) { toast('Could not load arrangement', true); }
}

// ── Input ─────────────────────────────────────────────────────────────────────
(function() {
  var ci = document.getElementById('chat-input');
  if (ci) ci.addEventListener('keydown', function(e) {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
  });
  var pp = document.getElementById('project-path');
  if (pp) pp.addEventListener('change', saveState);
  // Arrow keys rotate the scan knob (when not focused on a text input)
  document.addEventListener('keydown', function(e) {
    var tag = document.activeElement ? document.activeElement.tagName : '';
    if (tag === 'INPUT' || tag === 'TEXTAREA') return;
    if (e.key === 'ArrowRight' || e.key === 'ArrowUp') { e.preventDefault(); knobAdvance(1); }
    if (e.key === 'ArrowLeft'  || e.key === 'ArrowDown') { e.preventDefault(); knobAdvance(-1); }
    if (e.key === 'Enter') { e.preventDefault(); knobRun(); }
  });
  // Scroll wheel on knob shell
  var shell = document.getElementById('bk-face-shell');
  if (shell) {
    shell.addEventListener('wheel', function(e) {
      e.preventDefault();
      knobAdvance(e.deltaY > 0 ? 1 : -1);
    }, {passive: false});
  }
})();

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

// ── Theme ─────────────────────────────────────────────────────────────────────
function isLight() { return document.body.classList.contains('light'); }

function toggleTheme() {
  var nowLight = document.body.classList.toggle('light');
  localStorage.setItem('explore_theme', nowLight ? 'light' : 'dark');
  var btn = document.getElementById('theme-btn');
  if (btn) btn.textContent = nowLight ? '◑' : '◐';
  if (allTracks.length) renderHealthChart(allTracks, trackScores);
  if (selectedTrackIdx >= 0 && allTracks[selectedTrackIdx]) {
    var t = allTracks[selectedTrackIdx];
    renderFreqMap(allAudioData[t.name] || {}, t.name);
  }
}

// ── Gain Bridge slider resize ──────────────────────────────────────────────────
function resizeGBSliders() {
  var wraps = document.querySelectorAll('.gb-slider-wrap');
  wraps.forEach(function(wrap) {
    var h = wrap.clientHeight;
    var slider = wrap.querySelector('.gb-slider');
    if (slider && h > 20) {
      slider.style.width = Math.max(60, h - 20) + 'px';
    }
  });
}
window.addEventListener('resize', resizeGBSliders);

// ── Chord ID ──────────────────────────────────────────────────────────────────
function chordDropLbl(html) {
  var el = document.getElementById('chord-drop-lbl');
  if (el) el.innerHTML = html;
}

async function runChordID(file) {
  if (!file) return;
  var proc = document.getElementById('chord-processing');
  var res  = document.getElementById('chord-results');
  if (proc) proc.style.display = 'block';
  if (res)  res.style.display  = 'none';
  chordDropLbl('⏳ Analyzing ' + file.name + '…');

  var fd = new FormData();
  fd.append('file', file);

  try {
    var r = await fetch('/api/chords', {method:'POST', body: fd});
    var d = await r.json();
    if (proc) proc.style.display = 'none';

    if (d.error) {
      chordDropLbl('⚠ ' + d.error + ' · <span>try again</span>');
      toast('Chord ID error: ' + d.error, true);
      return;
    }

    // Key display
    document.getElementById('chord-key-big').textContent  = d.root || '—';
    document.getElementById('chord-key-mode').textContent = d.mode || '';
    document.getElementById('chord-key-conf').textContent = d.duration ? d.duration + 's' : '';

    // Chord chips
    var listEl = document.getElementById('chord-list');
    listEl.innerHTML = '';
    (d.chord_list || []).forEach(function(ch) {
      var chip = document.createElement('div');
      chip.className = 'chord-chip' + (ch === d.root || ch === d.root + 'm' ? ' tonic' : '');
      chip.textContent = ch;
      listEl.appendChild(chip);
    });

    // Timeline
    var tl = document.getElementById('chord-timeline');
    tl.innerHTML = '';
    var maxDur = Math.max.apply(null, (d.segments || []).map(function(s){ return s.dur; }));
    (d.segments || []).forEach(function(seg) {
      var row = document.createElement('div');
      row.className = 'chord-row';
      var pct = maxDur > 0 ? Math.round(seg.dur / maxDur * 100) : 0;
      row.innerHTML =
        '<div class="chord-row-name">' + seg.chord + '</div>' +
        '<div class="chord-row-bar"><div class="chord-row-fill" style="width:' + pct + '%"></div></div>' +
        '<div class="chord-row-time">' + seg.start + 's</div>';
      tl.appendChild(row);
    });

    if (res) res.style.display = 'block';
    chordDropLbl('✓ ' + file.name + ' · <span>upload another</span>');
    toast('Chord ID — ' + (d.chord_list || []).length + ' chords · ' + d.key);

    // Reset file input so same file can be re-uploaded
    var inp = document.getElementById('chord-file-input');
    if (inp) inp.value = '';

  } catch(e) {
    if (proc) proc.style.display = 'none';
    chordDropLbl('⚠ Failed · <span>try again</span>');
    toast('Chord ID failed: ' + e.message, true);
  }
}

// ── Init ──────────────────────────────────────────────────────────────────────
// Theme — default light
(function() {
  // Light is the HTML default — only switch to dark if explicitly saved
  if (localStorage.getItem('explore_theme') === 'dark') {
    document.body.classList.remove('light');
  }
  var btn = document.getElementById('theme-btn');
  if (btn) btn.textContent = document.body.classList.contains('light') ? '◑' : '◐';
})();
loadState();
drawKnob();
fetchGain();
setInterval(fetchGain, 2000);
ensureScanned();
// Size sliders after layout is painted
setTimeout(resizeGBSliders, 80);
setTimeout(resizeGBSliders, 400);

// Chord ID — wire file input via addEventListener (more reliable than onchange attr)
(function() {
  var inp  = document.getElementById('chord-file-input');
  var zone = document.getElementById('chord-drop-zone');
  if (inp) {
    inp.addEventListener('change', function() {
      if (this.files && this.files[0]) runChordID(this.files[0]);
    });
  }
  if (zone) {
    zone.addEventListener('click', function() {
      var i = document.getElementById('chord-file-input');
      if (i) i.click();
    });
  }
})();
</script>
</body>
</html>"""

if __name__ == "__main__":
    print(f"Explore running at http://127.0.0.1:{PORT}")
    app.run(host="0.0.0.0", port=PORT, debug=False, threaded=True)
