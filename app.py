import os, json, socket, time, threading
from pathlib import Path
from flask import Flask, jsonify, request, render_template_string
from dotenv import load_dotenv

load_dotenv()
app = Flask(__name__)
PORT = 5572

# ── Ableton socket ─────────────────────────────────────────────────────────────

def ableton_send(command_type, params=None):
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(8)
        s.connect(("localhost", 9877))
        msg = json.dumps({"type": command_type, "params": params or {}}).encode()
        s.sendall(msg)
        time.sleep(0.02)
        chunks = []
        s.settimeout(2)
        try:
            while True:
                chunk = s.recv(65536)
                if not chunk: break
                chunks.append(chunk)
        except socket.timeout:
            pass
        s.close()
        raw = b"".join(chunks)
        return json.loads(raw) if raw else {"ok": True}
    except Exception as e:
        return {"error": str(e)}

def get_session():
    return ableton_send("get_session_info").get("result", {})

def get_track(i):
    return ableton_send("get_track_info", {"track_index": i}).get("result", {})

def get_all_tracks():
    session = get_session()
    count = session.get("track_count", 0)
    tracks = []
    for i in range(min(count, 24)):
        t = get_track(i)
        if t:
            t["index"] = i
            tracks.append(t)
    return session, tracks

# ── Audio analysis ─────────────────────────────────────────────────────────────

def analyze_audio_file(path):
    try:
        import librosa
        import numpy as np
        import pyloudnorm as pyln
        import soundfile as sf

        y, sr = librosa.load(str(path), sr=None, mono=False)
        if y.ndim > 1:
            y_mono = librosa.to_mono(y)
            stereo_width = float(np.mean(np.abs(y[0] - y[1]))) / (float(np.mean(np.abs(y[0] + y[1]))) + 1e-9)
        else:
            y_mono = y
            stereo_width = 0.0

        peak = float(np.max(np.abs(y_mono)))
        rms  = float(np.sqrt(np.mean(y_mono**2)))

        try:
            data, rate = sf.read(str(path))
            meter = pyln.Meter(rate)
            lufs = meter.integrated_loudness(data if data.ndim > 1 else data.reshape(-1,1))
        except Exception:
            lufs = None

        stft  = np.abs(librosa.stft(y_mono))
        freqs = librosa.fft_frequencies(sr=sr)
        power = np.mean(stft**2, axis=1)

        def band_energy(lo, hi):
            mask = (freqs >= lo) & (freqs < hi)
            return float(np.sum(power[mask])) / (float(np.sum(power)) + 1e-9)

        spectral_centroid = float(np.mean(librosa.feature.spectral_centroid(y=y_mono, sr=sr)))
        dynamic_range     = float(20 * np.log10(peak / (rms + 1e-9))) if rms > 0 else 0

        return {
            "peak_db":              round(20 * np.log10(peak + 1e-9), 1),
            "rms_db":               round(20 * np.log10(rms  + 1e-9), 1),
            "lufs":                 round(lufs, 1) if lufs is not None else None,
            "dynamic_range_db":     round(dynamic_range, 1),
            "stereo_width":         round(stereo_width, 3),
            "spectral_centroid_hz": round(spectral_centroid, 0),
            "sub_energy":           round(band_energy(20, 80),     3),
            "bass_energy":          round(band_energy(80, 250),    3),
            "low_mid_energy":       round(band_energy(250, 600),   3),
            "mid_energy":           round(band_energy(600, 2500),  3),
            "high_mid_energy":      round(band_energy(2500, 8000), 3),
            "air_energy":           round(band_energy(8000, 20000),3),
        }
    except Exception as e:
        return {"error": str(e)}

def find_project_audio(project_path):
    p = Path(project_path)
    audio_files = []
    for ext in ["*.wav", "*.aif", "*.aiff", "*.mp3", "*.flac"]:
        audio_files.extend(p.rglob(ext))
    return audio_files[:24]

# ── Problem detectors ──────────────────────────────────────────────────────────

def detect_problems(tracks, audio_data):
    problems = []

    muddy = [t["name"] for t in tracks
             if audio_data.get(t.get("name",""), {}).get("low_mid_energy", 0) > 0.25]
    if len(muddy) >= 2:
        problems.append({
            "type": "mud", "severity": "high" if len(muddy) >= 3 else "medium",
            "title": "Low-Mid Buildup (Mud)",
            "detail": f"{len(muddy)} tracks competing between 250–600Hz: {', '.join(muddy[:4])}.",
            "fix": "Cut 2–4dB around 300Hz on the thickest offenders — usually pad or rhythm guitar."
        })

    low_heavy = [t["name"] for t in tracks
                 if (audio_data.get(t.get("name",""), {}).get("sub_energy", 0)
                   + audio_data.get(t.get("name",""), {}).get("bass_energy", 0)) > 0.45]
    if len(low_heavy) >= 2:
        problems.append({
            "type": "low_end_conflict", "severity": "high",
            "title": "Low End Conflict",
            "detail": f"{', '.join(low_heavy[:3])} are competing below 250Hz.",
            "fix": "High-pass everything except kick and bass. Side-chain compress bass to kick."
        })

    hot = [t["name"] for t in tracks
           if audio_data.get(t.get("name",""), {}).get("peak_db", -99) > -1]
    if hot:
        problems.append({
            "type": "clipping", "severity": "high",
            "title": "Hot Levels / Clipping Risk",
            "detail": f"{', '.join(hot[:3])} peaking near or above 0dBFS.",
            "fix": "Pull these down. Nothing should clip before the master bus."
        })

    mono_heavy = [t["name"] for t in tracks
                  if audio_data.get(t.get("name",""), {}).get("stereo_width", 1) < 0.05
                  and audio_data.get(t.get("name",""), {}).get("peak_db", -99) > -30]
    if len(mono_heavy) >= 3:
        problems.append({
            "type": "narrow_mix", "severity": "low",
            "title": "Narrow Stereo Image",
            "detail": "Most tracks are mono or near-mono. Mix may feel flat.",
            "fix": "Widen pads and guitars with stereo delay or chorus. Keep kick, bass, lead vocal mono."
        })

    if not problems:
        problems.append({
            "type": "healthy", "severity": "none",
            "title": "No Major Issues Detected",
            "detail": "Session looks clean from analysis.", "fix": ""
        })

    return problems

# ── Claude ─────────────────────────────────────────────────────────────────────

def ask_claude(system, user):
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        msg = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=1024,
            system=system,
            messages=[{"role": "user", "content": user}]
        )
        return msg.content[0].text, msg.usage.output_tokens
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
        audio_files = find_project_audio(project_path)
        for f in audio_files:
            result = analyze_audio_file(f)
            if "error" not in result:
                audio_data[f.stem] = result

    problems = detect_problems(tracks, audio_data)

    track_lines = []
    for t in tracks:
        kind  = "MIDI" if t.get("is_midi_track") else "Audio"
        devs  = ", ".join(d["name"] for d in t.get("devices", []) if d.get("name"))
        adat  = audio_data.get(t.get("name",""), {})
        aline = ""
        if adat and "peak_db" in adat:
            aline = (f" [peak:{adat['peak_db']}dB rms:{adat['rms_db']}dB "
                     f"lufs:{adat.get('lufs','?')} width:{adat.get('stereo_width','?')} "
                     f"centroid:{adat.get('spectral_centroid_hz','?')}Hz "
                     f"sub:{adat.get('sub_energy','?')} bass:{adat.get('bass_energy','?')} "
                     f"low_mid:{adat.get('low_mid_energy','?')} mid:{adat.get('mid_energy','?')}]")
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
    })

@app.route("/api/ask", methods=["POST"])
def api_ask():
    d = request.get_json() or {}
    prompt      = d.get("prompt", "")
    session_ctx = d.get("session_ctx", "")
    problem_ctx = d.get("problem_ctx", "")

    system = """You are Explore — an expert mix engineer and producer mentor inside Ableton Live.

You have live session data: track names, devices, and audio analysis metrics (peak, LUFS, stereo width, frequency band energies).

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

    user = f"[SESSION]\n{session_ctx}\n\n[DETECTED ISSUES]\n{problem_ctx}\n\n[QUESTION]\n{prompt}"
    text, tokens = ask_claude(system, user)
    return jsonify({"text": text, "tokens": tokens})

@app.route("/api/fix", methods=["POST"])
def api_fix():
    d = request.get_json() or {}
    action = d.get("action", "")
    params = d.get("params", {})
    if not action:
        return jsonify({"error": "No action"}), 400
    return jsonify(ableton_send(action, params))

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
<style>
*{box-sizing:border-box;margin:0;padding:0;}
:root{
  --bg:#030507;--panel:#060A0F;--panel2:#0A1018;--border:#162030;--border2:#1E2E40;
  --text:#D8EAF8;--dim:#405870;--teal:#009690;--purple:#7B2FD4;--gold:#C8A843;
  --red:#C84030;--green:#28A060;--orange:#C87030;
}
body{background:var(--bg);color:var(--text);font-family:'Inter',system-ui,sans-serif;font-size:12px;min-height:100vh;display:flex;flex-direction:column;}
.hdr{display:flex;align-items:center;gap:14px;padding:0 20px;height:52px;border-bottom:1px solid var(--border);background:rgba(3,5,7,.97);flex-shrink:0;position:sticky;top:0;z-index:10;}
.brand{font-size:16px;font-weight:900;letter-spacing:4px;color:var(--teal);}
.brand-sub{font-size:9px;font-weight:700;letter-spacing:.2em;color:var(--dim);text-transform:uppercase;}
.hdr-right{margin-left:auto;display:flex;gap:7px;align-items:center;}
.pill{padding:5px 11px;border-radius:20px;font-size:9px;font-weight:800;letter-spacing:.12em;text-transform:uppercase;border:1px solid var(--border2);background:transparent;color:var(--dim);cursor:pointer;transition:all .2s;}
.pill:hover{border-color:var(--teal);color:var(--teal);}
.pill.primary{background:var(--teal);color:#000;border-color:var(--teal);}
.pill.primary:hover{opacity:.85;}
.status-dot{width:7px;height:7px;border-radius:50%;background:var(--red);flex-shrink:0;}
.status-dot.on{background:var(--green);}
.main{display:flex;flex:1;overflow:hidden;height:calc(100vh - 52px);}
.sidebar{width:220px;border-right:1px solid var(--border);display:flex;flex-direction:column;flex-shrink:0;}
.panel-hdr{padding:8px 12px;font-size:9px;font-weight:800;letter-spacing:.2em;color:var(--dim);text-transform:uppercase;border-bottom:1px solid var(--border);display:flex;align-items:center;justify-content:space-between;}
.track-list{flex:1;overflow-y:auto;}
.track-row{padding:7px 12px;border-bottom:1px solid var(--border);display:flex;align-items:center;gap:8px;cursor:pointer;transition:background .1s;}
.track-row:hover{background:var(--panel2);}
.track-row.selected{background:rgba(0,150,144,.07);border-left:2px solid var(--teal);}
.track-num{font-size:9px;color:var(--dim);width:16px;flex-shrink:0;font-weight:700;}
.track-name{flex:1;font-size:11px;font-weight:600;overflow:hidden;white-space:nowrap;text-overflow:ellipsis;}
.track-badge{font-size:8px;padding:1px 4px;border-radius:2px;font-weight:700;flex-shrink:0;}
.badge-midi{background:rgba(0,150,144,.15);color:var(--teal);}
.badge-audio{background:rgba(123,47,212,.15);color:var(--purple);}
.center{flex:1;display:flex;flex-direction:column;overflow:hidden;border-right:1px solid var(--border);}
.problems-area{padding:14px;border-bottom:1px solid var(--border);overflow-y:auto;max-height:260px;}
.prob-card{background:var(--panel2);border:1px solid var(--border2);border-left:3px solid var(--border2);border-radius:6px;padding:10px 12px;margin-bottom:7px;}
.prob-card.high{border-left-color:var(--red);}
.prob-card.medium{border-left-color:var(--orange);}
.prob-card.low{border-left-color:var(--gold);}
.prob-card.none{border-left-color:var(--green);}
.prob-title{font-size:11px;font-weight:800;margin-bottom:3px;}
.prob-detail{font-size:10px;color:var(--dim);margin-bottom:5px;line-height:1.5;}
.prob-fix{font-size:10px;color:var(--teal);line-height:1.5;}
.chat-area{flex:1;overflow-y:auto;padding:14px;display:flex;flex-direction:column;gap:10px;}
.msg{display:flex;flex-direction:column;gap:3px;max-width:92%;}
.msg.user{align-self:flex-end;}
.msg.ai{align-self:flex-start;}
.msg-bubble{padding:9px 13px;border-radius:10px;font-size:11px;line-height:1.6;white-space:pre-wrap;}
.msg.user .msg-bubble{background:rgba(0,150,144,.12);border:1px solid rgba(0,150,144,.2);color:var(--text);}
.msg.ai .msg-bubble{background:var(--panel2);border:1px solid var(--border2);color:var(--text);}
.msg-meta{font-size:9px;color:var(--dim);padding:0 4px;}
.input-row{padding:10px;border-top:1px solid var(--border);display:flex;gap:7px;flex-shrink:0;}
.chat-input{flex:1;background:var(--panel2);border:1px solid var(--border2);border-radius:7px;padding:9px 12px;color:var(--text);font-size:11px;font-family:inherit;resize:none;height:42px;transition:border-color .15s;}
.chat-input:focus{outline:none;border-color:var(--teal);}
.send-btn{padding:0 14px;border-radius:7px;background:var(--teal);border:none;color:#000;font-size:10px;font-weight:800;letter-spacing:.1em;cursor:pointer;}
.send-btn:hover{opacity:.85;}
.send-btn:disabled{opacity:.35;cursor:not-allowed;}
.right-panel{width:250px;display:flex;flex-direction:column;flex-shrink:0;}
.scan-box{padding:12px;border-bottom:1px solid var(--border);}
.scan-label{font-size:9px;font-weight:800;letter-spacing:.15em;color:var(--dim);text-transform:uppercase;margin-bottom:7px;display:block;}
.path-input{width:100%;background:var(--panel2);border:1px solid var(--border2);border-radius:5px;padding:7px 9px;color:var(--text);font-size:10px;font-family:inherit;margin-bottom:7px;}
.path-input:focus{outline:none;border-color:var(--teal);}
.scan-btn{width:100%;padding:8px;border-radius:5px;background:var(--purple);border:none;color:#fff;font-size:10px;font-weight:800;letter-spacing:.1em;cursor:pointer;text-transform:uppercase;}
.scan-btn:hover{opacity:.85;}
.scan-btn:disabled{opacity:.35;}
.audio-list{flex:1;overflow-y:auto;padding:8px;}
.audio-row{background:var(--panel2);border:1px solid var(--border2);border-radius:5px;padding:8px 10px;margin-bottom:6px;}
.audio-name{font-size:10px;font-weight:700;margin-bottom:4px;overflow:hidden;white-space:nowrap;text-overflow:ellipsis;}
.audio-stats{display:grid;grid-template-columns:1fr 1fr;gap:2px;}
.audio-stat{font-size:8px;color:var(--dim);}
.audio-stat span{color:var(--text);font-weight:700;}
.empty{display:flex;flex-direction:column;align-items:center;justify-content:center;height:100%;color:var(--dim);gap:8px;padding:20px;text-align:center;font-size:11px;}
.empty-icon{font-size:26px;opacity:.3;}
::-webkit-scrollbar{width:3px;}
::-webkit-scrollbar-track{background:transparent;}
::-webkit-scrollbar-thumb{background:var(--border2);border-radius:2px;}
.toast{position:fixed;bottom:16px;left:50%;transform:translateX(-50%);background:var(--teal);color:#000;padding:8px 18px;border-radius:6px;font-size:11px;font-weight:800;opacity:0;transition:opacity .2s;pointer-events:none;z-index:999;}
.toast.show{opacity:1;}
</style>
</head>
<body>

<div class="hdr">
  <div>
    <div class="brand">EXPLORE</div>
    <div class="brand-sub">AI Mix Engineer</div>
  </div>
  <div id="ableton-dot" class="status-dot" title="Ableton"></div>
  <div class="hdr-right">
    <button class="pill primary" onclick="runScan()">⟳ Scan Session</button>
    <button class="pill" onclick="quickPrompt('Why does this mix sound muddy?')">Mud?</button>
    <button class="pill" onclick="quickPrompt('What is taking up the most space in this mix?')">Space?</button>
    <button class="pill" onclick="quickPrompt('What should I work on first?')">Priority?</button>
    <button class="pill" onclick="quickPrompt('Why does the chorus feel weak?')">Chorus?</button>
  </div>
</div>

<div class="main">

  <div class="sidebar">
    <div class="panel-hdr">Tracks <span id="track-count-lbl" style="color:var(--text)"></span></div>
    <div class="track-list" id="track-list">
      <div class="empty"><div class="empty-icon">🎚</div><div>Scan session to load tracks</div></div>
    </div>
  </div>

  <div class="center">
    <div class="problems-area" id="problems-area">
      <div class="empty" style="height:80px;">Run a scan to detect mix problems</div>
    </div>
    <div class="chat-area" id="chat-area">
      <div class="msg ai">
        <div class="msg-bubble">👋 I'm Explore — your AI mix engineer.\n\nScan your Ableton session above, then ask me anything:\n\n• "Why does this sound muddy?"\n• "What should I work on first?"\n• "Why doesn't the chorus hit?"\n• "What's competing with the vocal?"\n\nPoint me at your project folder for deep audio analysis.</div>
      </div>
    </div>
    <div class="input-row">
      <textarea class="chat-input" id="chat-input" placeholder="Ask about your mix..."></textarea>
      <button class="send-btn" id="send-btn" onclick="sendMessage()">Ask</button>
    </div>
  </div>

  <div class="right-panel">
    <div class="scan-box">
      <label class="scan-label">Project Folder (optional)</label>
      <input class="path-input" id="project-path" placeholder="/Users/you/Music/MyProject" type="text">
      <button class="scan-btn" id="scan-audio-btn" onclick="scanAudio()">Scan Audio Files</button>
    </div>
    <div class="audio-list" id="audio-list">
      <div class="empty"><div class="empty-icon">🎵</div><div>Point to your Ableton project folder for audio analysis</div></div>
    </div>
  </div>

</div>

<div class="toast" id="toast"></div>

<script>
let sessionCtx = '';
let problemCtx = '';

async function runScan() {
  toast('Scanning session...');
  try {
    const r = await fetch('/api/analyze', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({project_path: document.getElementById('project-path').value.trim()})
    });
    const d = await r.json();
    document.getElementById('ableton-dot').classList.add('on');
    sessionCtx = d.session_ctx || '';
    problemCtx = d.problem_ctx || '';
    renderTracks(d.tracks || []);
    renderProblems(d.problems || []);
    renderAudio(d.audio_data || {});
    toast('Loaded ' + (d.tracks || []).length + ' tracks');
  } catch(e) {
    document.getElementById('ableton-dot').classList.remove('on');
    toast('Cannot reach Ableton', true);
  }
}

async function scanAudio() {
  const path = document.getElementById('project-path').value.trim();
  if (!path) { toast('Enter project folder path first', true); return; }
  const btn = document.getElementById('scan-audio-btn');
  btn.disabled = true; btn.textContent = 'Scanning...';
  await runScan();
  btn.disabled = false; btn.textContent = 'Scan Audio Files';
}

function renderTracks(tracks) {
  document.getElementById('track-count-lbl').textContent = tracks.length;
  const el = document.getElementById('track-list');
  if (!tracks.length) { el.innerHTML = '<div class="empty"><div>No tracks found</div></div>'; return; }
  el.innerHTML = tracks.map((t, i) => {
    const b = t.is_midi_track
      ? '<span class="track-badge badge-midi">M</span>'
      : '<span class="track-badge badge-audio">A</span>';
    return '<div class="track-row" onclick="selectTrack(' + i + ')">' +
      '<span class="track-num">' + (i+1) + '</span>' +
      '<span class="track-name" title="' + (t.name||'') + '">' + (t.name||'Track '+(i+1)) + '</span>' +
      b + '</div>';
  }).join('');
}

function renderProblems(problems) {
  const el = document.getElementById('problems-area');
  if (!problems.length) { el.innerHTML = '<div class="empty" style="height:60px">No problems detected</div>'; return; }
  el.innerHTML = problems.map(p =>
    '<div class="prob-card ' + p.severity + '">' +
    '<div class="prob-title">' + p.title + '</div>' +
    '<div class="prob-detail">' + p.detail + '</div>' +
    (p.fix ? '<div class="prob-fix">→ ' + p.fix + '</div>' : '') +
    '</div>'
  ).join('');
}

function renderAudio(audioData) {
  const el = document.getElementById('audio-list');
  const keys = Object.keys(audioData).filter(k => audioData[k] && audioData[k].peak_db !== undefined);
  if (!keys.length) {
    el.innerHTML = '<div class="empty"><div class="empty-icon">🎵</div><div>No audio files analyzed</div></div>';
    return;
  }
  el.innerHTML = keys.map(name => {
    const d = audioData[name];
    return '<div class="audio-row">' +
      '<div class="audio-name" title="' + name + '">' + name + '</div>' +
      '<div class="audio-stats">' +
      '<div class="audio-stat">Peak <span>' + d.peak_db + 'dB</span></div>' +
      '<div class="audio-stat">LUFS <span>' + (d.lufs || '—') + '</span></div>' +
      '<div class="audio-stat">RMS <span>' + d.rms_db + 'dB</span></div>' +
      '<div class="audio-stat">Width <span>' + d.stereo_width + '</span></div>' +
      '<div class="audio-stat">Sub <span>' + Math.round(d.sub_energy*100) + '%</span></div>' +
      '<div class="audio-stat">Bass <span>' + Math.round(d.bass_energy*100) + '%</span></div>' +
      '<div class="audio-stat">LMid <span>' + Math.round(d.low_mid_energy*100) + '%</span></div>' +
      '<div class="audio-stat">Mid <span>' + Math.round(d.mid_energy*100) + '%</span></div>' +
      '</div></div>';
  }).join('');
}

function selectTrack(i) {
  document.querySelectorAll('.track-row').forEach((r,j) => r.classList.toggle('selected', i===j));
}

function quickPrompt(p) {
  document.getElementById('chat-input').value = p;
  sendMessage();
}

async function sendMessage() {
  const input = document.getElementById('chat-input');
  const prompt = input.value.trim();
  if (!prompt) return;
  input.value = '';
  addMessage('user', prompt);
  const btn = document.getElementById('send-btn');
  btn.disabled = true;
  const thinking = addMessage('ai', 'Thinking...');
  try {
    const r = await fetch('/api/ask', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({prompt, session_ctx: sessionCtx, problem_ctx: problemCtx})
    });
    const d = await r.json();
    thinking.querySelector('.msg-bubble').textContent = d.text || 'No response.';
    thinking.querySelector('.msg-meta').textContent = (d.tokens || '') + ' tok';
  } catch(e) {
    thinking.querySelector('.msg-bubble').textContent = 'Error connecting to server.';
  }
  btn.disabled = false;
  document.getElementById('chat-area').scrollTop = 99999;
}

function addMessage(role, text) {
  const area = document.getElementById('chat-area');
  const div = document.createElement('div');
  div.className = 'msg ' + role;
  div.innerHTML = '<div class="msg-bubble">' + text + '</div><div class="msg-meta"></div>';
  area.appendChild(div);
  area.scrollTop = 99999;
  return div;
}

document.getElementById('chat-input').addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
});

let toastTimer;
function toast(msg, err=false) {
  const el = document.getElementById('toast');
  el.textContent = msg;
  el.style.background = err ? 'var(--red)' : 'var(--teal)';
  el.style.color = err ? '#fff' : '#000';
  el.classList.add('show');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => el.classList.remove('show'), 2500);
}

fetch('/api/session').then(r => r.json()).then(d => {
  if (d.tracks && d.tracks.length) {
    document.getElementById('ableton-dot').classList.add('on');
  }
}).catch(() => {});
</script>
</body>
</html>"""

if __name__ == "__main__":
    print(f"Explore running at http://127.0.0.1:{PORT}")
    app.run(host="127.0.0.1", port=PORT, debug=False, threaded=True)
