# Explore

Points at an Ableton Live session and tells you, in plain language, what is wrong with the
mix.

## What it looks at

**The session structure**, read directly: track count, routing, plugin chains, gain
staging, what is soloed or muted and probably shouldn't be.

**The audio itself**: level, dynamic range, spectral balance, where two elements are
fighting for the same frequency range.

Then it says so in sentences. Not a score, not a wall of meters. "Your kick and your bass
are both peaking around 60 hertz and the bass is losing" is a useful thing to read. A
number between 1 and 100 is not.

## Why plain language

Every analysis tool in this space renders a spectrogram and leaves the interpretation to
you. That is fine if you already know what you are looking at, and useless if you are the
person who needs the help. The whole design constraint here is that the output has to be
actionable by someone who cannot read a spectrogram.

## Running it

```bash
make run
```

Or double-click `launch.command`.

Python, Flask.
