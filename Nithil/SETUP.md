# Setup (Windows)

Goal of this page: by the end of Day 1 you have generated a 10-second
soccer video on your own laptop. That's it.

**Important:** the soccer simulator (`gfootball`) is best supported on
Linux and Mac. On Windows you have two paths:

- **Path A (recommended): WSL2.** Run Linux inside Windows. The
  simulator and all our scripts work natively on Linux. This is the
  standard way professional ML/research devs work on Windows.
- **Path B (fallback): native Windows.** Use the precompiled
  `gfootball` package from PyPI. Sometimes works, sometimes weird.
  Try this only if Path A fails.

Path A takes ~1 extra hour to set up the first time but saves you days
of debugging weird errors later. Go with A unless I (Neha) tell you
otherwise.

You'll need ~6 GB of free disk space.

---

## Step 0: Reference clips (do this first while WSL installs)

Open the **Drive folder**:
https://drive.google.com/drive/folders/1XxiDJtDv7ffVHvAwExVfE7sq0no3m4vi?usp=sharing

Watch the videos in `reference_clips/`. There are four interesting
ones:

- `default_5v5.mov` — what a normal play looks like.
- `uniform_jerseys_5v5.mov` — same play, but both teams wear the same
  kit. The "team identification" cue is gone. (Spot the difference.)
- `ball_tiny/episode_done_*.mov` — the ball has been shrunk to 5% of
  its size. You won't see it. This is one of our inference probes.
- `compare_5v5/*.mov` — side-by-side baseline.

These are the kinds of stimuli we'll be generating thousands of. Watch
them now while WSL2 installs in the background. You'll come back to
"why does this matter" in Week 2.

You don't need to download anything else from Drive. Everything else
you'll generate yourself.

---

## Path A: WSL2 (recommended)

### Step 1: Install WSL2

Open **PowerShell as Administrator** (right-click the Start menu →
Windows Terminal (Admin)). Then:

```powershell
wsl --install -d Ubuntu-24.04
```

This installs Ubuntu inside Windows. Reboot when it asks. After reboot
Ubuntu will open automatically and ask you to make a username +
password. Pick anything you'll remember. The password won't show
characters when you type it — that's normal.

You're now in a Linux terminal inside Windows. Everything from here on
runs in this Ubuntu environment. To open it later: Start menu →
"Ubuntu".

### Step 2: Install Python and dependencies

In the Ubuntu terminal:

```bash
sudo apt update
sudo apt install -y python3.11 python3.11-venv python3-pip git \
    libsdl2-dev libsdl2-image-dev libsdl2-gfx-dev libsdl2-ttf-dev \
    libboost-all-dev cmake build-essential ffmpeg
```

This takes ~5 minutes. The `sudo` command will ask for your password.

### Step 3: Get the code

```bash
cd ~
git clone <repo-url-Neha-will-send-you>
cd new_sim
```

You can also access the WSL filesystem from VS Code on Windows by
installing the **WSL extension** and opening the folder from there. I
recommend this — VS Code on Windows + WSL is the best of both worlds.

### Step 4: Make a virtual environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

Your prompt should now start with `(.venv)`. You need to run
`source .venv/bin/activate` every time you open a new terminal.

### Step 5: Install gfootball and Python deps

```bash
pip install --upgrade pip setuptools wheel
pip install gfootball
pip install absl-py numpy six "gym==0.23.1" opencv-python psutil pygame
```

`pip install gfootball` will compile a bit of C++ in the background.
This step can take 5-15 minutes. If it fails, copy the error message
verbatim and send it to Neha.

### Step 6: Smoke test

```bash
cd experiments
python gen_description.py --n=2
```

You should see:

```
  2d7619893e  custom_1v1_base  seed=11  frame=10  L=2 R=2
  e56cb1b4e7  custom_1v1_base  seed=11  frame=30  L=2 R=2
wrote 2 description items → .../experiments/stimuli/description
```

Open the frame from Windows (WSL files are accessible at
`\\wsl$\Ubuntu-24.04\home\<you>\new_sim\...`):

```bash
explorer.exe stimuli/description/2d7619893e/
```

That opens a Windows Explorer window in the folder. Double-click
`frame.png` to view it. You should see a soccer pitch with two players.
**If you got this far, you're done with setup. 🎉**

### Step 7: Generate a video clip

```bash
python gen_prediction.py --n=1
explorer.exe stimuli/prediction/
```

Find the `clip.mov` inside and double-click. Windows Media Player or
the Movies & TV app will play it.

---

## Path B: Native Windows (skip unless Path A fails)

Only try this if WSL2 didn't work. Ping Neha before going down this
road; she may want to debug WSL with you first.

1. Install Python 3.9 from python.org. **Specifically 3.9** — newer
   versions don't have a precompiled `gfootball` wheel for Windows.
2. Open PowerShell:
   ```powershell
   cd ~/Desktop  # or wherever you want the project
   git clone <repo-url>
   cd new_sim
   py -3.9 -m venv .venv
   .venv\Scripts\Activate.ps1
   python -m pip install --upgrade pip setuptools wheel
   python -m pip install gfootball
   python -m pip install absl-py numpy six "gym==0.23.1" opencv-python psutil pygame
   ```
3. Install ffmpeg:
   - Download from https://www.gyan.dev/ffmpeg/builds/ (the
     "release essentials" build).
   - Unzip somewhere permanent. Add the `bin` folder to your PATH.
4. Smoke test the same as Step 6 above. Use `explorer .` instead of
   `explorer.exe .` to open Explorer.

If `python -m pip install gfootball` fails on Windows, it's because the
wheel doesn't exist for your Python version. Verify you're on 3.9.

---

## When things go wrong

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError: gfootball` | venv isn't activated. Run `source .venv/bin/activate` (WSL) or `.venv\Scripts\Activate.ps1` (Windows). |
| `ModuleNotFoundError: absl` | Same as above; install step skipped. |
| `Cannot locate gfootball.scenarios` | gfootball install failed silently. Re-run the install command and watch the output. |
| `ffmpeg not found` | Install ffmpeg (see Path B step 3 or `sudo apt install ffmpeg` on WSL). |
| Long SDL warnings on every run | Ignore. They look scary but don't break anything. |
| Video file opens but won't play | Try VLC instead of the default player. |

If you see something not on this list, copy the **entire error
message** into a note. Don't paraphrase. The exact wording matters.

---

## VS Code setup

Two extensions that make life much better:

1. **Python** (Microsoft) — code intelligence + debugger.
2. **WSL** (Microsoft) — only if you're using Path A. Lets you open the
   project folder *inside* WSL from Windows VS Code. After installing
   it, in WSL terminal run `code .` from the project root and VS Code
   will spawn connected to WSL.

When VS Code asks "Select Python Interpreter," pick the one inside your
venv (`.venv/bin/python` in WSL, `.venv\Scripts\python.exe` on
Windows). Now errors highlight live.

---

## What you don't need to do

You **do not** need to compile the simulator from C++ source. The
`pip install gfootball` command does that for you (or downloads a
prebuilt binary). If something there breaks, that's Neha's problem,
not yours.
