# Skateboard Trick Classifier

A project that watches a video of someone skateboarding and guesses
which trick they did: Ollie, Kickflip, Shove it, or whatever else you
teach it.

## The main idea

Most people would train a CNN on raw video for this. It works but it
needs a lot of labeled footage and a powerful GPU.

I went a simpler route instead. instead of feeding a model raw video, I use
Google's MediaPipe to track the skater's skeleton, turn that motion into a 
bunch of numbers, and train a regular scikit-learn classifier on those numbers. 
No GPU needed, runs fine on a laptop CPU.

## What it can do

- Give it a video file and it'll tell you which trick it thinks it
  saw, plus how confident it is.
- Record a trick straight from your webcam. Hit a space to start, hit it
  again to stop, and it predicts then replays the clip with the skeleton 
  drawn on top so you can see what it saw.
- Add new clips, or a whole new trick type, just by dropping video files
  into a folder.
- One menu (`menu.py`) ties everything together so you don't have to
  remember which script does what.

## How it works

1. Pull pose landmarks out of every frame with MediaPipe.
2. Normalize those joint positions so camera distance/angle don't skew
   things.
3. Turn the motion over time into a fixed set of numbers per clip (around
   90 features), things like foot spread, how symmetric the movement is,
   rotation angles, etc.
4. Save those numbers plus a label into a CSV.
5. Train a Random Forest on that CSV.
6. Feed it a new video and it spits out a predicted trick with a
   confidence score.

## Tech stack

Python 3, OpenCV for video, MediaPipe for pose tracking, pandas/NumPy for
the number crunching, scikit-learn for the actual classifier, and joblib
to save/load the trained model.

## What you need

- Python 3.8 to 3.11. The mediapipe version this project needs doesn't
  have a build for 3.12+, so stick to that range. Check with
  `python3 --version`.
- A webcam if you want to use the live recording feature.

## Setting it up

1. Grab a copy of this folder.
2. Run the installer:

   ```bash
   python3 install.py
   ```

   This makes a virtual environment (`venv/`) and installs everything
   into it, so it won't mess with anything else on your computer.

3. Activate it:

   ```bash
   source venv/bin/activate        # macOS / Linux
   venv\Scripts\activate           # Windows
   ```

4. Run it:

   ```bash
   python3 menu.py                 # macOS / Linux
   python menu.py                  # Windows
   ```

You'll need to activate the environment again every time you open a new
terminal.

## Using it

Just run `menu.py` and pick a number:

1. **Analyze a video**, predicts the trick in a video file you already
   have.
2. **Use webcam**, record a trick live and get a prediction right away.
3. **Add a new clip**, label a video (existing trick or a new
   one) and drop it into the `tricks/` folder.
4. **Rebuild dataset & retrain**, regenerates everything from whatever's
   currently sitting in `tricks/`.

### Adding a new trick

Make a folder inside `tricks/` named after the trick, put some clips in it, 
and rebuild. The folder name becomes the label automatically.

```
tricks/
    Ollie/
        clip1.mp4
        clip2.mp4
    Kickflip/
        clip1.mp4
    Heelflip/            <- this folder alone is the whole "add a trick" step
        clip1.mp4
```

## What each file does

```
menu.py                    the menu, start here
phase1_pose_extraction.py  standalone demo showing the pose skeleton on a video
build_dataset.py           scans tricks/<TrickName>/*.mp4 and builds trick_dataset.csv
train_classifier.py        trains trick_classifier.joblib from trick_dataset.csv
predict_trick.py           predicts the trick in a given video file
show_prediction.py         predicts, then replays the video with the skeleton + result
webcam_predict.py          record from your webcam and get a live prediction
quiet_mediapipe.py         small helper that silences MediaPipe's noisy startup logs
tricks/                    the training videos, one folder per trick
requirements.txt           the pinned dependencies
install.py                 sets up the environment (see setup steps above)
```

## If something breaks

- **`ModuleNotFoundError`** for cv2, mediapipe, etc: you're probably
  running your system Python instead of the venv one. Make sure you
  activated the environment first in this terminal.
- **Camera or screen recording permission popups on macOS**: totally
  normal the first time you use the webcam feature. Just allow them and
  run it again.
- **Window won't open / GL errors on Linux**: some minimal Linux setups
  are missing a system graphics library that OpenCV needs. Try
  `sudo apt install libgl1` on Debian/Ubuntu.
