# Tabletop Collage

- `python3 webcam.py`: saves a frame from a Raspberry Pi webcam to `data/calibration.jpg`
- `python3 calibrate.py data/calibration.jpg`: manually calibrate to establish
  homography between webcam and tabletop. Persisted at `data/cal2table.npy`
- `python3 match.py data/calibration.jpg data/screenshot.jpg`: automatically
  establishes homography between webcam and screenshot. Persisted at
  `data/shot2cal.npy`
- `python3 render.py` displays `calibration.jpg` and `screenshot.jpg` on table
