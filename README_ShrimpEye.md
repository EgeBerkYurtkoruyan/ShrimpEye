# ShrimpEye

A vision-only object-following robot for the **PiCar-4WD** + **Raspberry Pi camera**.

ShrimpEye detects an object that you place in front of its camera, locks onto it,
and follows it around an indoor room while trying to keep a roughly constant
distance. It uses only the camera (no ultrasonic/depth sensor): the distance is
judged from how large the object looks in the image.

- **Author:** Ege Berk Yurtkoruyan (student number 4593014)
- **Course:** Robotics, LIACS, Universiteit Leiden
- **Demo video:** https://REPLACE-WITH-YOUR-VIDEO-LINK
- **Report (PDF):** see `ShrimpEye_Technical_Report.pdf` in this repository

---

## How it works (short version)

The program is a state machine with three states:

1. **ACQUIRE** – the robot stands still and uses background subtraction to spot a
   newly introduced object. When the object is held still for ~1.2 s, it locks on.
2. **TRACK** – on lock, the object's colour is learned in HSV. From then on the
   object is found by colour over the whole frame, and a position gate and a
   size gate keep the lock on the right object. The robot drives/turns to keep
   it centred and at the locked distance (judged from the object's diameter).
3. **LOST** – if the object leaves view, the robot stops, takes a fresh
   background, and returns to ACQUIRE.

See the technical report for the full design and the reasoning behind each step.

---

## Hardware

- SunFounder **PiCar-4WD** chassis (assembled, motors connected)
- **Raspberry Pi** (Pi 4 recommended) with the PiCar-4WD HAT
- **Raspberry Pi Camera** connected and enabled
- Charged battery pack

## Software requirements

- Raspberry Pi OS (with the camera enabled)
- Python 3
- Python packages:
  - `opencv-python`
  - `numpy`
  - `picamera2`
  - `picar_4wd` (SunFounder library for the PiCar-4WD)

A display (or VNC) is recommended so you can see the two debug windows. To run
without a display, set `SHOW_DEBUG = False` near the top of `ShrimpEye.py`.

## Installation

```bash
# 1. Get the code
git clone https://REPLACE-WITH-YOUR-CODE-LINK
cd shrimpeye

# 2. Install Python dependencies
pip3 install opencv-python numpy

# picamera2 is usually pre-installed on Raspberry Pi OS. If not:
sudo apt update
sudo apt install -y python3-picamera2

# 3. Install the PiCar-4WD library (follow SunFounder's instructions)
#    https://github.com/sunfounder/picar-4wd
```

## Running

```bash
python3 ShrimpEye.py
```

### Startup procedure

1. Start the script with **no object in view**.
2. Wait for the terminal to print **"Background captured."**
3. Place your object in front of the camera and **hold it still** for about
   1.5 seconds.
4. When the terminal prints **"LOCKED"**, start moving the object – the robot
   will follow it and try to keep the same distance it had at lock time.
5. **Remove the object** from view: the robot stops and waits for a new one.
6. Press **ESC** in the debug window, or **Ctrl-C** in the terminal, to quit.

### What you should see

- A **"PiCar tracker"** window: the live camera view with a green box on the
  locked object, a small red centre dot, and a yellow circle showing the
  measured diameter used for distance control.
- A **"mask"** window: the binary mask. In ACQUIRE it shows the
  background-difference mask; in TRACK it shows the colour mask.

---

## Choosing a good object

For the most reliable behaviour:

- Use a **single, distinct, matte colour** that does **not** appear elsewhere in
  the room or on the floor.
- Avoid **shiny / glossy** objects – highlights leave holes in the colour mask.
- Keep the **lighting reasonably even**. The camera's automatic white balance is
  left on, so colours can shift a little while the robot moves; an evenly lit
  room reduces this.

---

## Tuning

All tunable values are constants near the top of `ShrimpEye.py`, with comments.
The ones you are most likely to touch:

| Parameter | Effect |
|---|---|
| `HUE_TOLERANCE` | Width of the accepted colour band (+/- around the locked hue). Widen if the object drops out; narrow if the background bleeds in. |
| `CHROMA_SAT_LOWER`, `CHROMA_VAL_LOWER` | How washed-out / dark a pixel may be and still count as the object. |
| `BASE_GATE_PX`, `MAX_GATE_PX` | Size of the position gate. Smaller = stricter (less chance of grabbing the wrong object), but a fast object can escape it. |
| `SIZE_GATE_LO`, `SIZE_GATE_HI` | How different in size a blob may be from the tracked object before it is rejected. |
| `DEADZONE_RATIO` | How tightly the robot holds the distance. Larger = calmer. |
| `SAFETY_RATIO` | How close the object may get before the robot backs up urgently. |
| `FORWARD/DRIVE/TURN` speed and gain values | Overall speed and how aggressively it reacts. |
| `ACQUIRE_TIMEOUT_SECONDS` | How long it waits on an empty view before refreshing the background. |

The live debug text (`diam=... target=...`) makes it easy to see what the
controller is doing while you tune.

---

## Troubleshooting

- **The robot never drives backward:** confirm the library has a `backward` (or
  `back`) function: `python3 -c "import picar_4wd as fc; print([n for n in dir(fc) if 'back' in n.lower()])"`.
- **The mask gets noisy while moving:** the camera's automatic white balance is
  shifting the colours. Use a more distinct, matte object and more even
  lighting. (Locking the camera's white balance/exposure is listed as future
  work in the report.)
- **It grabs the wrong object:** reduce `MAX_GATE_PX` and/or narrow
  `HUE_TOLERANCE`, and pick an object whose colour is more distinct from the
  room.
- **It loses the object in shadow:** lower `CHROMA_VAL_LOWER` so darker shades of
  the object still count.

---

## Files

- `ShrimpEye.py` – the complete robot program (single file).
- `ShrimpEye_Technical_Report.pdf` – the technical report.
- `README.md` – this file.
