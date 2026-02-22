# liquidmotiond

Daemon-friendly GIF emulation for **NZXT Kraken 2023 / Kraken 2023 Elite LCD (firmware 2.x)** using `liquidctl`'s Python API.

This project works around current `liquidctl` limitations for Kraken 2023 fw2.x LCD GIF mode by:

- keeping a **persistent** `liquidctl` device connection
- pre-decoding GIFs into **in-memory RGB565 frames**
- sending frames through the Kraken 2023 fw2 static transfer path
- pacing playback using GIF frame durations (no busy loop)

It is designed to be **systemd-friendly** and supports:

- temp-based animation switching
- periodic temp card display
- graceful shutdown on `SIGTERM` / `Ctrl-C`
- resetting the LCD back to the default **liquid** mode on exit

---

## Why this exists

On Kraken 2023 firmware 2.x, `liquidctl` can set static images but does not support GIF mode directly for the LCD on these devices/firmware combinations.

This daemon emulates GIF playback by sending static frames efficiently.

---

## Features

- **Low CPU usage** compared to subprocess/CLI-per-frame approaches
- **Persistent USB session** (`liquidctl` Python API)
- **No frame temp files required** (in-memory pipeline)
- **GIF frame duration aware** playback
- **Frame dedupe** (optional) to reduce repeated transfers
- **Threshold-based GIF switching** by liquid temperature
- **Temp card overlay/splash** with configurable font size
- **Graceful stop/reset** behavior for service use

---

## Requirements

- Python 3.10+ (3.11+ recommended)
- `liquidctl` **1.15.x** (pinned strongly recommended)
- Pillow (`PIL`)
- Kraken 2023 / Kraken 2023 Elite LCD on firmware 2.x
- Linux with USB access permissions for `liquidctl`

### Python packages

```bash
pip install "liquidctl==1.15.*" pillow
