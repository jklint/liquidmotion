#!/usr/bin/env python3
"""
liquidmotiond.py - Kraken 2023 fw2 GIF emulation daemon using liquidctl Python API
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

from PIL import Image, ImageDraw, ImageFont, ImageSequence
from liquidctl import find_liquidctl_devices

LOG = logging.getLogger("liquidmotiond")

LCD_SIZE = (240, 240)
DEFAULT_MIN_FRAME_MS = 33
DEFAULT_TEMP_POLL_INTERVAL = 2.0
DEFAULT_STATUS_CARD_SECONDS = 2.0


@dataclass(frozen=True)
class ThresholdRule:
    threshold_c: float
    gif_path: Path


@dataclass
class PreparedFrame:
    rgb565: bytes
    duration_s: float


@dataclass
class PreparedAnimation:
    source: Path
    frames: List[PreparedFrame]


class StopRequested(Exception):
    """Raised when service stop was requested."""


class ShutdownFlag:
    def __init__(self) -> None:
        self._stop = False

    def request_stop(self, signum: int, _frame) -> None:
        LOG.info("Received signal %s, stopping...", signum)
        self._stop = True

    def request(self) -> None:
        self._stop = True

    @property
    def stop(self) -> bool:
        return self._stop


class KrakenFw2Session:
    """
    Persistent liquidctl session for Kraken 2023 fw2.x using private driver methods.
    """

    def __init__(self, match: str = "kraken") -> None:
        self.match = match.lower()
        self.dev = None
        self.connected = False
        self._double_send_next_frame = True
        self._temp_key_candidates = ("Liquid temperature", "liquid temperature")

    def connect(self) -> None:
        devices = find_liquidctl_devices()
        candidates = [d for d in devices if self.match in d.description.lower()]
        if not candidates:
            raise RuntimeError(f"No liquidctl device found matching '{self.match}'")

        candidates.sort(
            key=lambda d: ("2023" not in d.description, "Elite" not in d.description, d.description)
        )
        self.dev = candidates[0]
        LOG.info("Using device: %s", self.dev.description)

        self.dev.connect()
        self.connected = True

        init_status = self.dev.initialize()
        if init_status:
            for key, value, unit in init_status:
                LOG.debug("init: %s=%s %s", key, value, unit)

        if not hasattr(self.dev, "_send_2023_data_fw2"):
            raise RuntimeError(
                "liquidctl driver missing expected private method '_send_2023_data_fw2'. "
                "Pin liquidctl==1.15.x."
            )

    def disconnect(self) -> None:
        if not self.connected or self.dev is None:
            return
        try:
            self.dev.disconnect()
        except Exception as exc:
            LOG.warning("Device disconnect failed: %s", exc)
        finally:
            self.connected = False

    def reset_display(self, settle_delay: float = 0.35) -> None:
        """
        Return LCD to default liquid display mode before exit.
        """
        if self.dev is None:
            return

        last_exc: Optional[Exception] = None

        for attempt in (1, 2):
            try:
                self.dev.set_screen("lcd", "liquid", None)
                LOG.info("LCD reset to 'liquid' mode (attempt %d)", attempt)
                time.sleep(settle_delay)
                return
            except TypeError:
                try:
                    self.dev.set_screen("lcd", "liquid")
                    LOG.info("LCD reset to 'liquid' mode (attempt %d)", attempt)
                    time.sleep(settle_delay)
                    return
                except Exception as exc:
                    last_exc = exc
                    LOG.warning(
                        "Failed to reset LCD to liquid mode (attempt %d): %s",
                        attempt,
                        exc,
                    )
                    time.sleep(0.15)
            except Exception as exc:
                last_exc = exc
                LOG.warning("Failed to reset LCD to liquid mode (attempt %d): %s", attempt, exc)
                time.sleep(0.15)

        try:
            LOG.warning("Trying reinitialize before final LCD reset...")
            self.dev.initialize()
            try:
                self.dev.set_screen("lcd", "liquid", None)
            except TypeError:
                self.dev.set_screen("lcd", "liquid")
            time.sleep(settle_delay)
            LOG.info("LCD reset to 'liquid' mode after reinitialize")
        except Exception as exc:
            LOG.warning("Final LCD reset attempt failed: %s", exc)
            if last_exc is not None:
                LOG.debug("Original reset error was: %s", last_exc)

    def set_brightness(self, brightness: int) -> None:
        if self.dev is None:
            return
        self.dev.set_screen("lcd", "brightness", int(brightness))

    def set_orientation(self, degrees: int) -> None:
        if self.dev is None:
            return
        if degrees not in (0, 90, 180, 270):
            raise ValueError("Orientation must be one of: 0, 90, 180, 270")
        self.dev.set_screen("lcd", "orientation", int(degrees))

    def get_liquid_temp_c(self) -> Optional[float]:
        if self.dev is None:
            return None
        try:
            status = self.dev.get_status()
        except Exception as exc:
            LOG.warning("get_status failed: %s", exc)
            return None

        for key, value, _unit in status:
            if str(key) in self._temp_key_candidates:
                try:
                    return float(value)
                except (TypeError, ValueError):
                    return None
        return None

    def send_rgb565_frame(self, frame_bytes: bytes) -> None:
        if self.dev is None:
            raise RuntimeError("Device not connected")

        header = [0x06, 0x00, 0x00, 0x00] + list(len(frame_bytes).to_bytes(4, "little"))

        sends = 2 if self._double_send_next_frame else 1
        for _ in range(sends):
            self.dev._send_2023_data_fw2(frame_bytes, header)

        self._double_send_next_frame = False

    def reinitialize(self) -> None:
        if self.dev is None:
            return
        LOG.warning("Reinitializing device...")
        self.dev.initialize()
        self._double_send_next_frame = True


def _rgb565_from_pil(img: Image.Image) -> bytes:
    rgb = img.convert("RGB")
    out = bytearray()
    for r, g, b in rgb.getdata():
        dr = r >> 3
        dg = g >> 2
        db = b >> 3
        out.append((dr << 3) + (dg >> 3))
        out.append(((dg & 0x7) << 5) + db)
    return bytes(out)


def _compose_opaque(frame: Image.Image, background=(0, 0, 0)) -> Image.Image:
    rgba = frame.convert("RGBA")
    base = Image.new("RGBA", rgba.size, background + (255,))
    composed = Image.alpha_composite(base, rgba)
    return composed.convert("RGB")


def _rotate_for_orientation(img: Image.Image, orientation_degrees: int) -> Image.Image:
    if orientation_degrees not in (0, 90, 180, 270):
        raise ValueError("Invalid orientation")
    return img.rotate(-orientation_degrees, expand=False)


def _load_font(font_path: Optional[Path], size: int):
    if font_path is not None:
        try:
            return ImageFont.truetype(str(font_path), size=size)
        except Exception as exc:
            LOG.warning("Failed to load font '%s': %s", font_path, exc)

    common_fonts = [
        "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
    ]
    for fp in common_fonts:
        try:
            return ImageFont.truetype(fp, size=size)
        except Exception:
            continue

    LOG.warning("No TTF font found; falling back to tiny default PIL font")
    return ImageFont.load_default()


def _ms_since(start: float) -> float:
    return (time.monotonic() - start) * 1000.0


def _log_timing(
    enabled: bool,
    label: str,
    elapsed_ms: float,
    warn_ms: float,
    extra: str = "",
) -> None:
    if not enabled:
        return

    msg = f"{label}: {elapsed_ms:.1f} ms"
    if extra:
        msg += f" | {extra}"

    if elapsed_ms >= warn_ms:
        LOG.warning(msg)
    else:
        LOG.debug(msg)


def prepare_gif_animation(
    gif_path: Path,
    orientation_degrees: int,
    min_frame_ms: int = DEFAULT_MIN_FRAME_MS,
    dedupe: bool = True,
) -> PreparedAnimation:
    img = Image.open(gif_path)
    frames: List[PreparedFrame] = []
    prev_bytes: Optional[bytes] = None

    for frame in ImageSequence.Iterator(img):
        duration_ms = int(frame.info.get("duration", img.info.get("duration", 100)) or 100)
        duration_ms = max(duration_ms, min_frame_ms)

        fr = _compose_opaque(frame)
        fr = fr.resize(LCD_SIZE, Image.Resampling.LANCZOS)
        fr = _rotate_for_orientation(fr, orientation_degrees)
        rgb565 = _rgb565_from_pil(fr)

        if dedupe and prev_bytes == rgb565 and frames:
            frames[-1].duration_s += duration_ms / 1000.0
            continue

        frames.append(PreparedFrame(rgb565=rgb565, duration_s=duration_ms / 1000.0))
        prev_bytes = rgb565

    if not frames:
        raise RuntimeError(f"No frames decoded from GIF: {gif_path}")

    return PreparedAnimation(source=gif_path, frames=frames)


def build_temp_card_image(
    temp_c: float,
    orientation_degrees: int,
    value_font_size: int = 72,
    label_font_size: int = 24,
    font_path: Optional[Path] = None,
) -> Image.Image:
    canvas = Image.new("RGB", LCD_SIZE, (0, 0, 0))
    draw = ImageDraw.Draw(canvas)

    font_value = _load_font(font_path, value_font_size)
    font_label = _load_font(font_path, label_font_size)

    label = "Liquid"
    value = f"{int(round(temp_c))} C"

    bbox_label = draw.textbbox((0, 0), label, font=font_label)
    bbox_value = draw.textbbox((0, 0), value, font=font_value)

    label_w = bbox_label[2] - bbox_label[0]
    label_h = bbox_label[3] - bbox_label[1]
    value_w = bbox_value[2] - bbox_value[0]
    value_h = bbox_value[3] - bbox_value[1]

    gap = max(8, label_font_size // 2)
    total_h = label_h + gap + value_h
    y0 = (LCD_SIZE[1] - total_h) // 2 - 8

    draw.text(
        ((LCD_SIZE[0] - label_w) // 2, y0),
        label,
        fill=(180, 180, 180),
        font=font_label,
    )
    draw.text(
        ((LCD_SIZE[0] - value_w) // 2, y0 + label_h + gap),
        value,
        fill=(255, 255, 255),
        font=font_value,
    )

    return _rotate_for_orientation(canvas, orientation_degrees)


def build_temp_card_rgb565(
    temp_c: float,
    orientation_degrees: int,
    value_font_size: int = 72,
    label_font_size: int = 24,
    font_path: Optional[Path] = None,
) -> bytes:
    return _rgb565_from_pil(
        build_temp_card_image(
            temp_c=temp_c,
            orientation_degrees=orientation_degrees,
            value_font_size=value_font_size,
            label_font_size=label_font_size,
            font_path=font_path,
        )
    )


def parse_thresholds(items: Optional[Sequence[Sequence[str]]]) -> List[ThresholdRule]:
    rules: List[ThresholdRule] = []
    if not items:
        return rules
    for threshold, path in items:
        rules.append(ThresholdRule(threshold_c=float(threshold), gif_path=Path(path).expanduser()))
    rules.sort(key=lambda r: r.threshold_c, reverse=True)
    return rules


def select_animation_path(
    temp_c: Optional[float],
    default_gif: Path,
    threshold_rules: Sequence[ThresholdRule],
) -> Path:
    if temp_c is None:
        return default_gif
    for rule in threshold_rules:
        if temp_c >= rule.threshold_c:
            return rule.gif_path
    return default_gif


def sleep_until(deadline: float, stopflag: ShutdownFlag) -> None:
    while True:
        now = time.monotonic()
        remaining = deadline - now
        if remaining <= 0:
            return
        if stopflag.stop:
            raise StopRequested()
        time.sleep(min(remaining, 0.25))


def configure_logging(verbose: bool, debug_timing: bool) -> None:
    level = logging.DEBUG if (verbose or debug_timing) else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )

    # Keep third-party chatter from crushing the terminal buffer.
    noisy_loggers = [
        "liquidctl",
        "liquidctl.driver",
        "usb",
        "usb.core",
        "usb.backend",
        "hid",
        "PIL",
    ]
    for name in noisy_loggers:
        if name != "liquidmotiond":
            logging.getLogger(name).setLevel(logging.WARNING)


def main() -> int:
    parser = argparse.ArgumentParser(description="Kraken 2023 fw2 GIF emulation daemon")
    parser.add_argument("default_gif", type=Path, help="Default GIF path")
    parser.add_argument(
        "--threshold-gif",
        action="append",
        nargs=2,
        metavar=("TEMP_C", "GIF_PATH"),
        help="Temperature threshold and GIF path; may be used multiple times",
    )
    parser.add_argument(
        "--orientation",
        type=int,
        default=0,
        choices=(0, 90, 180, 270),
        help="LCD orientation in degrees (default: 0)",
    )
    parser.add_argument(
        "--brightness",
        type=int,
        default=None,
        help="LCD brightness 0-100 (optional)",
    )
    parser.add_argument(
        "--match",
        type=str,
        default="kraken",
        help="Substring to match liquidctl device description (default: kraken)",
    )
    parser.add_argument(
        "--temp-poll-interval",
        type=float,
        default=DEFAULT_TEMP_POLL_INTERVAL,
        help="Seconds between liquid temperature polls (default: 2.0)",
    )
    parser.add_argument(
        "--show-temp-card-seconds",
        type=float,
        default=DEFAULT_STATUS_CARD_SECONDS,
        help="Show a temp card this many seconds when polling temp (default: 2.0, set 0 to disable)",
    )
    parser.add_argument(
        "--temp-font-size",
        type=int,
        default=72,
        help="Temperature value font size in points (default: 72)",
    )
    parser.add_argument(
        "--temp-label-font-size",
        type=int,
        default=24,
        help="Temperature label font size in points (default: 24)",
    )
    parser.add_argument(
        "--font-path",
        type=Path,
        default=None,
        help="Optional TTF font path for temp card text",
    )
    parser.add_argument(
        "--min-frame-ms",
        type=int,
        default=DEFAULT_MIN_FRAME_MS,
        help="Minimum per-frame duration in milliseconds (FPS safety cap; default: 33)",
    )
    parser.add_argument(
        "--no-dedupe",
        action="store_true",
        help="Disable deduplication of identical consecutive frames",
    )
    parser.add_argument(
        "--reinit-on-error",
        action="store_true",
        help="Attempt liquidctl initialize() on transfer/status errors",
    )
    parser.add_argument(
        "--tempdir-prefix",
        type=str,
        default="liquidmotiond-",
        help="Prefix for temp dir in system temp location (default: liquidmotiond-)",
    )
    parser.add_argument(
        "--reset-mode-on-exit",
        type=str,
        default="liquid",
        choices=("liquid", "none"),
        help="LCD mode to restore on exit (default: liquid)",
    )
    parser.add_argument(
        "--reset-settle-delay",
        type=float,
        default=0.35,
        help="Seconds to wait after exit LCD reset command (default: 0.35)",
    )
    parser.add_argument(
        "--debug-timing",
        action="store_true",
        help="Log timing for frame sends and temp polls without enabling raw third-party debug spam",
    )
    parser.add_argument(
        "--debug-timing-warn-ms",
        type=float,
        default=250.0,
        help="Warn when frame send or temp poll exceeds this many ms (default: 250)",
    )
    parser.add_argument(
        "--debug-frame-log-every",
        type=int,
        default=10,
        help="When --debug-timing is enabled, log every Nth frame (default: 10)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose logging for this script",
    )
    args = parser.parse_args()

    configure_logging(args.verbose, args.debug_timing)

    default_gif = args.default_gif.expanduser()
    if not default_gif.is_file():
        LOG.error("Default GIF not found: %s", default_gif)
        return 2

    threshold_rules = parse_thresholds(args.threshold_gif)
    for rule in threshold_rules:
        if not rule.gif_path.is_file():
            LOG.error("Threshold GIF not found: %s", rule.gif_path)
            return 2

    stopflag = ShutdownFlag()
    signal.signal(signal.SIGTERM, stopflag.request_stop)
    signal.signal(signal.SIGINT, stopflag.request_stop)

    with tempfile.TemporaryDirectory(prefix=args.tempdir_prefix) as tempdir:
        tempdir_path = Path(tempdir)
        LOG.debug("Using temp dir: %s", tempdir_path)

        session = KrakenFw2Session(match=args.match)
        animation_cache: dict[Path, PreparedAnimation] = {}
        current_animation_path: Optional[Path] = None
        current_animation: Optional[PreparedAnimation] = None

        last_temp_poll = 0.0
        frame_index = 0
        frame_counter = 0

        try:
            session.connect()
            session.set_orientation(args.orientation)
            if args.brightness is not None:
                session.set_brightness(int(args.brightness))

            def get_or_prepare_animation(gif_path: Path) -> PreparedAnimation:
                cached = animation_cache.get(gif_path)
                if cached is not None:
                    return cached
                LOG.info("Preparing animation: %s", gif_path)
                prepared = prepare_gif_animation(
                    gif_path=gif_path,
                    orientation_degrees=args.orientation,
                    min_frame_ms=args.min_frame_ms,
                    dedupe=not args.no_dedupe,
                )
                animation_cache[gif_path] = prepared
                total_duration = sum(f.duration_s for f in prepared.frames)
                LOG.info(
                    "Prepared %s: %d frames (%.2fs loop)",
                    gif_path.name,
                    len(prepared.frames),
                    total_duration,
                )
                return prepared

            current_animation_path = default_gif
            current_animation = get_or_prepare_animation(current_animation_path)

            while not stopflag.stop:
                now = time.monotonic()

                if now - last_temp_poll >= args.temp_poll_interval:
                    temp: Optional[float] = None
                    try:
                        poll_start = time.monotonic()
                        temp = session.get_liquid_temp_c()
                        poll_ms = _ms_since(poll_start)

                        _log_timing(
                            args.debug_timing,
                            "temp_poll",
                            poll_ms,
                            args.debug_timing_warn_ms,
                            extra=f"temp={temp!r}",
                        )

                        if temp is not None:
                            desired_path = select_animation_path(
                                temp_c=temp,
                                default_gif=default_gif,
                                threshold_rules=threshold_rules,
                            )
                            if desired_path != current_animation_path:
                                LOG.info(
                                    "Switching animation due to temp %.1fC: %s -> %s",
                                    temp,
                                    current_animation_path.name if current_animation_path else "?",
                                    desired_path.name,
                                )
                                current_animation_path = desired_path
                                current_animation = get_or_prepare_animation(desired_path)
                                frame_index = 0
                    except StopRequested:
                        raise
                    except Exception as exc:
                        LOG.warning("Temperature handling failed: %s", exc)
                        if args.reinit_on_error:
                            try:
                                session.reinitialize()
                                session.set_orientation(args.orientation)
                                if args.brightness is not None:
                                    session.set_brightness(int(args.brightness))
                            except Exception as rexc:
                                LOG.warning("Reinit after temp error failed: %s", rexc)

                    if temp is not None and args.show_temp_card_seconds > 0:
                        try:
                            temp_frame = build_temp_card_rgb565(
                                temp_c=temp,
                                orientation_degrees=args.orientation,
                                value_font_size=args.temp_font_size,
                                label_font_size=args.temp_label_font_size,
                                font_path=args.font_path.expanduser() if args.font_path else None,
                            )

                            temp_send_start = time.monotonic()
                            session.send_rgb565_frame(temp_frame)
                            temp_send_ms = _ms_since(temp_send_start)

                            _log_timing(
                                args.debug_timing,
                                "temp_card_send",
                                temp_send_ms,
                                args.debug_timing_warn_ms,
                            )

                            sleep_until(time.monotonic() + args.show_temp_card_seconds, stopflag)
                        except StopRequested:
                            raise
                        except Exception as exc:
                            LOG.warning("Temp card handling failed: %s", exc)
                            if args.reinit_on_error:
                                try:
                                    session.reinitialize()
                                    session.set_orientation(args.orientation)
                                    if args.brightness is not None:
                                        session.set_brightness(int(args.brightness))
                                except Exception as rexc:
                                    LOG.warning("Reinit after temp card error failed: %s", rexc)

                    last_temp_poll = time.monotonic()

                if current_animation is None or not current_animation.frames:
                    raise RuntimeError("No prepared animation frames available")

                frame = current_animation.frames[frame_index]
                frame_counter += 1

                loop_start = time.monotonic()
                send_start = time.monotonic()

                try:
                    frame_start = time.monotonic()
                    session.send_rgb565_frame(frame.rgb565)
                except Exception as exc:
                    LOG.warning("Frame transfer failed: %s", exc)
                    if args.reinit_on_error:
                        try:
                            session.reinitialize()
                            session.set_orientation(args.orientation)
                            if args.brightness is not None:
                                session.set_brightness(int(args.brightness))
                        except Exception as rexc:
                            LOG.warning("Reinit after frame error failed: %s", rexc)
                    else:
                        sleep_until(time.monotonic() + 0.5, stopflag)

                send_ms = _ms_since(send_start)

                if args.debug_timing and (
                    frame_counter % max(1, args.debug_frame_log_every) == 0
                ):
                    _log_timing(
                        True,
                        "frame_send",
                        send_ms,
                        args.debug_timing_warn_ms,
                        extra=(
                            f"frame={frame_counter} idx={frame_index} "
                            f"duration={frame.duration_s * 1000.0:.1f}ms "
                            f"bytes={len(frame.rgb565)}"
                        ),
                    )

                deadline = frame_start + frame.duration_s
                frame_index = (frame_index + 1) % len(current_animation.frames)
                sleep_until(deadline, stopflag)

                loop_ms = _ms_since(loop_start)
                if args.debug_timing and (
                    frame_counter % max(1, args.debug_frame_log_every) == 0
                ):
                    _log_timing(
                        True,
                        "frame_loop",
                        loop_ms,
                        args.debug_timing_warn_ms,
                        extra=f"frame={frame_counter}",
                    )

        except StopRequested:
            LOG.info("Stop requested")
        except KeyboardInterrupt:
            LOG.info("Interrupted")
        except Exception as exc:
            LOG.exception("Fatal error: %s", exc)
            return 1
        finally:
            stopflag.request()
            try:
                if args.reset_mode_on_exit == "liquid":
                    session.reset_display(settle_delay=args.reset_settle_delay)
            finally:
                session.disconnect()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
