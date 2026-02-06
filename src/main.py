#!/usr/bin/env python3
import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

try:
    import cv2
    import mss
    import numpy as np
except Exception as exc:
    print("Missing dependencies. Install from requirements.txt:", exc)
    sys.exit(1)

try:
    from pynput.keyboard import Controller, Key
except Exception as exc:
    print("Missing pynput. Install from requirements.txt:", exc)
    sys.exit(1)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config" / "config.json"
ICON_PATH = ROOT / "support" / "Captura de tela 2026-02-01 164642.png"


@dataclass
class ROI:
    x: int
    y: int
    width: int
    height: int

    @property
    def bbox(self):
        return {"left": int(self.x), "top": int(self.y), "width": int(self.width), "height": int(self.height)}


@dataclass
class DebugCfg:
    enabled: bool
    log_every_n: int
    save_frames: bool
    frames_dir: str


@dataclass
class RedHSV:
    h1: tuple
    h2: tuple
    h3: tuple
    h4: tuple


@dataclass
class AppConfig:
    threshold_percent: float
    full_percent: float
    bar_color: Optional[Tuple[float, float, float]]
    color_tol: float
    trigger_frames: int
    cooldown_seconds: float
    poll_interval_ms: int
    use_grounding: bool
    roi: ROI
    red_hsv: RedHSV
    debug: DebugCfg


def load_config(path: Path) -> AppConfig:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    roi = data["roi"]
    red = data["red_hsv"]
    dbg = data.get("debug", {})
    return AppConfig(
        threshold_percent=float(data["threshold_percent"]),
        full_percent=float(data.get("full_percent", 100.0)),
        bar_color=tuple(data["bar_color"]) if data.get("bar_color") else None,
        color_tol=float(data.get("color_tol", 28.0)),
        trigger_frames=int(data["trigger_frames"]),
        cooldown_seconds=float(data["cooldown_seconds"]),
        poll_interval_ms=int(data["poll_interval_ms"]),
        use_grounding=bool(data.get("use_grounding", True)),
        roi=ROI(int(roi["x"]), int(roi["y"]), int(roi["width"]), int(roi["height"])),
        red_hsv=RedHSV(tuple(red["h1"]), tuple(red["h2"]), tuple(red["h3"]), tuple(red["h4"])),
        debug=DebugCfg(
            enabled=bool(dbg.get("enabled", False)),
            log_every_n=int(dbg.get("log_every_n", 10)),
            save_frames=bool(dbg.get("save_frames", False)),
            frames_dir=str(dbg.get("frames_dir", "debug_frames")),
        ),
    )


def save_config(path: Path, cfg: AppConfig) -> None:
    data = {
        "threshold_percent": cfg.threshold_percent,
        "full_percent": cfg.full_percent,
        "bar_color": list(cfg.bar_color) if cfg.bar_color else None,
        "color_tol": cfg.color_tol,
        "trigger_frames": cfg.trigger_frames,
        "cooldown_seconds": cfg.cooldown_seconds,
        "poll_interval_ms": cfg.poll_interval_ms,
        "use_grounding": cfg.use_grounding,
        "roi": {
            "x": cfg.roi.x,
            "y": cfg.roi.y,
            "width": cfg.roi.width,
            "height": cfg.roi.height,
        },
        "red_hsv": {
            "h1": list(cfg.red_hsv.h1),
            "h2": list(cfg.red_hsv.h2),
            "h3": list(cfg.red_hsv.h3),
            "h4": list(cfg.red_hsv.h4),
        },
        "debug": {
            "enabled": cfg.debug.enabled,
            "log_every_n": cfg.debug.log_every_n,
            "save_frames": cfg.debug.save_frames,
            "frames_dir": cfg.debug.frames_dir,
        },
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def setup_logging(debug: bool):
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="[%(asctime)s] %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )


def build_red_mask(roi_bgr, red: RedHSV):
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, red.h1, red.h2)
    mask2 = cv2.inRange(hsv, red.h3, red.h4)
    mask_hsv = cv2.bitwise_or(mask1, mask2)

    b = roi_bgr[:, :, 0].astype(np.int16)
    g = roi_bgr[:, :, 1].astype(np.int16)
    r = roi_bgr[:, :, 2].astype(np.int16)
    mask_bgr = (r > g + 15) & (r > b + 15) & (r > 50)
    mask_bgr = (mask_bgr.astype(np.uint8) * 255)

    mask = cv2.bitwise_or(mask_hsv, mask_bgr)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    return mask


def estimate_bar_color_and_tol(roi_bgr) -> Optional[Tuple[Tuple[float, float, float], float]]:
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1].astype(np.float32)
    v = hsv[:, :, 2].astype(np.float32)
    sat_thresh = np.percentile(s, 90)
    mask = (s >= sat_thresh) & (v >= 40)
    if np.count_nonzero(mask) < 10:
        b = roi_bgr[:, :, 0].astype(np.float32)
        g = roi_bgr[:, :, 1].astype(np.float32)
        r = roi_bgr[:, :, 2].astype(np.float32)
        score = r - (g + b) / 2.0
        score_thresh = np.percentile(score, 95)
        mask = score >= score_thresh
        if np.count_nonzero(mask) < 10:
            return None
    pixels = roi_bgr[mask].astype(np.float32)
    color = tuple(np.mean(pixels, axis=0).tolist())

    lab = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    target = cv2.cvtColor(np.uint8([[color]]), cv2.COLOR_BGR2LAB).astype(np.float32)[0, 0]
    dist = np.linalg.norm(lab - target, axis=2)
    tol = float(np.percentile(dist[mask], 90) + 5.0)
    tol = max(12.0, min(45.0, tol))
    return color, tol


def select_roi_interactive(red: RedHSV) -> ROI:
    with mss.mss() as sct:
        monitor = sct.monitors[1]
        frame = np.array(sct.grab(monitor))
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    window = "Select AG Bar (2 clicks)"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    clicks = []

    def on_mouse(event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONUP:
            if len(clicks) < 2:
                clicks.append((x, y))

    cv2.setMouseCallback(window, on_mouse)

    while True:
        display = frame_bgr.copy()
        instructions = [
            "Clique na borda ESQUERDA e depois na borda DIREITA da barra do AG.",
            "ENTER/SPACE para confirmar, C para cancelar.",
        ]
        for idx, text in enumerate(instructions):
            cv2.putText(display, text, (12, 24 + idx * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)
            cv2.putText(display, text, (12, 24 + idx * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        if len(clicks) >= 1:
            cv2.circle(display, clicks[0], 5, (0, 255, 0), 2)
        if len(clicks) >= 2:
            cv2.circle(display, clicks[1], 5, (0, 255, 0), 2)
            cv2.line(display, clicks[0], clicks[1], (0, 255, 0), 1)

        cv2.imshow(window, display)
        key = cv2.waitKey(20) & 0xFF
        if key in (27, ord("c"), ord("C")):
            cv2.destroyAllWindows()
            raise RuntimeError("ROI selection canceled.")
        if key in (13, 10, 32) and len(clicks) >= 2:
            break

    cv2.destroyAllWindows()

    (x0, y0), (x1, y1) = clicks[0], clicks[1]
    if x0 == x1:
        raise RuntimeError("Selecao invalida: largura zero.")
    left = max(0, min(x0, x1))
    right = min(frame_bgr.shape[1] - 1, max(x0, x1))
    width = max(2, right - left + 1)

    y_center = int(round((y0 + y1) / 2))
    band_top = max(0, y_center - 20)
    band_bottom = min(frame_bgr.shape[0], y_center + 21)
    crop = frame_bgr[band_top:band_bottom, left : left + width]
    mask = build_red_mask(crop, red)
    red_per_row = np.count_nonzero(mask, axis=1)
    row_idx = int(np.argmax(red_per_row))
    min_row_red = max(4, int(width * 0.05))
    if red_per_row[row_idx] < min_row_red:
        logging.warning("Nao detectei a barra pela cor; usando linha clicada.")
        row_abs = y_center
    else:
        row_abs = band_top + row_idx

    height = 5
    top = max(0, row_abs - 2)
    if top + height > frame_bgr.shape[0]:
        top = max(0, frame_bgr.shape[0] - height)

    roi = ROI(int(left), int(top), int(width), int(height))
    logging.info("ROI gerado a partir de 2 cliques: %s", roi)
    return roi


def load_icon_template(path: Path):
    if not path.exists():
        return None
    icon = cv2.imread(str(path))
    if icon is None:
        return None
    return cv2.cvtColor(icon, cv2.COLOR_BGR2GRAY)


def icon_present(icon_gray, search_bgr, threshold: float = 0.75) -> bool:
    if icon_gray is None:
        return True
    search_gray = cv2.cvtColor(search_bgr, cv2.COLOR_BGR2GRAY)
    if (search_gray.shape[0] < icon_gray.shape[0]) or (search_gray.shape[1] < icon_gray.shape[1]):
        return False
    res = cv2.matchTemplate(search_gray, icon_gray, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, _ = cv2.minMaxLoc(res)
    return max_val >= threshold


def compute_life_percent(roi_bgr, red: RedHSV):
    mask = build_red_mask(roi_bgr, red)

    red_ratio = np.count_nonzero(mask) / mask.size
    if red_ratio <= 0:
        return None, red_ratio, mask

    # The AG bar is often 1-3 pixels tall; use the strongest row band to
    # estimate fill by counting columns with any red pixel.
    red_per_row = np.count_nonzero(mask, axis=1)
    row_idx = int(np.argmax(red_per_row))
    band_half = 1
    band_top = max(0, row_idx - band_half)
    band_bottom = min(mask.shape[0], row_idx + band_half + 1)
    band = mask[band_top:band_bottom]

    red_pixels_per_col = np.count_nonzero(band, axis=0)
    filled_cols = np.count_nonzero(red_pixels_per_col > 0)
    min_row_red = max(4, int(mask.shape[1] * 0.05))
    if red_per_row[row_idx] < min_row_red or red_ratio < 0.0005:
        return None, red_ratio, mask
    percent = (filled_cols / mask.shape[1]) * 100.0
    percent = max(0.0, min(100.0, percent))
    return percent, red_ratio, mask


def compute_life_percent_by_color(roi_bgr, color: Tuple[float, float, float], tol: float):
    lab = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    target = cv2.cvtColor(np.uint8([[color]]), cv2.COLOR_BGR2LAB).astype(np.float32)[0, 0]
    dist = np.linalg.norm(lab - target, axis=2)
    mask = (dist <= tol).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

    ratio = np.count_nonzero(mask) / mask.size
    if ratio <= 0:
        return None, ratio, mask

    red_per_row = np.count_nonzero(mask, axis=1)
    row_idx = int(np.argmax(red_per_row))
    band_half = 1
    band_top = max(0, row_idx - band_half)
    band_bottom = min(mask.shape[0], row_idx + band_half + 1)
    band = mask[band_top:band_bottom]

    per_col = np.count_nonzero(band, axis=0)
    filled_cols = np.count_nonzero(per_col > 0)
    min_row = max(4, int(mask.shape[1] * 0.05))
    if red_per_row[row_idx] < min_row or ratio < 0.0005:
        return None, ratio, mask
    percent = (filled_cols / mask.shape[1]) * 100.0
    percent = max(0.0, min(100.0, percent))
    return percent, ratio, mask


def build_icon_search_bbox(roi: ROI, screen_w: int, screen_h: int):
    x0 = max(0, roi.x - 60)
    y0 = max(0, roi.y - 110)
    x1 = min(screen_w, roi.x + roi.width + 60)
    y1 = min(screen_h, roi.y + roi.height + 80)
    return {"left": x0, "top": y0, "width": x1 - x0, "height": y1 - y0}


def show_start_dialog(cfg: AppConfig, preselect_roi: bool) -> Optional[Tuple[float, bool, bool]]:
    try:
        import tkinter as tk
        from tkinter import messagebox
    except Exception:
        logging.warning("UI indisponivel. Use --threshold ou config.json.")
        return cfg.threshold_percent, preselect_roi

    result = {"ok": False, "threshold": cfg.threshold_percent, "select_roi": preselect_roi, "calibrate": False}

    root = tk.Tk()
    root.title("AG Monitor")
    root.resizable(False, False)
    root.attributes("-topmost", True)

    title = tk.Label(root, text="AG Monitor", font=("Segoe UI", 12, "bold"))
    title.grid(row=0, column=0, columnspan=2, padx=16, pady=(14, 6))

    tk.Label(root, text="Threshold (%)").grid(row=1, column=0, sticky="w", padx=16, pady=6)
    threshold_var = tk.StringVar(value=str(cfg.threshold_percent))
    threshold_entry = tk.Entry(root, textvariable=threshold_var, width=10)
    threshold_entry.grid(row=1, column=1, sticky="e", padx=16, pady=6)

    roi_text = f"ROI atual: x={cfg.roi.x} y={cfg.roi.y} w={cfg.roi.width} h={cfg.roi.height}"
    tk.Label(root, text=roi_text).grid(row=2, column=0, columnspan=2, padx=16, pady=(4, 6))

    select_roi_var = tk.BooleanVar(value=preselect_roi)
    select_roi_chk = tk.Checkbutton(root, text="Selecionar ROI agora", variable=select_roi_var)
    select_roi_chk.grid(row=3, column=0, columnspan=2, sticky="w", padx=16, pady=(0, 8))

    calibrate_var = tk.BooleanVar(value=True)
    calibrate_chk = tk.Checkbutton(root, text="Calibrar (assumir vida cheia agora)", variable=calibrate_var)
    calibrate_chk.grid(row=4, column=0, columnspan=2, sticky="w", padx=16, pady=(0, 8))

    def on_start():
        raw = threshold_var.get().strip().replace(",", ".")
        try:
            value = float(raw)
        except ValueError:
            messagebox.showerror("Erro", "Threshold invalido. Use um numero entre 1 e 100.")
            return
        if value <= 0 or value > 100:
            messagebox.showerror("Erro", "Threshold deve estar entre 1 e 100.")
            return
        result["ok"] = True
        result["threshold"] = value
        result["select_roi"] = bool(select_roi_var.get())
        result["calibrate"] = bool(calibrate_var.get())
        root.destroy()

    def on_cancel():
        root.destroy()

    btn_frame = tk.Frame(root)
    btn_frame.grid(row=5, column=0, columnspan=2, pady=(4, 12))
    tk.Button(btn_frame, text="Iniciar", width=12, command=on_start).grid(row=0, column=0, padx=6)
    tk.Button(btn_frame, text="Cancelar", width=12, command=on_cancel).grid(row=0, column=1, padx=6)

    root.protocol("WM_DELETE_WINDOW", on_cancel)
    threshold_entry.focus_set()
    root.mainloop()

    if not result["ok"]:
        return None
    return float(result["threshold"]), bool(result["select_roi"]), bool(result["calibrate"])


def scale_percent(raw_percent: float, full_percent: float) -> float:
    if full_percent <= 0:
        return raw_percent
    scaled = (raw_percent / full_percent) * 100.0
    return max(0.0, min(100.0, scaled))


def calibrate_full_percent(cfg: AppConfig, sct: mss.mss, samples: int = 12) -> Optional[float]:
    raw_values = []
    for _ in range(samples):
        roi_frame = np.array(sct.grab(cfg.roi.bbox))
        roi_bgr = cv2.cvtColor(roi_frame, cv2.COLOR_BGRA2BGR)
        raw = None
        if cfg.bar_color:
            raw, _ratio, _mask = compute_life_percent_by_color(roi_bgr, cfg.bar_color, cfg.color_tol)
        if raw is None:
            raw, _ratio, _mask = compute_life_percent(roi_bgr, cfg.red_hsv)
        if raw is not None:
            raw_values.append(raw)
        time.sleep(cfg.poll_interval_ms / 1000.0)
    if not raw_values:
        return None
    return max(raw_values)


def calibrate_color(cfg: AppConfig, sct: mss.mss, samples: int = 10) -> bool:
    colors = []
    tols = []
    for _ in range(samples):
        roi_frame = np.array(sct.grab(cfg.roi.bbox))
        roi_bgr = cv2.cvtColor(roi_frame, cv2.COLOR_BGRA2BGR)
        result = estimate_bar_color_and_tol(roi_bgr)
        if result:
            color, tol = result
            colors.append(color)
            tols.append(tol)
        time.sleep(cfg.poll_interval_ms / 1000.0)
    if not colors:
        return False
    avg = np.mean(np.array(colors), axis=0)
    cfg.bar_color = (float(avg[0]), float(avg[1]), float(avg[2]))
    cfg.color_tol = float(np.mean(tols))
    return True


def main():
    parser = argparse.ArgumentParser(description="AG life monitor with ESC trigger")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Path to config.json")
    parser.add_argument("--threshold", type=float, help="Override threshold percent")
    parser.add_argument("--no-grounding", action="store_true", help="Disable icon grounding")
    parser.add_argument("--select-roi", action="store_true", help="Interactively select ROI before running")
    parser.add_argument("--select-roi-only", action="store_true", help="Select ROI and exit")
    parser.add_argument("--no-ui", action="store_true", help="Skip the startup UI")
    parser.add_argument("--calibrate", action="store_true", help="Calibrate full life at startup")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--no-debug", action="store_true", help="Disable debug logging")
    parser.add_argument("--save-frames", action="store_true", help="Save ROI/mask frames when debugging")
    parser.add_argument("--dry-run", action="store_true", help="Do not send ESC (debug only)")

    args = parser.parse_args()

    cfg_path = Path(args.config)
    cfg = load_config(cfg_path)

    if args.threshold is not None:
        cfg.threshold_percent = float(args.threshold)
    if args.no_grounding:
        cfg.use_grounding = False
    if args.debug:
        cfg.debug.enabled = True
    if args.no_debug:
        cfg.debug.enabled = False
    if args.save_frames:
        cfg.debug.save_frames = True

    setup_logging(cfg.debug.enabled)

    if not args.no_ui and not args.select_roi_only:
        ui_result = show_start_dialog(cfg, args.select_roi)
        if ui_result is None:
            logging.info("Canceled by user. Exiting.")
            return 0
        cfg.threshold_percent, ui_select_roi, ui_calibrate = ui_result
        args.select_roi = args.select_roi or ui_select_roi
        args.calibrate = args.calibrate or ui_calibrate
        save_config(cfg_path, cfg)

    if args.select_roi or args.select_roi_only:
        logging.info("Clique na borda esquerda e direita da barra do AG, depois ENTER/SPACE.")
        try:
            new_roi = select_roi_interactive(cfg.red_hsv)
        except Exception as exc:
            logging.error("ROI selection failed: %s", exc)
            return 1
        cfg.roi = new_roi
        save_config(cfg_path, cfg)
        logging.info("Saved ROI to %s: %s", cfg_path, cfg.roi)
        if args.select_roi_only:
            return 0

    if cfg.debug.save_frames:
        frames_dir = Path(cfg.debug.frames_dir)
        frames_dir.mkdir(parents=True, exist_ok=True)

    icon_gray = load_icon_template(ICON_PATH)

    keyboard = Controller()
    below_count = 0
    last_trigger = 0.0
    frame_idx = 0

    with mss.mss() as sct:
        monitor = sct.monitors[1]
        screen_w = monitor["width"]
        screen_h = monitor["height"]

        icon_bbox = build_icon_search_bbox(cfg.roi, screen_w, screen_h)

        if args.calibrate:
            logging.info("Calibrando cor da barra...")
            if calibrate_color(cfg, sct):
                save_config(cfg_path, cfg)
                logging.info("Cor calibrada: bar_color=%s tol=%.1f", cfg.bar_color, cfg.color_tol)
            else:
                logging.warning("Falha na calibracao de cor.")

            logging.info("Calibrando vida cheia...")
            full = calibrate_full_percent(cfg, sct)
            if full:
                cfg.full_percent = full
                save_config(cfg_path, cfg)
                logging.info("Calibracao salva: full_percent=%.1f", cfg.full_percent)
            else:
                logging.warning("Falha na calibracao. Mantendo full_percent=%.1f", cfg.full_percent)

        logging.info(
            "Starting monitor: threshold=%.1f%% roi=%s grounding=%s full=%.1f",
            cfg.threshold_percent,
            cfg.roi,
            cfg.use_grounding,
            cfg.full_percent,
        )

        while True:
            frame_idx += 1
            roi_frame = np.array(sct.grab(cfg.roi.bbox))
            roi_bgr = cv2.cvtColor(roi_frame, cv2.COLOR_BGRA2BGR)

            grounded = True
            if cfg.use_grounding:
                search_frame = np.array(sct.grab(icon_bbox))
                search_bgr = cv2.cvtColor(search_frame, cv2.COLOR_BGRA2BGR)
                grounded = icon_present(icon_gray, search_bgr)

            if grounded:
                if cfg.bar_color:
                    percent, red_ratio, mask = compute_life_percent_by_color(
                        roi_bgr,
                        cfg.bar_color,
                        cfg.color_tol,
                    )
                else:
                    percent, red_ratio, mask = compute_life_percent(roi_bgr, cfg.red_hsv)
                if percent is None:
                    below_count = 0
                    if cfg.debug.enabled and frame_idx % cfg.debug.log_every_n == 0:
                        logging.debug("bar not detected; red_ratio=%.3f", red_ratio)
                    time.sleep(cfg.poll_interval_ms / 1000.0)
                    continue
                scaled = scale_percent(percent, cfg.full_percent)
                if cfg.debug.enabled and frame_idx % cfg.debug.log_every_n == 0:
                    logging.debug(
                        "life_raw=%.1f%% life=%.1f%% red_ratio=%.3f below=%d grounded=%s",
                        percent,
                        scaled,
                        red_ratio,
                        below_count,
                        grounded,
                    )

                if cfg.debug.save_frames and frame_idx % cfg.debug.log_every_n == 0:
                    ts = int(time.time() * 1000)
                    cv2.imwrite(str(Path(cfg.debug.frames_dir) / f"roi_{ts}.png"), roi_bgr)
                    cv2.imwrite(str(Path(cfg.debug.frames_dir) / f"mask_{ts}.png"), mask)

                now = time.monotonic()
                if scaled <= cfg.threshold_percent:
                    below_count += 1
                else:
                    below_count = 0

                if below_count >= cfg.trigger_frames and (now - last_trigger) >= cfg.cooldown_seconds:
                    if args.dry_run:
                        logging.warning("DRY RUN: threshold reached, would send ESC now.")
                    else:
                        logging.warning("Threshold reached. Sending ESC.")
                        keyboard.press(Key.esc)
                        keyboard.release(Key.esc)
                    last_trigger = now
                    below_count = 0
            else:
                if cfg.debug.enabled and frame_idx % cfg.debug.log_every_n == 0:
                    logging.debug("grounding failed; skipping frame")

            time.sleep(cfg.poll_interval_ms / 1000.0)


if __name__ == "__main__":
    raise SystemExit(main())
