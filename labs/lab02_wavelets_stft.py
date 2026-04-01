from __future__ import annotations

"""Lab 02 (skeleton): Wavelets (Haar) + STFT bridge."""

import argparse
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np
import numpy.typing as npt
from scipy import signal

ThresholdMode = Literal["soft", "hard"]


def haar_dwt1(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute one-level 1D Haar DWT.

    For odd-length inputs, pad one sample (edge/reflect policy, document choice).

    Args:
        x: 1D numeric signal.

    Returns:
        (approx, detail): each length ~N/2.
    """
    arr = np.asarray(x, dtype=np.float32).ravel()
    if arr.size % 2 == 1:
        arr = np.pad(arr, (0, 1), mode="edge")
    s2 = np.float32(np.sqrt(2.0))
    approx = (arr[0::2] + arr[1::2]) / s2
    detail = (arr[0::2] - arr[1::2]) / s2
    return approx.astype(np.float32), detail.astype(np.float32)


def haar_idwt1(approx: np.ndarray, detail: np.ndarray) -> np.ndarray:
    """
    Invert one-level 1D Haar DWT.

    Args:
        approx: Approximation coefficients.
        detail: Detail coefficients.

    Returns:
        Reconstructed signal.
    """
    a = np.asarray(approx, dtype=np.float32).ravel()
    d = np.asarray(detail, dtype=np.float32).ravel()
    if a.shape != d.shape:
        raise ValueError("approx and detail must have the same shape")
    s2 = np.float32(np.sqrt(2.0))
    even = (a + d) / s2
    odd = (a - d) / s2
    out = np.empty(a.size * 2, dtype=np.float32)
    out[0::2] = even
    out[1::2] = odd
    return out


def haar_dwt2(image: np.ndarray) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Compute one-level 2D separable Haar DWT for grayscale images.

    Args:
        image: 2D grayscale image.

    Returns:
        LL, (LH, HL, HH).
    """
    img = np.asarray(image, dtype=np.float32)
    if img.ndim != 2:
        raise ValueError("haar_dwt2 expects a 2D grayscale image")

    pad_h = img.shape[0] % 2
    pad_w = img.shape[1] % 2
    if pad_h or pad_w:
        img = np.pad(img, ((0, pad_h), (0, pad_w)), mode="edge")

    s2 = np.float32(np.sqrt(2.0))
    low = (img[:, 0::2] + img[:, 1::2]) / s2
    high = (img[:, 0::2] - img[:, 1::2]) / s2

    ll = (low[0::2, :] + low[1::2, :]) / s2
    lh = (low[0::2, :] - low[1::2, :]) / s2
    hl = (high[0::2, :] + high[1::2, :]) / s2
    hh = (high[0::2, :] - high[1::2, :]) / s2

    return ll.astype(np.float32), (lh.astype(np.float32), hl.astype(np.float32), hh.astype(np.float32))


def haar_idwt2(LL: np.ndarray, bands: tuple[np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
    """
    Invert one-level 2D Haar DWT.

    Args:
        LL: Low-low sub-band.
        bands: Tuple `(LH, HL, HH)`.

    Returns:
        Reconstructed image (crop policy for odd sizes should be documented).
    """
    lh, hl, hh = bands
    ll = np.asarray(LL, dtype=np.float32)
    lh = np.asarray(lh, dtype=np.float32)
    hl = np.asarray(hl, dtype=np.float32)
    hh = np.asarray(hh, dtype=np.float32)

    if not (ll.shape == lh.shape == hl.shape == hh.shape):
        raise ValueError("LL, LH, HL, HH must all have the same shape")

    s2 = np.float32(np.sqrt(2.0))

    low_even = (ll + lh) / s2
    low_odd = (ll - lh) / s2
    high_even = (hl + hh) / s2
    high_odd = (hl - hh) / s2

    h2, w2 = ll.shape
    low = np.empty((2 * h2, w2), dtype=np.float32)
    high = np.empty((2 * h2, w2), dtype=np.float32)
    low[0::2, :] = low_even
    low[1::2, :] = low_odd
    high[0::2, :] = high_even
    high[1::2, :] = high_odd

    even_cols = (low + high) / s2
    odd_cols = (low - high) / s2
    out = np.empty((2 * h2, 2 * w2), dtype=np.float32)
    out[:, 0::2] = even_cols
    out[:, 1::2] = odd_cols
    return out


def wavelet_threshold(coeffs: Any, threshold: float, mode: ThresholdMode = "soft") -> Any:
    """
    Apply thresholding to coefficient arrays.

    Args:
        coeffs: Array or nested tuples/lists of arrays.
        threshold: Non-negative threshold value.
        mode: `"soft"` or `"hard"`.

    Returns:
        Thresholded coefficients with same structure.
    """
    thr = float(threshold)
    if thr < 0.0:
        raise ValueError("threshold must be non-negative")

    if isinstance(coeffs, np.ndarray):
        c = np.asarray(coeffs, dtype=np.float32)
        if mode == "hard":
            return np.where(np.abs(c) >= thr, c, 0.0).astype(np.float32)
        if mode == "soft":
            return (np.sign(c) * np.maximum(np.abs(c) - thr, 0.0)).astype(np.float32)
        raise ValueError("mode must be 'soft' or 'hard'")

    if isinstance(coeffs, tuple):
        return tuple(wavelet_threshold(c, thr, mode) for c in coeffs)
    if isinstance(coeffs, list):
        return [wavelet_threshold(c, thr, mode) for c in coeffs]
    return coeffs


def wavelet_denoise(image: np.ndarray, levels: int, threshold: float, mode: ThresholdMode = "soft") -> np.ndarray:
    """
    Denoise image via multi-level Haar thresholding.

    Args:
        image: 2D grayscale image.
        levels: Number of decomposition levels.
        threshold: Coefficient threshold.
        mode: `"soft"` or `"hard"`.

    Returns:
        Denoised image with deterministic behavior.
    """
    if levels < 1:
        raise ValueError("levels must be >= 1")

    img = np.asarray(image, dtype=np.float32)
    if img.ndim != 2:
        raise ValueError("wavelet_denoise expects a 2D grayscale image")

    details_per_level: list[tuple[tuple[int, int], tuple[np.ndarray, np.ndarray, np.ndarray]]] = []
    current = img
    for _ in range(levels):
        original_shape = current.shape
        ll, bands = haar_dwt2(current)
        thr_bands = wavelet_threshold(bands, threshold, mode)
        details_per_level.append((original_shape, thr_bands))
        current = ll

    recon = current
    for shape, bands in reversed(details_per_level):
        recon = haar_idwt2(recon, bands)
        recon = recon[: shape[0], : shape[1]]

    return recon.astype(np.float32)


def stft1(
    x: np.ndarray,
    fs_hz: float,
    frame_len: int,
    hop_len: int,
    window: str = "hann",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute STFT for 1D signal using SciPy.

    Returns:
        `(freqs_hz, times_s, Zxx)` where `Zxx` is complex.
    """
    noverlap = frame_len - hop_len
    if noverlap < 0:
        raise ValueError("hop_len must be <= frame_len")
    freqs_hz, times_s, zxx = signal.stft(
        np.asarray(x, dtype=np.float32),
        fs=float(fs_hz),
        window=window,
        nperseg=int(frame_len),
        noverlap=int(noverlap),
        boundary=None,
        padded=False,
    )
    return freqs_hz, times_s, zxx


def spectrogram_magnitude(Zxx: np.ndarray, log_scale: bool = True) -> np.ndarray:
    """
    Convert STFT matrix to magnitude spectrogram.

    Args:
        Zxx: Complex STFT matrix.
        log_scale: If True, return `log(1 + magnitude)`.

    Returns:
        Non-negative finite magnitude matrix.
    """
    mag = np.abs(np.asarray(Zxx, dtype=np.complex64)).astype(np.float32)
    if log_scale:
        mag = np.log1p(mag)
    return np.nan_to_num(mag, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def normalize_to_uint8(x: npt.ArrayLike) -> npt.NDArray[np.uint8]:
    """Min-max normalize an array to `[0,255]` for visualization."""
    arr = np.asarray(x, dtype=np.float64)
    mn, mx = float(np.min(arr)), float(np.max(arr))
    if mx <= mn:
        return np.zeros_like(arr, dtype=np.uint8)
    scaled = (arr - mn) * (255.0 / (mx - mn))
    return np.clip(np.round(scaled), 0, 255).astype(np.uint8)


def main() -> int:
    """
    Lab 02 demo (skeleton).

    Expected behavior after implementation:
    - wavelet denoising demo on image from `./imgs/`
    - LL/LH/HL/HH band visualization
    - STFT spectrogram demo on synthetic chirp signal
    - save outputs to `./out/lab02/` (no GUI windows)
    """
    parser = argparse.ArgumentParser(description="Lab 02 skeleton (implement functions first).")
    parser.add_argument("--img", type=str, default="lenna.png", help="Image from ./imgs/")
    parser.add_argument("--out", type=str, default="out/lab02", help="Output directory (relative to repo root)")
    args = parser.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def save_figure(path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()

    repo_root = Path(__file__).resolve().parents[1]
    imgs_dir = repo_root / "imgs"
    out_dir = (repo_root / args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    img = cv2.imread(str(imgs_dir / args.img), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(str(imgs_dir / args.img))

    missing: list[str] = []

    # --- Wavelet demo ---
    try:
        rng = np.random.default_rng(0)
        noisy = img.astype(np.float32) + rng.normal(0.0, 20.0, size=img.shape).astype(np.float32)
        noisy = np.clip(noisy, 0.0, 255.0)
        den = wavelet_denoise(noisy, levels=2, threshold=20.0, mode="soft")

        ll, (lh, hl, hh) = haar_dwt2(img.astype(np.float32))

        plt.figure(figsize=(12, 4))
        for i, (title, im) in enumerate(
            [
                ("Original", img),
                ("Noisy (Gaussian)", noisy),
                ("Wavelet denoised", den),
            ],
            start=1,
        ):
            plt.subplot(1, 3, i)
            plt.title(title)
            plt.imshow(normalize_to_uint8(im), cmap="gray")
            plt.axis("off")
        save_figure(out_dir / "wavelet_denoise.png")

        plt.figure(figsize=(10, 8))
        for i, (title, band) in enumerate(
            [
                ("LL", ll),
                ("LH", lh),
                ("HL", hl),
                ("HH", hh),
            ],
            start=1,
        ):
            plt.subplot(2, 2, i)
            plt.title(title)
            plt.imshow(normalize_to_uint8(band), cmap="gray")
            plt.axis("off")
        save_figure(out_dir / "wavelet_bands.png")
    except NotImplementedError as exc:
        missing.append(str(exc))

    # --- STFT bridge demo ---
    try:
        fs = 400.0
        duration_s = 2.0
        t = np.arange(int(fs * duration_s), dtype=np.float64) / fs
        f0, f1 = 15.0, 120.0
        k = (f1 - f0) / duration_s
        phase = 2.0 * np.pi * (f0 * t + 0.5 * k * t * t)
        x = np.sin(phase)

        freqs, times, zxx = stft1(x, fs_hz=fs, frame_len=128, hop_len=32, window="hann")
        mag = spectrogram_magnitude(zxx, log_scale=True)

        plt.figure(figsize=(8, 4))
        plt.pcolormesh(times, freqs, mag, shading="gouraud")
        plt.title("STFT Spectrogram (log-magnitude)")
        plt.xlabel("Time [s]")
        plt.ylabel("Frequency [Hz]")
        plt.colorbar(label="log(1 + |Zxx|)")
        save_figure(out_dir / "stft_spectrogram.png")
    except NotImplementedError as exc:
        missing.append(str(exc))

    if missing:
        (out_dir / "STATUS.txt").write_text(
            "Lab 02 demo is incomplete. Implement the TODO functions in labs/lab02_wavelets_stft.py.\n\n"
            + "\n".join(f"- {m}" for m in missing)
            + "\n",
            encoding="utf-8",
        )
        print(f"Wrote {out_dir / 'STATUS.txt'}")
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
