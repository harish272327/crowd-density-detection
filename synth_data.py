"""
Synthetic crowd scene generator.

Why synthetic data: this sandbox has NO internet access, so we can't
download a real crowd-counting dataset (ShanghaiTech / Mall / UCF-CC-50 /
JHU-Crowd++, etc). To still deliver a genuinely *trained* model (not just
hand-set thresholds), we procedurally generate crowd scenes with a KNOWN
ground-truth people count, using a simple perspective model (people higher
up in the frame = farther away = smaller), overlap, lighting/texture noise,
and varied backgrounds. This gives the regressor real signal to learn from:
naive "count distinct blobs" breaks down under overlap/occlusion, so it has
to fall back on texture/edge-density cues the way classical holistic crowd
counting methods (Chan & Vasconcelos, etc.) do.

See README.md for how to swap this out for a real dataset.
"""
import numpy as np
import cv2


def _random_background(w, h, rng):
    # Base gradient (floor/plaza look) + a subtle color cast + noise
    top_color = rng.integers(60, 140, size=3)
    bot_color = rng.integers(90, 180, size=3)
    grad = np.linspace(0, 1, h).reshape(h, 1, 1)
    bg = (top_color * (1 - grad) + bot_color * grad).astype(np.float32)
    bg = np.repeat(bg, w, axis=1)

    # occasional horizontal "pavement lines" for texture
    if rng.random() < 0.6:
        n_lines = rng.integers(2, 6)
        for _ in range(n_lines):
            y = rng.integers(int(h * 0.3), h)
            shade = rng.integers(-15, 15)
            bg[y:y + 1, :, :] += shade

    noise = rng.normal(0, 3, size=(h, w, 3))
    bg = np.clip(bg + noise, 0, 255).astype(np.uint8)
    bg = cv2.GaussianBlur(bg, (3, 3), 0)
    return bg


def _draw_person(img, cx, cy, scale, rng):
    """Draw a simple person blob (head + shoulders) with perspective scale."""
    h_head = max(2, int(3.5 * scale))
    body_w = max(2, int(4.0 * scale))
    body_h = max(3, int(6.0 * scale))

    color = tuple(int(c) for c in rng.integers(20, 220, size=3))

    # body (ellipse)
    cv2.ellipse(img, (cx, cy + h_head), (body_w, body_h), 0, 0, 360, color, -1)
    # head (circle), slightly different shade
    head_color = tuple(int(np.clip(c + rng.integers(-20, 20), 0, 255)) for c in color)
    cv2.circle(img, (cx, cy), h_head, head_color, -1)


def generate_scene(width=384, height=288, count=None, max_count=150, rng=None):
    """Generate one synthetic crowd image + its ground-truth count.

    Perspective: y in [0.28*h, h] maps to a scale range so people near the
    top of the frame are small (far away) and people near the bottom are
    large (close up) -- this is what makes density-texture features (not
    just "count blobs") necessary at high density.
    """
    if rng is None:
        rng = np.random.default_rng()
    if count is None:
        # bias sampling towards low/medium counts, still cover the high end
        count = int(rng.exponential(scale=max_count / 3))
        count = min(count, max_count)

    img = _random_background(width, height, rng)
    horizon = height * 0.28

    for _ in range(count):
        y = rng.uniform(horizon, height - 2)
        depth = (y - horizon) / (height - horizon)  # 0=far, 1=near
        scale = 0.6 + depth * 2.2  # far people ~0.6x, near people ~2.8x
        x = rng.uniform(0, width - 1)
        # slight horizontal clustering (crowds clump, not uniform-random)
        if rng.random() < 0.35:
            x = np.clip(x + rng.normal(0, 20), 0, width - 1)
        _draw_person(img, int(x), int(y), scale, rng)

    # mild blur to fuse overlapping people at high density (camera/compression realism)
    if rng.random() < 0.7:
        k = rng.choice([1, 3, 3, 5])
        if k > 1:
            img = cv2.GaussianBlur(img, (int(k), int(k)), 0)

    return img, count


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    for i, c in enumerate([2, 20, 60, 120]):
        img, gt = generate_scene(count=c, rng=rng)
        cv2.imwrite(f"/tmp/sample_{i}_{gt}.jpg", img)
        print(f"sample {i}: requested={c} actual={gt}")
