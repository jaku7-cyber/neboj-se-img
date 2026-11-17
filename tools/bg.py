"""Procedural cinematic backgrounds for fearnot.daily (no external images, no licensing).

scene_layers(theme, w, h, seed) -> list of (RGBA PIL image, parallax_factor)
  layers are back-to-front; parallax 0 = static sky, 1 = nearest layer.
compose(layers, w, h, t=0, shift=40) -> RGB image with grain, vignette and text-safe darkening.
"""
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

# palettes: sky top, sky horizon, sun, mountain base (near), fog
PALETTES = {
    'dawn':   dict(top=(28, 36, 64), hor=(236, 150, 92), sun=(255, 214, 150), land=(26, 22, 34), fog=(230, 170, 140), stars=False, sun_y=0.56),
    'dusk':   dict(top=(18, 20, 38), hor=(170, 82, 70), sun=(255, 170, 110), land=(16, 14, 22), fog=(140, 90, 100), stars=False, sun_y=0.60),
    'storm':  dict(top=(22, 26, 32), hor=(92, 102, 112), sun=(200, 210, 220), land=(14, 16, 20), fog=(110, 118, 126), stars=False, sun_y=0.40),
    'night':  dict(top=(6, 8, 22), hor=(40, 52, 96), sun=(200, 210, 255), land=(8, 9, 16), fog=(60, 70, 110), stars=True, sun_y=0.30),
    'gold':   dict(top=(44, 38, 52), hor=(242, 188, 96), sun=(255, 236, 180), land=(34, 26, 24), fog=(240, 200, 140), stars=False, sun_y=0.52),
    'mist':   dict(top=(58, 70, 82), hor=(176, 186, 190), sun=(240, 240, 230), land=(30, 38, 40), fog=(196, 204, 206), stars=False, sun_y=0.45),
}
# struggle / theme keyword -> (palette, scene)
THEMES = [
    (('fear', 'strach', 'anx', 'úzkost', 'odvah', 'courage'), ('storm', 'peaks')),
    (('rest', 'odpoč', 'únav', 'tired', 'weary', 'pokoj', 'peace'), ('dawn', 'lake')),
    (('strength', 'síla', 'vytrval', 'persever', 'mužnost'), ('gold', 'peaks')),
    (('freedom', 'svoboda', 'chain', 'pokušení', 'temptation', 'čistota', 'purity'), ('dawn', 'valley')),
    (('hope', 'naděje', 'lonel', 'osamě', 'loss', 'ztráta'), ('night', 'lake')),
    (('trust', 'důvěra', 'shepherd', 'pastýř'), ('mist', 'hills')),
    (('fall', 'pád', 'rise', 'vstáv', 'forgiv', 'odpuš', 'shame', 'stud', 'failure', 'selhání'), ('dawn', 'valley')),
    (('grat', 'vděč', 'identity', 'identit'), ('gold', 'hills')),
]


def pick(theme, seed):
    t = (theme or '').lower()
    for keys, val in THEMES:
        if any(k in t for k in keys):
            return val
    opts = [('dawn', 'peaks'), ('mist', 'forest'), ('night', 'lake'), ('gold', 'hills'), ('dusk', 'valley')]
    return opts[seed % len(opts)]


def lerp(a, b, t):
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


def fbm2d(w, h, rng, octaves=5, base=4):
    out = np.zeros((h, w), np.float32)
    amp, tot = 1.0, 0.0
    for o in range(octaves):
        gw, gh = base * 2 ** o + 1, int((base * 2 ** o) * h / w) + 2
        g = rng.random((gh, gw)).astype(np.float32)
        im = Image.fromarray((g * 255).astype(np.uint8)).resize((w, h), Image.BICUBIC)
        out += np.asarray(im, np.float32) / 255 * amp
        tot += amp
        amp *= 0.5
    return out / tot


def ridge(w, rng, rough=0.55, octaves=7):
    n = 2 ** octaves + 1
    y = np.zeros(n)
    y[0], y[-1] = rng.random(), rng.random()
    step, scale = n - 1, 1.0
    while step > 1:
        half = step // 2
        for i in range(half, n - 1, step):
            y[i] = (y[i - half] + y[i + half]) / 2 + (rng.random() - 0.5) * scale
        step, scale = half, scale * rough
    xs = np.linspace(0, n - 1, w)
    return np.interp(xs, np.arange(n), y)


def sky_layer(w, h, p, rng):
    yy = np.linspace(0, 1, h)[:, None]
    hor = p['sun_y'] + 0.05
    t = np.clip(yy / hor, 0, 1) ** 1.6
    top, ho = np.array(p['top'], np.float32), np.array(p['hor'], np.float32)
    img = top * (1 - t[..., None]) + ho * t[..., None]
    img = np.repeat(img, w, axis=1)
    # sun glow
    sx, sy = w * (0.5 + (rng.random() - 0.5) * 0.25), h * p['sun_y']
    X, Y = np.meshgrid(np.arange(w), np.arange(h))
    d = np.sqrt((X - sx) ** 2 + ((Y - sy) * 1.15) ** 2)
    glow = np.exp(-(d / (w * 0.42)) ** 2)[..., None] * 0.55 + np.exp(-(d / (w * 0.07)) ** 2)[..., None] * 0.9
    img = img + (np.array(p['sun'], np.float32) - img) * np.clip(glow, 0, 1)
    # clouds
    cl = fbm2d(w, h, rng, 6, 3)
    cl = np.clip((cl - 0.52) * 3.2, 0, 1) * np.clip(1 - yy / (p['sun_y'] + 0.1), 0, 1) ** 0.6
    lit = np.clip(1 - d / (w * 0.9), 0, 1)[..., None]
    ccol = np.array(p['top'], np.float32) * 0.6 + np.array(p['hor'], np.float32) * 0.4
    ccol = ccol + (np.array(p['sun'], np.float32) - ccol) * lit * 0.7
    img = img * (1 - cl[..., None] * 0.75) + ccol * cl[..., None] * 0.75
    if p['stars']:
        s = rng.random((h, w))
        star = (s > 0.9993) * np.clip(1 - yy / (p['sun_y'] + 0.15), 0, 1)
        img = img + star[..., None] * 220
    im = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).convert('RGBA')
    # god rays
    if not p['stars']:
        rays = Image.new('L', (w, h), 0)
        dr = ImageDraw.Draw(rays)
        for k in range(14):
            ang = math.pi * (0.95 + 1.1 * rng.random())
            spread = 0.02 + 0.03 * rng.random()
            L = h * 1.4
            pts = [(sx, sy), (sx + L * math.cos(ang - spread), sy - L * math.sin(ang - spread)),
                   (sx + L * math.cos(ang + spread), sy - L * math.sin(ang + spread))]
            dr.polygon(pts, fill=int(18 + 22 * rng.random()))
        rays = rays.filter(ImageFilter.GaussianBlur(30))
        sunc = Image.new('RGBA', (w, h), p['sun'] + (255,))
        im = Image.composite(sunc, im, rays.point(lambda v: v))
    return im, (sx, sy)


def mountain_layer(w, h, p, rng, base_y, amp, depth, rough=0.55, trees=False):
    """depth 0 (far) .. 1 (near) -> color fades to fog"""
    pad = int(w * 0.12)
    W = w + 2 * pad
    r = ridge(W, rng, rough)
    ys = h * base_y - (r - r.min()) / (np.ptp(r) + 1e-6) * h * amp
    mask = Image.new('L', (W, h), 0)
    poly = [(x, ys[x]) for x in range(W)] + [(W, h), (0, h)]
    ImageDraw.Draw(mask).polygon(poly, fill=255)
    if trees:
        d = ImageDraw.Draw(mask)
        x = 0
        while x < W:
            th = h * (0.025 + 0.05 * rng.random())
            tw = th * 0.32
            yb = ys[min(int(x), W - 1)] + 6
            for k in range(4):
                f = 1 - k * 0.22
                d.polygon([(x, yb - th * (1 - k * 0.24) - th * 0.15), (x - tw * f, yb - th * k * 0.2), (x + tw * f, yb - th * k * 0.2)], fill=255)
            x += tw * (0.9 + 1.4 * rng.random())
    col_near = np.array(p['land'], np.float32)
    col_far = np.array(p['fog'], np.float32) * 0.55 + np.array(p['hor'], np.float32) * 0.25
    c = col_far + (col_near - col_far) * depth
    # vertical gradient inside the layer (lighter at the ridge top -> haze)
    yy = np.linspace(0, 1, h)[:, None]
    grad = np.clip((yy - (base_y - amp)) / (amp + 0.25), 0, 1)
    top_c = c + (np.array(p['fog'], np.float32) - c) * (0.35 * (1 - depth))
    arr = top_c * (1 - grad[..., None]) + c * grad[..., None]
    arr = np.repeat(arr, W, axis=1)
    im = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).convert('RGBA')
    im.putalpha(mask)
    return im, pad


def fog_layer(w, h, p, rng, y0, y1, strength=0.55):
    n = fbm2d(w, h, rng, 5, 2)
    yy = np.linspace(0, 1, h)[:, None]
    band = np.exp(-((yy - (y0 + y1) / 2) / ((y1 - y0) / 2 + 1e-6)) ** 2)
    a = np.clip(n * 1.3 - 0.25, 0, 1) * band * strength
    im = Image.new('RGBA', (w, h), p['fog'] + (0,))
    im.putalpha(Image.fromarray((a * 255).astype(np.uint8)))
    return im


def water_layer(w, h, p, sky_img, y0, sun_xy):
    """lake: mirror of the sky with ripples"""
    hh = h - int(h * y0)
    refl = sky_img.crop((0, int(h * y0) - hh, w, int(h * y0))).transpose(Image.FLIP_TOP_BOTTOM).convert('RGB')
    arr = np.asarray(refl, np.float32) * 0.75
    rows = np.arange(hh)
    shift = (np.sin(rows * 0.35) * (rows / hh) * 14).astype(int)
    for i in range(hh):
        arr[i] = np.roll(arr[i], shift[i], axis=0)
    # sun glitter column
    X = np.arange(w)
    col = np.exp(-((X - sun_xy[0]) / (w * 0.06)) ** 2)
    rip = (np.sin(rows[:, None] * 0.9 + X[None, :] * 0.05) > 0.6) * col[None, :] * 90
    arr = arr + rip[..., None] * np.array(p['sun'], np.float32) / 255
    im = Image.new('RGBA', (w, h), (0, 0, 0, 0))
    im.paste(Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)), (0, int(h * y0)))
    return im


def scene_layers(theme, w, h, seed=0):
    rng = np.random.default_rng(seed)
    pal_name, kind = pick(theme, seed)
    p = PALETTES[pal_name]
    sky, sun = sky_layer(w, h, p, rng)
    layers = [(sky, 0.0)]
    def add_m(base_y, amp, depth, par, rough=0.55, trees=False):
        im, pad = mountain_layer(w, h, p, rng, base_y, amp, depth, rough, trees)
        layers.append((im, par, pad))
    if kind == 'peaks':
        add_m(0.62, 0.20, 0.15, 0.15, 0.6)
        layers.append((fog_layer(w, h, p, rng, 0.55, 0.68, 0.5), 0.25))
        add_m(0.72, 0.16, 0.5, 0.45, 0.55)
        add_m(0.84, 0.12, 1.0, 0.9, 0.5, trees=True)
    elif kind == 'lake':
        add_m(0.60, 0.12, 0.25, 0.15, 0.5)
        add_m(0.635, 0.06, 0.7, 0.4, 0.45, trees=True)
        layers.append((water_layer(w, h, p, sky, 0.635, sun), 0.0))
        layers.append((fog_layer(w, h, p, rng, 0.6, 0.67, 0.45), 0.3))
    elif kind == 'valley':
        add_m(0.60, 0.18, 0.2, 0.15, 0.6)
        add_m(0.70, 0.14, 0.55, 0.45, 0.5)
        layers.append((fog_layer(w, h, p, rng, 0.66, 0.76, 0.55), 0.5))
        add_m(0.90, 0.10, 1.0, 0.95, 0.45, trees=True)
    elif kind == 'hills':
        add_m(0.62, 0.08, 0.25, 0.15, 0.35)
        add_m(0.72, 0.09, 0.55, 0.4, 0.35)
        layers.append((fog_layer(w, h, p, rng, 0.68, 0.76, 0.35), 0.45))
        add_m(0.86, 0.08, 1.0, 0.9, 0.3)
    else:  # forest
        add_m(0.58, 0.10, 0.2, 0.15, 0.5, trees=True)
        layers.append((fog_layer(w, h, p, rng, 0.52, 0.66, 0.6), 0.25))
        add_m(0.72, 0.10, 0.6, 0.45, 0.5, trees=True)
        layers.append((fog_layer(w, h, p, rng, 0.68, 0.80, 0.5), 0.6))
        add_m(0.88, 0.08, 1.0, 0.95, 0.45, trees=True)
    return layers, pal_name, kind


def finish(img, rng_seed=0, darken_center=0.42):
    w, h = img.size
    arr = np.asarray(img.convert('RGB'), np.float32)
    yy = np.linspace(-1, 1, h)[:, None]; xx = np.linspace(-1, 1, w)[None, :]
    vig = np.clip(1 - 0.38 * (xx ** 2 + yy ** 2), 0.45, 1)
    # darken the text area (centre) for legibility
    band = 1 - darken_center * np.exp(-((yy + 0.05) / 0.75) ** 2)
    arr = arr * (vig * band)[..., None]
    g = np.random.default_rng(rng_seed).normal(0, 6, (h, w, 1))
    arr = arr + g
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def compose(layers, w, h, t=0.0, shift=60, zoom=0.0):
    """t in 0..1 drives parallax; returns RGBA canvas (no grain)"""
    out = Image.new('RGBA', (w, h), (0, 0, 0, 255))
    for item in layers:
        im, par = item[0], item[1]
        pad = item[2] if len(item) > 2 else 0
        dx = int(pad - shift * par * t) if pad else 0
        if pad:
            im = im.crop((dx, 0, dx + w, h))
        if zoom and par == 0.0:
            z = 1 + zoom * t
            bw, bh = int(w * z), int(h * z)
            im = im.resize((bw, bh)).crop(((bw - w) // 2, (bh - h) // 2, (bw - w) // 2 + w, (bh - h) // 2 + h))
        out.alpha_composite(im)
    return out


def background(theme, w, h, seed=0):
    layers, *_ = scene_layers(theme, w, h, seed)
    return finish(compose(layers, w, h, 0.5), seed)
