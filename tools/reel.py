#!/usr/bin/env python3
"""fearnot.daily – generator of a 9:16 Reel (MP4) and a 3-slide carousel from verse.json.

Usage:
  python3 reel.py --verse verse.json --datum 2026-11-17 [--znacka fearnot.daily] [--vystup vystup] [--karusel]

Reel (~9.5 s, 1080x1920, 30 fps, H.264 + silent AAC):
  1) hook – words pop in one by one          0.0 – 2.4 s
  2) verse – lines fade in, ref in gold      2.4 – 7.2 s
  3) short line + question + follow CTA      7.2 – 9.6 s
Carousel (1080x1350 JPEG x3): hook / verse / reflection + question + CTA.
Text comes from the record's "en" block (hook, text, ref, kratce, uvaha, otazka).
"""
import argparse, json, os, re, subprocess, sys
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import hashlib
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bg as scenes

W, H, FPS = 1080, 1920, 30
BG_TOP, BG_BOT = (16, 18, 24), (6, 7, 10)
WHITE, GREY, GOLD = (244, 242, 236), (160, 160, 168), (232, 184, 74)
FONT_DIR = '/usr/share/fonts/truetype/dejavu/'
F_SERIF_B = FONT_DIR + 'DejaVuSerif-Bold.ttf'
F_SERIF = FONT_DIR + 'DejaVuSerif.ttf'
F_SANS_B = FONT_DIR + 'DejaVuSans-Bold.ttf'
F_SANS = FONT_DIR + 'DejaVuSans.ttf'


def font(path, size):
    return ImageFont.truetype(path, size)


def gradient(w, h):
    im = Image.new('RGB', (w, h))
    px = im.load()
    for y in range(h):
        t = y / (h - 1)
        c = tuple(int(BG_TOP[i] * (1 - t) + BG_BOT[i] * t) for i in range(3))
        for x in range(w):
            px[x, y] = c
    # soft vignette glow in the upper third
    glow = Image.new('L', (w, h), 0)
    ImageDraw.Draw(glow).ellipse([-w * 0.3, -h * 0.15, w * 1.3, h * 0.55], fill=38)
    glow = glow.filter(ImageFilter.GaussianBlur(160))
    gold = Image.new('RGB', (w, h), GOLD)
    return Image.composite(gold, im, glow)


def wrap(draw, text, fnt, max_w):
    words, lines, cur = text.split(), [], ''
    for wd in words:
        t = (cur + ' ' + wd).strip()
        if draw.textlength(t, font=fnt) <= max_w:
            cur = t
        else:
            if cur:
                lines.append(cur)
            cur = wd
    if cur:
        lines.append(cur)
    # avoid a lonely last word
    if len(lines) > 1 and len(lines[-1].split()) == 1:
        prev = lines[-2].split()
        if len(prev) > 2:
            lines[-2] = ' '.join(prev[:-1]); lines[-1] = prev[-1] + ' ' + lines[-1]
    return lines


def fit(draw, text, path, size, max_w, max_lines):
    while size > 28:
        f = font(path, size)
        lines = wrap(draw, text, f, max_w)
        if len(lines) <= max_lines:
            return f, lines
        size -= 4
    f = font(path, size)
    return f, wrap(draw, text, f, max_w)


def strip_stars(s):
    return s.replace('*', '')


def ease(t):
    t = max(0.0, min(1.0, t))
    return 1 - (1 - t) ** 3


def text_layer(size, items):
    """items: list of (text, font, color, (x,y), alpha 0..1, anchor)"""
    layer = Image.new('RGBA', size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    for txt, f, col, (x, y), a, anchor in items:
        if a <= 0:
            continue
        d.text((x, y), txt, font=f, fill=col + (int(255 * a),), anchor=anchor)
    return layer


def seed_for(rec):
    return int(hashlib.md5((rec.get('datum', '') + rec.get('ref', '')).encode()).hexdigest()[:8], 16)


class Finisher:
    """vignette + centre darkening + film grain, precomputed for speed"""
    def __init__(self, w, h, seed, darken=0.42, center=0.36):
        yy = np.linspace(-1, 1, h)[:, None]; xx = np.linspace(-1, 1, w)[None, :]
        vig = np.clip(1 - 0.38 * (xx ** 2 + yy ** 2), 0.45, 1)
        band = 1 - darken * np.exp(-((yy - (center * 2 - 1)) / 0.6) ** 2)
        bottom = 1 - 0.45 * np.clip((yy - 0.6) / 0.4, 0, 1)   # darker footer for brand/CTA
        self.mul = (vig * band * bottom)[..., None].astype(np.float32)
        r = np.random.default_rng(seed)
        self.grain = [r.normal(0, 6, (h, w, 1)).astype(np.float32) for _ in range(4)]
        self.i = 0

    def __call__(self, img):
        arr = np.asarray(img.convert('RGB'), np.float32) * self.mul + self.grain[self.i % 4]
        self.i += 1
        return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def brand(d, znacka, w, h, color=GREY, size=34, y_off=90):
    d.text((w / 2, h - y_off), '@' + znacka, font=font(F_SANS_B, size), fill=color, anchor='ms')


def render_reel(rec, znacka, out_mp4):
    en = rec['en']
    hook = strip_stars(en['hook'])
    verse = en['text'].strip().strip('“”"')
    ref = en['ref'] + ' (KJV)'
    kratce = en.get('kratce', '')
    otazka = en.get('otazka', '')
    sd = seed_for(rec)
    layers, pal, kind = scenes.scene_layers(rec.get('tema', '') + ' ' + en.get('hook', ''), W, H, sd)
    fin = Finisher(W, H, sd)
    probe = ImageDraw.Draw(Image.new('RGB', (10, 10)))
    MW = W - 2 * 110
    # scene 1 – hook
    fh, hook_lines = fit(probe, hook, F_SERIF_B, 92, MW, 5)
    lh1 = int(fh.size * 1.25)
    y1 = H * 0.34 - lh1 * len(hook_lines) / 2
    hook_words = []  # (word, x_center_of_line, line_index)
    for i, ln in enumerate(hook_lines):
        hook_words.append((ln.split(), i))
    total_words = sum(len(w) for w, _ in hook_words)
    # scene 2 – verse
    fv, vlines = fit(probe, '“' + verse + '”', F_SERIF, 62, MW, 11)
    lh2 = int(fv.size * 1.42)
    fref = font(F_SANS_B, 40)
    block_h = lh2 * len(vlines) + 90
    y2 = H * 0.36 - block_h / 2
    # scene 3
    parts = re.split(r'(\*[^*]+\*)', kratce)
    fk, klines = fit(probe, strip_stars(kratce), F_SERIF_B, 78, MW, 4)
    fq, qlines = fit(probe, otazka, F_SANS, 46, MW, 4)
    star_word = next((p.strip('*') for p in parts if p.startswith('*')), None)
    fcta = font(F_SANS_B, 44)

    T1, T2, T3 = 2.4, 7.2, 9.6
    n = int(T3 * FPS)
    cmd = ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}',
           '-r', str(FPS), '-i', '-', '-f', 'lavfi', '-i', 'anullsrc=channel_layout=stereo:sample_rate=44100',
           '-shortest', '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '20', '-pix_fmt', 'yuv420p',
           '-c:a', 'aac', '-b:a', '96k', '-movflags', '+faststart', out_mp4]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for k in range(n):
        t = k / FPS
        frame = fin(scenes.compose(layers, W, H, t / T3, shift=110, zoom=0.05))
        items = []
        if t < T1 + 0.3:
            fade_out = 1 - ease((t - T1 + 0.1) / 0.4) if t > T1 - 0.1 else 1
            shown = int(total_words * min(1, t / (T1 * 0.7))) + 1
            idx = 0
            for words, li in hook_words:
                line = ' '.join(words)
                lw = probe.textlength(line, font=fh)
                x = W / 2 - lw / 2
                for wd in words:
                    a = (1.0 if idx < shown - 1 else ease((t - (idx / total_words) * T1 * 0.7) / 0.15)) * fade_out
                    items.append((wd, fh, WHITE, (x, y1 + li * lh1), max(0, a), 'la'))
                    x += probe.textlength(wd + ' ', font=fh)
                    idx += 1
        if T1 - 0.1 < t < T2 + 0.3:
            fade_out = 1 - ease((t - T2 + 0.1) / 0.4) if t > T2 - 0.1 else 1
            for i, ln in enumerate(vlines):
                start = T1 + i * (min(0.38, 3.2 / max(1, len(vlines))))
                a = ease((t - start) / 0.45) * fade_out
                items.append((ln, fv, WHITE, (W / 2, y2 + i * lh2), a, 'ma'))
            start = T1 + len(vlines) * min(0.38, 3.2 / max(1, len(vlines))) + 0.1
            items.append(('— ' + ref, fref, GOLD, (W / 2, y2 + lh2 * len(vlines) + 40), ease((t - start) / 0.45) * fade_out, 'ma'))
        if t > T2 - 0.1:
            a = ease((t - T2) / 0.45)
            yk = H * 0.22
            for i, ln in enumerate(klines):
                # gold for the *star* word, white otherwise
                lw = probe.textlength(ln, font=fk)
                x = W / 2 - lw / 2
                for wd in ln.split():
                    col = GOLD if star_word and wd.strip('.,!?;:') in star_word.split() else WHITE
                    items.append((wd, fk, col, (x, yk + i * fk.size * 1.25), a, 'la'))
                    x += probe.textlength(wd + ' ', font=fk)
            yq = yk + len(klines) * fk.size * 1.25 + 80
            a2 = ease((t - T2 - 0.6) / 0.45)
            for i, ln in enumerate(qlines):
                items.append((ln, fq, GREY, (W / 2, yq + i * fq.size * 1.35), a2, 'ma'))
            a3 = ease((t - T2 - 1.2) / 0.45)
            items.append(('Follow for a verse every day', fcta, GOLD, (W / 2, H * 0.86), a3, 'ma'))
        frame = Image.alpha_composite(frame.convert('RGBA'), text_layer((W, H), items)).convert('RGB')
        d = ImageDraw.Draw(frame)
        brand(d, znacka, W, H, y_off=110)
        p.stdin.write(frame.tobytes())
    p.stdin.close()
    if p.wait() != 0:
        sys.exit('CHYBA: ffmpeg selhal')


def render_carousel(rec, znacka, outdir):
    en = rec['en']
    CW, CH = 1080, 1350
    MW = CW - 2 * 100
    sd = seed_for(rec)
    layers, *_ = scenes.scene_layers(rec.get('tema', '') + ' ' + en.get('hook', ''), CW, 1920, sd)
    tall = scenes.compose(layers, CW, 1920, 0.5).crop((0, 200, CW, 200 + CH))
    fin = Finisher(CW, CH, sd, darken=0.5, center=0.33)
    bg = fin(tall)
    files = []
    # slide 1 – hook
    im = bg.copy(); d = ImageDraw.Draw(im)
    f, lines = fit(d, strip_stars(en['hook']), F_SERIF_B, 96, MW, 5)
    y = CH * 0.32 - len(lines) * f.size * 1.25 / 2
    for i, ln in enumerate(lines):
        d.text((CW / 2, y + i * f.size * 1.25), ln, font=f, fill=WHITE, anchor='ma')
    d.text((CW / 2, CH - 160), 'swipe →', font=font(F_SANS_B, 40), fill=GOLD, anchor='ms')
    brand(d, znacka, CW, CH); files.append(im)
    # slide 2 – verse
    im = bg.copy(); d = ImageDraw.Draw(im)
    f, lines = fit(d, '“' + en['text'].strip().strip('“”"') + '”', F_SERIF, 64, MW, 12)
    lh = f.size * 1.42
    y = CH * 0.33 - (len(lines) * lh + 90) / 2
    for i, ln in enumerate(lines):
        d.text((CW / 2, y + i * lh), ln, font=f, fill=WHITE, anchor='ma')
    d.text((CW / 2, y + len(lines) * lh + 40), '— ' + en['ref'] + ' (KJV)', font=font(F_SANS_B, 40), fill=GOLD, anchor='ma')
    brand(d, znacka, CW, CH); files.append(im)
    # slide 3 – reflection + question + CTA
    im = bg.copy(); d = ImageDraw.Draw(im)
    f, lines = fit(d, en.get('uvaha', ''), F_SERIF, 52, MW, 9)
    y = 140
    for i, ln in enumerate(lines):
        d.text((CW / 2, y + i * f.size * 1.4), ln, font=f, fill=WHITE, anchor='ma')
    y += len(lines) * f.size * 1.4 + 70
    fq, ql = fit(d, en.get('otazka', ''), F_SANS_B, 46, MW, 4)
    for i, ln in enumerate(ql):
        d.text((CW / 2, y + i * fq.size * 1.35), ln, font=fq, fill=GOLD, anchor='ma')
    d.text((CW / 2, CH - 160), 'Save it · Share it · Follow for daily verses', font=font(F_SANS, 34), fill=GREY, anchor='ms')
    brand(d, znacka, CW, CH); files.append(im)
    out = []
    for i, im in enumerate(files, 1):
        pth = os.path.join(outdir, f'en_karusel_{i}.jpg')
        im.save(pth, quality=92)
        out.append(pth)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--verse', default='verse.json')
    ap.add_argument('--datum', required=True)
    ap.add_argument('--znacka', default='fearnot.daily')
    ap.add_argument('--vystup', default='vystup')
    ap.add_argument('--karusel', action='store_true', help='also/only render the 3-slide carousel')
    ap.add_argument('--jen-karusel', action='store_true')
    a = ap.parse_args()
    data = json.load(open(a.verse, encoding='utf-8'))
    recs = data if isinstance(data, list) else data.get('verse', data.get('verses', []))
    rec = next((r for r in recs if r.get('datum') == a.datum), None)
    if not rec or 'en' not in rec:
        sys.exit(f'CHYBA: ve {a.verse} není záznam s datem {a.datum} a blokem "en"')
    for key in ('hook', 'text', 'ref', 'kratce', 'otazka'):
        if not rec['en'].get(key):
            sys.exit(f'CHYBA: chybí en.{key}')
    outdir = os.path.join(a.vystup, a.datum)
    os.makedirs(outdir, exist_ok=True)
    if not a.jen_karusel:
        mp4 = os.path.join(outdir, 'en_reel.mp4')
        render_reel(rec, a.znacka, mp4)
        # cover frame for review (verse scene)
        subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-ss', '6.5', '-i', mp4, '-frames:v', '1',
                        os.path.join(outdir, 'en_reel_nahled.jpg')], check=True)
        print('OK reel', mp4)
    if a.karusel or a.jen_karusel:
        print('OK karusel', *render_carousel(rec, a.znacka, outdir))


if __name__ == '__main__':
    main()
