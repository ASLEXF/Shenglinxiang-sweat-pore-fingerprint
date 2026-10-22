# End-to-end checks for palmprint.exe: build, CLI, determinism, mirroring, layers, height
# map, image statistics, a 300-seed crash sweep and template frequencies.
# Needs numpy and Pillow. Run from anywhere: python palmprint_generator/tests/run_tests.py
import collections, json, os, re, subprocess, time
import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEP = os.path.join(ROOT, "deployment")
EXE = os.path.join(DEP, "palmprint.exe")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_out")
os.makedirs(OUT, exist_ok=True)
results = []

def check(name, ok, detail=""):
    results.append((name, ok, detail))
    print(("PASS " if ok else "FAIL ") + name + ("  -- " + detail if detail else ""), flush=True)

def run(args, timeout=600):
    t = time.time()
    p = subprocess.run([EXE] + args, cwd=DEP, capture_output=True, text=True, timeout=timeout)
    return p.returncode, p.stdout + p.stderr, time.time() - t

def load(path):
    return np.asarray(Image.open(path).convert("L"))

# 1. clean build
p = subprocess.run(["cmd", "/c", os.path.join(ROOT, "build.bat")], capture_output=True, text=True)
check("build.bat prints BUILD_OK", "BUILD_OK" in p.stdout, f"exit {p.returncode}")
check("no .obj left in deployment", not any(f.endswith(".obj") for f in os.listdir(DEP)))

# 2. CLI argument handling
rc, out, _ = run(["-h"]); check("-h exits 0 with usage", rc == 0 and "usage" in out, f"exit {rc}")
rc, out, _ = run([]); check("no args exits 1", rc == 1, f"exit {rc}")
rc, out, _ = run(["-o", "x.jpg", "-bogus"]); check("unknown arg exits 1", rc == 1, f"exit {rc}")
rc, out, _ = run(["-o"]); check("missing value exits 1", rc == 1, f"exit {rc}")
rc, out, _ = run(["-o", os.path.join(OUT, "d39.png"), "-dpi", "39"]); check("dpi 39 rejected", rc == 1, f"exit {rc}")
rc, out, _ = run(["-o", os.path.join(OUT, "bad.xyz"), "-dpi", "60"]); check("unknown image extension fails cleanly (exit 1, no crash)", rc == 1, f"exit {rc} {out.strip().splitlines()[-1] if out.strip() else ''}")

# 3. determinism
a = os.path.join(OUT, "det_a.png"); b = os.path.join(OUT, "det_b.png")
run(["-o", a, "-seed", "1"]); run(["-o", b, "-seed", "1"])
check("same seed -> identical image", np.array_equal(load(a), load(b)))

# 4. right hand is the exact mirror of the left hand
r = os.path.join(OUT, "det_r.png"); run(["-o", r, "-seed", "1", "-right"])
check("-right == horizontal flip of left", np.array_equal(load(r), load(a)[:, ::-1]))

# 5. different seeds differ
c = os.path.join(OUT, "det_c.png"); run(["-o", c, "-seed", "2"])
check("seed 2 differs from seed 1", not np.array_equal(load(a), load(c)) if load(a).shape == load(c).shape else True)

# 6. layers
ld = os.path.join(OUT, "layers1")
rc, out, _ = run(["-o", os.path.join(OUT, "lay.png"), "-seed", "1", "-layers", ld])
names = ["regions.png", "orientation.png", "period.png", "ridges.png", "creases.png", "palmprint.jpg"]
missing = [n for n in names if not os.path.exists(os.path.join(ld, n))]
check("-layers writes all 6 files", not missing, "missing " + ",".join(missing) if missing else "")
img = load(os.path.join(OUT, "lay.png"))
shapes = {n: Image.open(os.path.join(ld, n)).size for n in names if n not in missing}
check("layer sizes match image", all(s == (img.shape[1], img.shape[0]) for s in shapes.values()), str(set(shapes.values())))

# 7. image sanity on seed 1 (250 dpi)
reg = np.asarray(Image.open(os.path.join(ld, "regions.png")).convert("RGB"))
hand = reg.any(axis=2)
cre = load(os.path.join(ld, "creases.png")) > 0
bg = img[~hand]
check("background outside hand is white", (bg >= 250).mean() > 0.999, f"{(bg >= 250).mean():.5f}")
inside = img[hand & ~cre]
dark = (inside < 128).mean()
check("ridge ink coverage inside hand 30-65%", 0.30 < dark < 0.65, f"{dark:.3f}")
check("creases cover 1-8% of hand", 0.01 < cre[hand].mean() < 0.08, f"{cre[hand].mean():.4f}")
check("creases only inside hand", not (cre & ~hand).any())
# hand shape varies with seed (size 0.9-1.1, width 0.93-1.07, spread 0.75-1.3)
check("250 dpi canvas is 12-21 x 16-25 cm", 12 < img.shape[1] / 98.425 < 21 and 16 < img.shape[0] / 98.425 < 25, str(img.shape[::-1]))

# 8. crash sweep + template stats, 300 seeds at 100 dpi (generic preset)
fails, warns = [], []
lines = collections.Counter(); fingers = collections.Counter(); palm = collections.Counter()
thumb_seq, dims, sexes = [], [], collections.Counter()
t0 = time.time()
N = 300
for s in range(1, N + 1):
    rc, out, _ = run(["-o", os.path.join(OUT, "sweep.png"), "-dpi", "100", "-seed", str(s), "-population", "generic"])
    if rc != 0: fails.append((s, rc))
    if "warning" in out: warns.append(s)
    m = re.search(r"^lines(.*)$", out, re.M)
    if m:
        for tok in m.group(1).split():
            lines[tok.split("/")[0]] += 1
            if "/" in tok: lines[tok.split("=")[0] + "=" + tok.split("/")[1]] += 1
    m = re.search(r"^fingers(.*)$", out, re.M)
    if m:
        for tok in m.group(1).split(): fingers[tok] += 1
        thumb_seq.append(m.group(1).split()[0].split("=")[1])
    m = re.search(r"^palm thenar(.*)$", out, re.M)
    if m:
        for tok in ("thenar" + m.group(1)).split(): palm[tok] += 1
    m = re.search(r"sex=(\w) .*hand=([0-9.]+) palm=([0-9.]+) breadth=([0-9.]+) middle=([0-9.]+) atd=([0-9]+)", out)
    if m:
        sexes[m.group(1)] += 1
        dims.append((m.group(1),) + tuple(float(v) for v in m.groups()[1:]))
check(f"{N} seeds at 100 dpi all exit 0", not fails, str(fails[:10]) + f" ({time.time()-t0:.0f}s)")
check("no crease-start warnings", not warns, str(warns[:10]))
def share(finger, *patterns):
    return sum(fingers[f"{finger}={p}"] for p in patterns) / N
check("little finger loops > 80% (lit. 86-90%)", share("little", "left-loop", "right-loop") > 0.8, f"{share('little', 'left-loop', 'right-loop'):.3f}")
check("ring finger whorls > 35% (lit. 45-56%)", share("ring", "whorl", "twin-loop") > 0.35, f"{share('ring', 'whorl', 'twin-loop'):.3f}")
check("thumb whorls 30-50% (lit. 30-46%)", 0.3 < share("thumb", "whorl", "twin-loop") < 0.5, f"{share('thumb', 'whorl', 'twin-loop'):.3f}")
radial_index = share("index", "right-loop")
radial_other = max(share(f, "right-loop") for f in ("middle", "ring", "little"))
check("radial loops concentrated on the index", radial_index > 2 * radial_other, f"index {radial_index:.3f}, max other {radial_other:.3f}")
arch_index = share("index", "arch", "tented")
check("arches concentrated on the index (lit. 10-17%)", arch_index > max(share(f, "arch", "tented") for f in ("thumb", "ring", "little")), f"index {arch_index:.3f}")
total = sum(fingers.values())
loops = sum(v for k, v in fingers.items() if "loop" in k and "twin" not in k) / total
check("all-finger loops 55-70% (lit. ~60%)", 0.55 < loops < 0.70, f"{loops:.3f}")
check("I4 loops 30-65% of palms", 0.3 < palm["I4=1"] / N < 0.65, f"{palm['I4=1'] / N:.3f}")
check("both sexes drawn, 40-60% male", 0.4 < sexes["M"] / N < 0.6, str(dict(sexes)))
for sex, lo, hi in (("F", 16.5, 19.5), ("M", 17.8, 20.8)):
    hand = np.mean([d[1] for d in dims if d[0] == sex])
    check(f"mean hand length {sex} in {lo}-{hi} cm", lo < hand < hi, f"{hand:.2f}")
breadth = {sex: np.mean([d[3] for d in dims if d[0] == sex]) for sex in "FM"}
check("mean breadth 7-9.5 cm, men wider", all(7 < b < 9.5 for b in breadth.values()) and breadth["M"] > breadth["F"], str({k: round(v, 2) for k, v in breadth.items()}))
atd = np.mean([d[5] for d in dims])
check("mean atd 33-48 deg (lit. ~41)", 33 < atd < 48, f"{atd:.1f}")
print("line template counts:", dict(sorted(lines.items())))
print("finger pattern counts:", dict(sorted(fingers.items())))
print("palm pattern counts:", dict(sorted(palm.items())))

# 8a. fingertip pore density at 600 dpi, one woman and one man (han seeds 1 and 4). The
# literature figures (F 52 / M 45) are matched at 1000 dpi (measured 49.0 / 42.2, too slow
# for this suite); density reads about 30% higher at 600 dpi, so the ranges here are
# regression bounds around the 600 dpi readings (63.8 / 54.5), not a literature check.
for seed, sex, lo, hi in (("1", "F", 57, 70), ("4", "M", 48, 61)):
    rc, out, _ = run(["-o", os.path.join(OUT, "pores.png"), "-dpi", "600", "-seed", seed, "-population", "han"])
    m = re.search(r"distal=([0-9.]+)/9mm2", out)
    check(f"600 dpi fingertip pores {sex} {lo}-{hi} per 9 mm2 (1000 dpi: lit. F 52 / M 45)",
          f"sex={sex}" in out and m and lo < float(m.group(1)) < hi, (m.group(0) if m else out[-200:]) + (" sex mismatch" if f"sex={sex}" not in out else ""))

# 8a2. presets are salted: the same seed gives a different hand under another preset
diff = 0
for s in range(1, 51):
    a = re.search(r"^fingers.*$", run(["-o", os.path.join(OUT, "pa.png"), "-dpi", "40", "-seed", str(s), "-population", "han"])[1], re.M).group(0)
    b = re.search(r"^fingers.*$", run(["-o", os.path.join(OUT, "pb.png"), "-dpi", "40", "-seed", str(s), "-population", "japanese"])[1], re.M).group(0)
    diff += a != b
check("han vs japanese: finger patterns differ for > 40% of seeds", diff > 20, f"{diff}/50")

# 8b2. each named preset reproduces its own table: whorl share per finger, I4 loops and
# mean hand length by sex, over 200 seeds at 60 dpi
PRESETS = {
    # whorl + twin loop per finger (thumb..little), I4 %, hand mean F/M cm
    "han": ([52.9, 46.2, 36.7, 61.6, 24.3], 68.01, (17.1, 18.3)),
    "european": ([43.0, 33.2, 17.2, 42.0, 14.9], 45.85, (18.0, 19.3)),
    "japanese": ([55.3, 45.3, 34.3, 59.5, 28.3], 79.0, (17.54, 18.96)),
}
for name, (whorls, i4, hands) in PRESETS.items():
    M = 200
    cnt = collections.Counter(); i4n = 0; hl = {"F": [], "M": []}; bad = 0
    for s in range(1, M + 1):
        rc, out, _ = run(["-o", os.path.join(OUT, "preset.png"), "-dpi", "60", "-seed", str(s), "-population", name])
        if rc != 0 or f"population={name}" not in out: bad += 1; continue
        for tok in re.search(r"^fingers(.*)$", out, re.M).group(1).split(): cnt[tok] += 1
        i4n += "I4=1" in out
        m = re.search(r"sex=(\w) .*hand=([0-9.]+)", out); hl[m.group(1)].append(float(m.group(2)))
    got = [100 * (cnt[f"{f}=whorl"] + cnt[f"{f}=twin-loop"]) / M for f in ("thumb", "index", "middle", "ring", "little")]
    ok = bad == 0 and all(abs(g - w) < 11 for g, w in zip(got, whorls))
    check(f"{name}: per-finger whorl share within 11 points of table", ok, " ".join(f"{g:.0f}/{w:.0f}" for g, w in zip(got, whorls)))
    check(f"{name}: I4 loops within 11 points of {i4:.0f}%", abs(100 * i4n / M - i4) < 11, f"{100 * i4n / M:.1f}")
    means = [np.mean(hl[k]) for k in ("F", "M")]
    check(f"{name}: mean hand F/M within 0.5 cm of {hands}", all(abs(m - h) < 0.5 for m, h in zip(means, hands)), f"{means[0]:.2f}/{means[1]:.2f}")

# 8c. population flag
rc, out, _ = run(["-o", os.path.join(OUT, "pop.png"), "-dpi", "60", "-population", "generic"])
check("-population generic accepted", rc == 0 and "population=generic" in out, f"exit {rc}")
rc, out, _ = run(["-o", os.path.join(OUT, "pop.png"), "-dpi", "60", "-population", "martian"])
check("unknown population rejected", rc == 1, f"exit {rc}")

# 8b. hand outline varies with seed: hand area and canvas size spread over 12 seeds
areas, sizes = [], set()
for s in range(1, 13):
    rc, out, _ = run(["-o", os.path.join(OUT, "shape.png"), "-dpi", "100", "-seed", str(s), "-layers", os.path.join(OUT, "shape_layers")])
    reg = np.asarray(Image.open(os.path.join(OUT, "shape_layers", "regions.png")).convert("RGB"))
    areas.append(reg.any(axis=2).sum())
    sizes.add(reg.shape[:2])
cv = np.std(areas) / np.mean(areas)
check("hand area varies across seeds (CV > 5%)", cv > 0.05, f"CV {cv:.3f}, {len(sizes)} distinct canvas sizes")

# 9. edge seeds
for s in ["0", "2147483647", "4294967295"]:
    rc, out, _ = run(["-o", os.path.join(OUT, f"edge{s}.png"), "-dpi", "80", "-seed", s])
    check(f"seed {s} runs and is echoed exactly", rc == 0 and f"seed={s} " in out, f"exit {rc} seed line: {out.splitlines()[0] if out else ''}")
check("seed 4294967295 differs from 2147483647",
      not np.array_equal(load(os.path.join(OUT, "edge4294967295.png")), load(os.path.join(OUT, "edge2147483647.png"))))
for s in ["-1", "abc", "12x", "4294967296", ""]:
    rc, out, _ = run(["-o", os.path.join(OUT, "edge.png"), "-dpi", "80", "-seed", s])
    check(f"seed {s!r} rejected", rc == 1, f"exit {rc}")

# 10. dpi range + timing
for d in (40, 100, 250, 500):
    rc, out, dt = run(["-o", os.path.join(OUT, f"dpi{d}.png"), "-dpi", str(d), "-seed", "3"])
    size = Image.open(os.path.join(OUT, f"dpi{d}.png")).size if rc == 0 else None
    check(f"dpi {d} runs", rc == 0, f"{dt:.1f}s size {size}")
    check(f"dpi {d}: low-dpi pore note {'printed' if d <= 500 else 'absent'}", ("note:" in out) == (d <= 500))

# 11. jpg output path with new directory
nd = os.path.join(OUT, "newdir", "sub", "x.jpg")
rc, out, _ = run(["-o", nd, "-seed", "4", "-dpi", "80"])
check("output into non-existent directory", rc == 0 and os.path.exists(nd), f"exit {rc}")

# 12. height map (-height): 16 bit grey, mid grey = flat
def height(name, extra, seed="1", dpi="250"):
    img = os.path.join(OUT, name + ".png")
    hp = os.path.join(OUT, name + "_height.png")
    rc, out, _ = run(["-o", img, "-seed", seed, "-dpi", dpi, "-height", hp, "-layers", os.path.join(OUT, name + "_layers")] + extra)
    return rc, out, img, hp

def load_height(path):
    im = Image.open(path)
    return im, np.asarray(im).astype(int)

rc, out, img_l, hp_l = height("hmap_l", [])
check("-height writes a file and a summary line", rc == 0 and os.path.exists(hp_l) and "\nheight " in out, f"exit {rc}")
im, hei = load_height(hp_l)
check("height: single channel 16 bit, same size as the -o image",
      im.mode in ("I;16", "I") and hei.ndim == 2 and hei.max() > 255 and im.size == Image.open(img_l).size, f"{im.mode} {im.size}")
check("-height leaves the -o image unchanged", np.array_equal(load(img_l), load(os.path.join(OUT, "det_a.png"))))
ld = os.path.join(OUT, "hmap_l_layers")
hand = load(os.path.join(ld, "regions.png")) > 0
check("height: everything outside the hand is flat 32768", (~hand).sum() > 0 and (hei[~hand] == 32768).all())
edge = hand.copy()
edge[1:-1, 1:-1] &= ~(hand[:-2, 1:-1] & hand[2:, 1:-1] & hand[1:-1, :-2] & hand[1:-1, 2:])
edge[0], edge[-1], edge[:, 0], edge[:, -1] = hand[0], hand[-1], hand[:, 0], hand[:, -1]
check("height: flat at the outline (no cliff from creases, ridges or pores)",
      edge.sum() > 100 and np.abs(hei[edge] - 32768).max() < 0.1 * 32768, f"{edge.sum()} px, max dev {np.abs(hei[edge] - 32768).max()}")

# height follows the ridge layer; creases are the lowest
ridge = 255.0 - np.asarray(Image.open(os.path.join(ld, "ridges.png")).convert("L")).astype(float)
crease = np.asarray(Image.open(os.path.join(ld, "creases.png")).convert("L")) > 0
core = hand & ~crease
core[:8], core[-8:], core[:, :8], core[:, -8:] = False, False, False, False
cc = np.corrcoef(hei[core].astype(float), ridge[core])[0, 1]
check("height: correlates with the ridge layer", cc > 0.5, f"r = {cc:.2f}")
check("height: creases are lower than the furrows",
      hei[crease & hand].mean() < np.percentile(hei[core], 5), f"{hei[crease & hand].mean():.0f} vs {np.percentile(hei[core], 5):.0f}")

# fingertip pads are rougher than the palm centre (BGR palette in save_layers: distal
# (220,80,40) and central (160,160,160) read back as RGB)
reg = np.asarray(Image.open(os.path.join(ld, "regions.png")).convert("RGB")).astype(int)
pad = (reg == [220, 80, 40]).all(2) & ~crease
palm = (reg == [160, 160, 160]).all(2) & ~crease
ratio = hei[pad].std() / hei[palm].std()
pc = np.corrcoef(hei[palm].astype(float), ridge[palm])[0, 1]
check("height: palm ridges are still there (correlation with the ridge layer > 0.4)", pc > 0.4, f"r = {pc:.2f}")
check("height: fingertip pads vary more than the palm centre (std ratio > 1.15)", ratio > 1.15,
      f"pad std {hei[pad].std():.0f}, palm std {hei[palm].std():.0f}, ratio {ratio:.2f}")

# determinism, mirroring and options
rc, out, _, hp_l2 = height("hmap_l2", [])
check("height: same seed -> identical map", np.array_equal(hei, load_height(hp_l2)[1]))
rc, out, _, hp_r = height("hmap_r", ["-right"])
check("height: -right is the horizontal flip of the left hand", np.array_equal(load_height(hp_r)[1], hei[:, ::-1]))
rc, out, _, hp_rel = height("hmap_relief", ["-relief", "0.2"])
check("-relief changes the printed mm figures, not the image", np.array_equal(load_height(hp_rel)[1], hei) and "0.200 mm" in out)
rc1, o1, _ = run(["-o", os.path.join(OUT, "rs_1.png"), "-dpi", "500", "-seed", "1", "-ridge-scale", "1"])
rc2, o2, _ = run(["-o", os.path.join(OUT, "rs_08.png"), "-dpi", "500", "-seed", "1", "-ridge-scale", "0.8"])
m1, m2 = re.search(r"ridge period=([\d.]+)", o1), re.search(r"ridge period=([\d.]+)", o2)
check("-ridge-scale 0.8 shrinks the ridge period to 0.8x",
      rc1 == 0 and rc2 == 0 and m1 and m2 and abs(float(m2.group(1)) / float(m1.group(1)) - 0.8) < 0.02,
      f"{m1.group(1) if m1 else None} px -> {m2.group(1) if m2 else None} px")
rc, out_np, _, hp_np = height("hmap_np", ["-no-pores"])
hei_np = load_height(hp_np)[1]
check("-no-pores: pore count 0 (and not 0 by default)", rc == 0 and "pores count=0 " in out_np and "pores count=0 " not in o1)
# the pits are blurred, so each pore touches ~25 px and nearly every pixel differs a little;
# what matters is that the ridges are the same and the map without pores has no pits
cp = np.corrcoef(hei[core].astype(float), hei_np[core].astype(float))[0, 1]
check("-no-pores: same ridges (correlation with the default map > 0.9) and no pits (higher mean)",
      cp > 0.9 and hei_np[core].mean() > hei[core].mean(), f"r = {cp:.3f}, mean {hei_np[core].mean():.0f} vs {hei[core].mean():.0f}")
rc, out_c1, _, hp_c1 = height("hmap_c1", ["-crease-scale", "1"])
cr_default = (load(os.path.join(OUT, "hmap_l_layers", "creases.png")) > 0).sum()
cr_full = (load(os.path.join(OUT, "hmap_c1_layers", "creases.png")) > 0).sum()
check("-crease-scale: default (0.5) draws clearly thinner creases than -crease-scale 1 (area < 70%)",
      rc == 0 and 0 < cr_default < 0.7 * cr_full, f"{cr_default} vs {cr_full} px")
for bad in (["-crease-scale", "0.1"], ["-crease-scale", "1.6"], ["-crease-scale", "abc"]):
    rc, out, _ = run(["-o", os.path.join(OUT, "badmap.png"), "-dpi", "60"] + bad)
    check(f"{' '.join(bad)} rejected", rc == 1, f"exit {rc}")
for bad in (["-ridge-scale", "0.3"], ["-ridge-scale", "1.6"], ["-ridge-scale", "abc"]):
    rc, out, _ = run(["-o", os.path.join(OUT, "badmap.png"), "-dpi", "60"] + bad)
    check(f"{' '.join(bad)} rejected", rc == 1, f"exit {rc}")
for bad in (["-relief", "0"],["-relief", "abc"], ["-relief", "11"], ["-maps", "x"], ["-skin", "1,2,3"], ["-normal-dx"]):
    rc, out, _ = run(["-o", os.path.join(OUT, "badmap.png"), "-dpi", "60"] + bad)
    check(f"{' '.join(bad)} rejected", rc == 1, f"exit {rc}")
rc, out, _ = run(["-o", os.path.join(OUT, "badmap.png"), "-dpi", "60", "-dump-creases"])
check("-dump-creases without a path rejected", rc == 1, f"exit {rc}")

# 13. -dump-creases and the crease depth model (depth = 1.6 * cross(u) * along(t), editor curves)
def dc_run(name, extra):
    dc_img = os.path.join(OUT, name + ".png")
    dc_json = os.path.join(OUT, name + ".json")
    dc_rc, dc_out, _ = run(["-o", dc_img, "-seed", "1", "-height", os.path.join(OUT, name + "_height.png"),
                            "-layers", os.path.join(OUT, name + "_layers"), "-dump-creases", dc_json] + extra)
    return dc_rc, dc_img, dc_json, os.path.join(OUT, name + "_height.png"), os.path.join(OUT, name + "_layers")

def dc_offset(dc, dc_hand, dc_crease):
    # The JSON is cropped to the palm; the crop origin is not written, so take the first guess
    # from the outline extremes (approxPolyDP may move them ~1.5 px) and refine it by how many
    # stroke samples land on the crease layer.
    ys, xs = np.nonzero(dc_hand)
    ol = np.array(dc["outline"])
    gx, gy = xs.min() - ol[:, 0].min(), ys.min() - ol[:, 1].min()
    pts = np.array([p for s in dc["strokes"] for p in s["pts"]], dtype=float)
    best = (-1, 0, 0)
    for dy in range(-3, 4):
        for dx in range(-3, 4):
            x = pts[:, 0].astype(int) + gx + dx; y = pts[:, 1].astype(int) + gy + dy
            ok = (x >= 0) & (y >= 0) & (x < dc_hand.shape[1]) & (y < dc_hand.shape[0])
            x, y = x[ok], y[ok]
            inside = dc_hand[y, x]
            rate = dc_crease[y[inside], x[inside]].mean() if inside.any() else 0
            best = max(best, (rate, gx + dx, gy + dy))
    return best

dc_rc, dc_img, dc_json, dc_hp, dc_ld = dc_run("dump_l", [])
dc_ok = dc_rc == 0 and os.path.exists(dc_json)
try:
    dc = json.load(open(dc_json)) if dc_ok else {}
except ValueError as e:
    dc, dc_ok = {}, False
check("-dump-creases writes valid JSON with dpi/width/height/px_per_cm/outline/strokes",
      dc_ok and all(k in dc for k in ("dpi", "width", "height", "px_per_cm", "outline", "strokes")) and dc["dpi"] == 250,
      f"exit {dc_rc}, keys {sorted(dc)}")
dc_pts_ok = dc_ok and len(dc["strokes"]) >= 5 and all(
    len(p) == 4 and 0 <= p[3] <= 1 and p[2] >= 0 for s in dc["strokes"] for p in s["pts"]) and all(
    all(a[3] <= b[3] for a, b in zip(s["pts"], s["pts"][1:])) for s in dc["strokes"])
check("-dump-creases: >= 5 strokes, points [x,y,r>=0,t in 0..1], t non-decreasing along each piece",
      dc_pts_ok, f"{len(dc.get('strokes', []))} strokes")
check("-dump-creases leaves the -o image unchanged", dc_ok and np.array_equal(load(dc_img), load(os.path.join(OUT, "det_a.png"))))

dc_hand = load(os.path.join(dc_ld, "regions.png")) > 0
dc_crease = load(os.path.join(dc_ld, "creases.png")) > 0
dc_rate, dc_x0, dc_y0 = dc_offset(dc, dc_hand, dc_crease) if dc_ok else (0, 0, 0)
check("-dump-creases: stroke samples inside the hand lie on the crease layer (> 90%)", dc_rate > 0.9, f"{dc_rate:.3f} at crop origin ({dc_x0},{dc_y0})")

dc_rc_r, _, dc_json_r, _, dc_ld_r = dc_run("dump_r", ["-right"])
try:
    dc_r = json.load(open(dc_json_r))
    dc_rate_r = dc_offset(dc_r, load(os.path.join(dc_ld_r, "regions.png")) > 0, load(os.path.join(dc_ld_r, "creases.png")) > 0)[0]
except (OSError, ValueError):
    dc_rate_r = 0
check("-dump-creases -right: strokes are mirrored with the image (> 90% on the crease layer)", dc_rc_r == 0 and dc_rate_r > 0.9, f"{dc_rate_r:.3f}")

# depth: 1 relief unit = 32767 grey levels below 32768
dc_hei = load_height(dc_hp)[1]
dc_units = (dc_hei - 32768) / 32767.0
dc_min = dc_units[dc_hand].min()
check("crease depth: deepest point -0.8..-1.0 units and not clipped at 0", dc_hei[dc_hand].min() > 0 and -1.0 < dc_min < -0.8, f"min {dc_min:.3f}")
# along-the-stroke profile: deep in the middle, shallow at the ends (away from the outline fade)
dc_inner = dc_hand.copy()
for _ in range(int(round(0.15 * 250 / 2.54)) + 1):
    dc_inner[1:-1, 1:-1] &= dc_inner[:-2, 1:-1] & dc_inner[2:, 1:-1] & dc_inner[1:-1, :-2] & dc_inner[1:-1, 2:]
dc_mid, dc_end = [], []
for s in dc.get("strokes", []):
    for x, y, r, t in s["pts"]:
        x, y = x + dc_x0, y + dc_y0
        if 0 <= x < dc_inner.shape[1] and 0 <= y < dc_inner.shape[0] and dc_inner[y, x]:
            if 0.4 <= t <= 0.6: dc_mid.append(-dc_units[y, x])
            elif t < 0.15 or t > 0.85: dc_end.append(-dc_units[y, x])
dc_m = np.mean(dc_mid) if dc_mid else 0; dc_e = np.mean(dc_end) if dc_end else 9
check("crease depth: middle of the stroke (t 0.4-0.6) > 0.6 units and > ends (t < 0.15 or > 0.85) + 0.3",
      len(dc_mid) > 50 and len(dc_end) > 20 and dc_m > 0.6 and dc_m > dc_e + 0.3,
      f"mid {dc_m:.2f} (n={len(dc_mid)}), ends {dc_e:.2f} (n={len(dc_end)})")

print()
passed = sum(ok for _, ok, _ in results)
print(f"{passed}/{len(results)} passed")
raise SystemExit(0 if passed == len(results) else 1)
