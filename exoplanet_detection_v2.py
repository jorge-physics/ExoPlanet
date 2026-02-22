"""
Exoplanet Detection: Transit Photometry & Doppler Spectroscopy
Controls: SPACE=pause  ESC=quit  ↑↓=planet radius  drag slider=density
"""

import sys
import numpy as np
import pygame

pygame.init()

W, H   = 1600, 1100
HW, HH = W // 2, H // 2
screen  = pygame.display.set_mode((W, H))
pygame.display.set_caption("Exoplanet Detection Methods")
clock   = pygame.time.Clock()

BG          = (8,  8,  16)
PANEL_BG    = (12, 12, 24)
PANEL_DARK  = (6,  6,  14)
GRID_COL    = (28, 28, 50)
BORDER      = (65, 65, 100)
WHITE       = (255, 255, 255)
L_GRAY      = (180, 180, 200)
GRAY        = (100, 100, 130)
D_GRAY      = (55,  55,  75)
STAR_COL    = (255, 220, 60)
BARY_COL    = (200, 80,  80)
LC_LINE     = (60,  220, 120)
BLUE_COL    = (60,  140, 255)
RED_COL     = (255, 80,  60)
TRANSIT_COL = (255, 80,  80)
OCC_COL     = (180, 120, 255)

Fsm  = pygame.font.SysFont("monospace", 12)
Fmed = pygame.font.SysFont("monospace", 15)
Flg  = pygame.font.SysFont("monospace", 17, bold=True)

PERIOD      = 9.0
OMEGA       = 2 * np.pi / PERIOD
R_P         = 0.72
SCALE       = 115
R_STAR_PX   = 27
R_PLAN_PX   = 10
R_PLAN_MIN  = 3
R_PLAN_MAX  = R_STAR_PX // 2

DENS_MIN      = 0.5
DENS_MAX      = 5.0
DENSITY_RATIO = 1.0

H_LINES = [("Hd", 410.2), ("Hg", 434.0), ("Hb", 486.1), ("Ha", 656.3)]
WL_MIN, WL_MAX = 395.0, 700.0
DOPPLER_SCALE  = 14.0
RV_YLIM        = DENS_MAX * (R_PLAN_MAX / R_STAR_PX) ** 3

F_LO, F_HI    = 0.75, 1.05

HIST    = 500
lc_hist = np.ones(HIST)
rv_hist = np.zeros(HIST)
ptr     = 0

SL_PAD  = 60
SL_X    = SL_PAD
SL_Y    = HH + HH - 55
SL_W    = HW - 2 * SL_PAD
SL_H    = 8
SL_DRAG = False

def txt(surf, s, font, col, x, y, anchor="topleft"):
    img = font.render(s, True, col)
    r   = img.get_rect(**{anchor: (x, y)})
    surf.blit(img, r)

def panel(surf, x, y, w, h, title):
    pygame.draw.rect(surf, PANEL_BG,   (x, y, w, h))
    pygame.draw.rect(surf, PANEL_DARK, (x+2, y+2, w-4, 22))
    pygame.draw.rect(surf, BORDER,     (x, y, w, h), 2)
    txt(surf, title, Flg, WHITE, x+10, y+5)

def grid(surf, x, y, w, h, nx=6, ny=5):
    for i in range(1, nx):
        xp = x + w * i // nx
        pygame.draw.line(surf, GRID_COL, (xp, y), (xp, y+h))
    for i in range(1, ny):
        yp = y + h * i // ny
        pygame.draw.line(surf, GRID_COL, (x, yp), (x+w, yp))

def sphere(surf, cx, cy, radius, base_col):
    r, g, b = base_col
    steps = max(radius, 4)
    for i in range(steps, 0, -1):
        t   = i / steps
        hi  = 1.0 - t
        ox  = int(-radius * 0.25 * hi)
        oy  = int(-radius * 0.25 * hi)
        rc  = int(radius * t)
        col = (min(255, int(r + (255-r)*hi*0.9)),
               min(255, int(g + (255-g)*hi*0.9)),
               min(255, int(b + (255-b)*hi*0.9)))
        if rc > 0:
            pygame.draw.circle(surf, col, (cx+ox, cy+oy), rc)

def planet_colour(density_ratio):
    anchors = [
        (0.5, (63,  84,  186)),   # gas giant — Neptune blue
        (1.5, (80,  200, 180)),   # ice/water — teal
        (3.0, (180, 100,  60)),   # rocky     — terracotta
        (5.0, (160, 160, 175)),   # iron/metal — Mercury steel
    ]
    # clamp
    d = max(DENS_MIN, min(DENS_MAX, density_ratio))
    # find segment
    for i in range(len(anchors) - 1):
        d0, c0 = anchors[i]
        d1, c1 = anchors[i + 1]
        if d <= d1:
            t = (d - d0) / (d1 - d0)
            return tuple(int(c0[j] + (c1[j] - c0[j]) * t) for j in range(3))
    return anchors[-1][1]

def wl_to_rgb(wl):
    if   wl < 380: r, g, b = 0.5, 0.0, 0.5
    elif wl < 440: r, g, b = (440-wl)/60, 0.0, 1.0
    elif wl < 490: r, g, b = 0.0, (wl-440)/50, 1.0
    elif wl < 510: r, g, b = 0.0, 1.0, (510-wl)/20
    elif wl < 580: r, g, b = (wl-510)/70, 1.0, 0.0
    elif wl < 645: r, g, b = 1.0, (645-wl)/65, 0.0
    else:          r, g, b = 1.0, 0.0, 0.0
    if   wl < 420: f = 0.3 + 0.7*(wl-380)/40
    elif wl > 680: f = 0.3 + 0.7*(700-wl)/20
    else:          f = 1.0
    return (int(r*f*255), int(g*f*255), int(b*f*255))

def make_spec_bg(w, h):
    s = pygame.Surface((w, h))
    for xi in range(w):
        wl  = WL_MIN + xi / w * (WL_MAX - WL_MIN)
        col = tuple(max(0, int(c * 0.55)) for c in wl_to_rgb(wl))
        pygame.draw.line(s, col, (xi, 0), (xi, h))
    return s

def wl_to_x(wl, x0, w):
    return int(x0 + (wl - WL_MIN) / (WL_MAX - WL_MIN) * w)

def observer_arrow(surf, cx, cy, dx, dy, length=40, label="Observer"):
    nx, ny = dx, dy
    tip  = (int(cx + nx*length),       int(cy + ny*length))
    base = (int(cx + nx*(length-12)),  int(cy + ny*(length-12)))
    pygame.draw.line(surf, L_GRAY, (cx, cy), base, 2)
    px_, py_ = -ny*7, nx*7
    pygame.draw.polygon(surf, L_GRAY, [
        tip,
        (int(base[0]+px_), int(base[1]+py_)),
        (int(base[0]-px_), int(base[1]-py_)),
    ])
    lx = int(cx + nx*(length+4))
    ly = int(cy + ny*(length+4))
    txt(surf, label, Fsm, L_GRAY, lx, ly, anchor="center")

def draw_density_slider(surf):
    pygame.draw.rect(surf, BORDER, (SL_X, SL_Y - SL_H//2, SL_W, SL_H), 0)
    fill_w = int((DENSITY_RATIO - DENS_MIN) / (DENS_MAX - DENS_MIN) * SL_W)
    for xi in range(fill_w):
        t   = xi / max(SL_W - 1, 1)
        col = (int(140 + (120-140)*t), int(190 + (100-190)*t), int(255 + (85-255)*t))
        pygame.draw.line(surf, col,
                         (SL_X + xi, SL_Y - SL_H//2),
                         (SL_X + xi, SL_Y + SL_H//2))
    knob_x  = SL_X + fill_w
    knob_col = planet_colour(DENSITY_RATIO)
    pygame.draw.circle(surf, knob_col, (knob_x, SL_Y), 10)
    pygame.draw.circle(surf, WHITE,    (knob_x, SL_Y), 10, 2)
    txt(surf, f"Planet density:  {DENSITY_RATIO:.2f} x rho_star",
        Fmed, planet_colour(DENSITY_RATIO),
        SL_X + SL_W//2, SL_Y - 26, anchor="midtop")
    txt(surf, f"{DENS_MIN}x", Fsm, L_GRAY, SL_X,        SL_Y + 14, anchor="midtop")
    txt(surf, f"{DENS_MAX}x", Fsm, L_GRAY, SL_X + SL_W, SL_Y + 14, anchor="midtop")

def draw_edge_on(surf, theta, in_transit, in_occ):
    OX, OY, OW, OH = 0, 0, HW, HH
    panel(surf, OX, OY, OW, OH, "Edge-on View   (y -> right,  z -> up)")
    cx = OX + OW // 2
    cy = OY + OH // 2 + 14
    pygame.draw.line(surf, D_GRAY, (OX+20, cy), (OX+OW-20, cy), 1)
    txt(surf, "y", Fsm, D_GRAY, OX+OW-16, cy-14)
    pygame.draw.line(surf, D_GRAY, (cx, OY+30), (cx, OY+OH-20), 1)
    txt(surf, "z", Fsm, D_GRAY, cx+6, OY+33)
    for r, col in [(R_P*SCALE, (40, 40, 80)), (R_S*SCALE, (80, 45, 20))]:
        pygame.draw.line(surf, col, (cx - int(r), cy), (cx + int(r), cy), 1)
    xp = R_P * np.cos(theta)
    yp = R_P * np.sin(theta)
    xs = -R_S * np.cos(theta)
    ys = -R_S * np.sin(theta)
    p_sx = cx + int(yp * SCALE)
    p_sy = cy
    s_sx = cx + int(ys * SCALE)
    s_sy = cy
    bodies = [("star", s_sx, s_sy, xs), ("planet", p_sx, p_sy, xp)]
    bodies.sort(key=lambda b: b[3])
    pcol = planet_colour(DENSITY_RATIO)
    for name, bx, by, _ in bodies:
        if name == "star":
            sphere(surf, bx, by, R_STAR_PX, STAR_COL)
        else:
            sphere(surf, bx, by, R_PLAN_PX, pcol)
    pygame.draw.line(surf, BARY_COL, (cx-6, cy), (cx+6, cy), 2)
    pygame.draw.line(surf, BARY_COL, (cx, cy-6), (cx, cy+6), 2)
    obs_x = OX + OW - 30;  obs_y = cy
    pygame.draw.circle(surf, L_GRAY, (obs_x, obs_y), 9, 2)
    pygame.draw.circle(surf, L_GRAY, (obs_x, obs_y), 2)
    txt(surf, "x (obs.)", Fsm, L_GRAY, obs_x, obs_y - 20, anchor="center")
    #txt(surf, "Star",   Fsm, STAR_COL, s_sx + R_STAR_PX + 4, s_sy - 7)
    #txt(surf, "Planet", Fsm, pcol,     p_sx + R_PLAN_PX + 4, p_sy - 7)
    ratio = R_PLAN_PX / R_STAR_PX
    txt(surf, f"R_p / R_star = {ratio:.3f}", Fmed, L_GRAY, OX+12, OY+OH-55)
    if in_transit:
        txt(surf, "TRANSIT",     Fmed, TRANSIT_COL, OX+12, OY+OH-35)
    elif in_occ:
        txt(surf, "OCCULTATION", Fmed, OCC_COL,     OX+12, OY+OH-35)
    ph = np.degrees(theta % (2*np.pi))
    txt(surf, f"th = {ph:5.1f} deg", Fsm, GRAY, OX+OW-90, OY+OH-28)

def draw_top_down(surf, theta):
    OX, OY, OW, OH = 0, HH, HW, HH
    panel(surf, OX, OY, OW, OH, "Top-down View   (y -> right,  x -> down)")
    cx = OX + OW // 2
    cy = OY + OH // 2 - 40
    pygame.draw.line(surf, D_GRAY, (OX+20, cy), (OX+OW-20, cy), 1)
    txt(surf, "y", Fsm, D_GRAY, OX+OW-16, cy-14)
    pygame.draw.line(surf, D_GRAY, (cx, OY+30), (cx, OY+OH-80), 1)
    txt(surf, "x", Fsm, D_GRAY, cx+6, OY+OH-82)
    r_p_px = int(R_P * SCALE)
    r_s_px = max(4, int(R_S * SCALE))
    pygame.draw.circle(surf, (50, 50, 150), (cx, cy), r_p_px, 2)
    pygame.draw.circle(surf, (150, 80, 30), (cx, cy), r_s_px, 2)
    pygame.draw.line(surf, BARY_COL, (cx-6, cy), (cx+6, cy), 2)
    pygame.draw.line(surf, BARY_COL, (cx, cy-6), (cx, cy+6), 2)
    xp = R_P * np.cos(theta)
    yp = R_P * np.sin(theta)
    xs = -R_S * np.cos(theta)
    ys = -R_S * np.sin(theta)
    p_tx = cx + int(yp * SCALE)
    p_ty = cy + int(xp * SCALE)
    s_tx = cx + int(ys * SCALE)
    s_ty = cy + int(xs * SCALE)
    pcol = planet_colour(DENSITY_RATIO)
    sphere(surf, s_tx, s_ty, R_STAR_PX, STAR_COL)
    sphere(surf, p_tx, p_ty, R_PLAN_PX, pcol)
    txt(surf, "Star",   Fsm, STAR_COL, s_tx + R_STAR_PX + 4, s_ty - 7)
    txt(surf, "Planet", Fsm, pcol,     p_tx + R_PLAN_PX + 4, p_ty - 7)
    for yi in range(cy, OY+OH-80, 14):
        pygame.draw.line(surf, (40, 40, 70), (cx, yi), (cx, yi+8), 1)
    txt(surf, "v Observer (+x)", Fsm, L_GRAY, cx, OY+OH-78, anchor="midtop")
    txt(surf, "-- planet orbit", Fsm, (100, 100, 200), OX+10, OY+OH-95)
    txt(surf, "-- star orbit",   Fsm, (180, 110,  60), OX+10, OY+OH-80)
    draw_density_slider(surf)

def draw_light_curve(surf, flux, lc_h, p):
    OX, OY, OW, OH = HW, 0, HW, HH
    panel(surf, OX, OY, OW, OH, "Transit Light Curve")
    ml, mr, mt, mb = 58, 14, 30, 38
    px0 = OX + ml;  py0 = OY + mt
    pw  = OW - ml - mr
    ph  = OH - mt - mb
    pygame.draw.rect(surf, PANEL_DARK, (px0, py0, pw, ph))
    grid(surf, px0, py0, pw, ph, nx=6, ny=5)
    pygame.draw.rect(surf, BORDER, (px0, py0, pw, ph), 1)
    def fy(f):
        return py0 + int((1 - (f - F_LO) / (F_HI - F_LO)) * ph)
    ref_y = fy(1.0)
    pygame.draw.line(surf, GRID_COL, (px0, ref_y), (px0+pw, ref_y), 1)
    n_ticks = 6
    for i in range(n_ticks + 1):
        fv = F_LO + i * (F_HI - F_LO) / n_ticks
        yp = fy(fv)
        if py0 <= yp <= py0+ph:
            pygame.draw.line(surf, BORDER, (px0-4, yp), (px0, yp))
            txt(surf, f"{fv:.4f}", Fsm, L_GRAY, px0-6, yp, anchor="midright")
    txt(surf, "Time  (one orbit ->)", Fsm, GRAY, px0 + pw//2, py0+ph+6, anchor="midtop")
    yl = Fsm.render("Relative Flux", True, GRAY)
    yl = pygame.transform.rotate(yl, 90)
    surf.blit(yl, (OX+4, py0 + ph//2 - yl.get_height()//2))
    order = [(p + i) % HIST for i in range(HIST)]
    pts   = []
    for xi, idx in enumerate(order):
        xp_ = px0 + int(xi / (HIST-1) * pw)
        yp_ = fy(np.clip(lc_h[idx], F_LO, F_HI))
        pts.append((xp_, yp_))
    if len(pts) >= 2:
        pygame.draw.lines(surf, LC_LINE, False, pts, 2)
    pygame.draw.circle(surf, WHITE, (px0+pw, fy(flux)), 5)
    depth = (1.0 - flux) * 100
    if depth > 0.05:
        txt(surf, f"Transit depth  D = {depth:.2f}%", Fmed, TRANSIT_COL, px0+6, py0+6)
    else:
        txt(surf, "Out of transit", Fsm, GRAY, px0+6, py0+6)

_SW = HW - 22 - 22
_SH = 22
SPEC_BG = make_spec_bg(_SW, _SH)

def draw_spectrum(surf, doppler_norm, rv_h, p):
    OX, OY, OW, OH = HW, HH, HW, HH
    panel(surf, OX, OY, OW, OH, "Doppler Spectroscopy  (H Balmer absorption)")
    ml, mr, mt = 52, 22, 52
    sx0 = OX + ml;  sy0 = OY + mt
    sw  = OW - ml - mr
    surf.blit(SPEC_BG, (sx0, sy0))
    pygame.draw.rect(surf, BORDER, (sx0, sy0, sw, _SH), 1)
    shift       = -doppler_norm * DOPPLER_SCALE * (R_S / R_P)
    shifted_col = BLUE_COL if doppler_norm > 0.03 else \
                  (RED_COL if doppler_norm < -0.03 else L_GRAY)
    for name, wl0 in H_LINES:
        xr = wl_to_x(wl0, sx0, sw)
        if sx0 <= xr <= sx0+sw:
            for dx in [-1, 0, 1]:
                pygame.draw.line(surf, (30, 30, 30), (xr+dx, sy0), (xr+dx, sy0+_SH))
            pygame.draw.line(surf, (130, 130, 130), (xr, sy0), (xr, sy0+_SH), 1)
        wl_s = wl0 + shift
        if WL_MIN <= wl_s <= WL_MAX:
            xs = wl_to_x(wl_s, sx0, sw)
            for dx in [-1, 0, 1]:
                pygame.draw.line(surf, (5, 5, 5), (xs+dx, sy0), (xs+dx, sy0+_SH))
            pygame.draw.line(surf, shifted_col, (xs, sy0), (xs, sy0+_SH), 2)
    for name, wl0 in H_LINES:
        wl_s = wl0 + shift
        xs = wl_to_x(np.clip(wl_s, WL_MIN, WL_MAX), sx0, sw)
        txt(surf, name, Fsm, shifted_col, xs, sy0+_SH+4, anchor="midtop")
    tick_y = sy0 + _SH + 2
    for wl_t in range(400, 710, 50):
        xt = wl_to_x(wl_t, sx0, sw)
        pygame.draw.line(surf, BORDER, (xt, sy0+_SH), (xt, sy0+_SH+5))
        txt(surf, str(wl_t), Fsm, GRAY, xt, tick_y, anchor="midtop")
    txt(surf, "Wavelength (nm)", Fsm, GRAY, sx0+sw//2, tick_y+14, anchor="midtop")
    if abs(doppler_norm) < 0.04:
        ds = "Dv ~ 0   (zero crossing)"
        dc = L_GRAY
    elif doppler_norm > 0:
        ds = f"<< Blueshift   Dl = -{abs(shift):.1f} nm  (approaching)"
        dc = BLUE_COL
    else:
        ds = f">> Redshift    Dl = +{abs(shift):.1f} nm  (receding)"
        dc = RED_COL
    txt(surf, ds, Fmed, dc, sx0 + sw//2, sy0 + _SH + 48, anchor="midbottom")
    txt(surf, "--- rest wavelength", Fsm, (150,150,150), sx0,     sy0-16)
    txt(surf, "--- doppler shifted", Fsm, shifted_col,   sx0+160, sy0-16)
    rv_y0 = sy0 + _SH + 58
    rv_h_ = OH - (rv_y0 - OY) - 28
    rv_x0 = sx0
    rv_w  = sw
    pygame.draw.rect(surf, PANEL_DARK, (rv_x0, rv_y0, rv_w, rv_h_))
    grid(surf, rv_x0, rv_y0, rv_w, rv_h_, nx=6, ny=4)
    pygame.draw.rect(surf, BORDER, (rv_x0, rv_y0, rv_w, rv_h_), 1)
    mid_y = rv_y0 + rv_h_//2
    pygame.draw.line(surf, GRAY, (rv_x0, mid_y), (rv_x0+rv_w, mid_y), 1)
    txt(surf, "Radial Velocity", Fsm, GRAY,     rv_x0+4, rv_y0+3)
    txt(surf, "Blue^",          Fsm, BLUE_COL,  rv_x0+4, rv_y0+4)
    txt(surf, "Redv",           Fsm, RED_COL,   rv_x0+4, rv_y0+rv_h_-16)
    txt(surf, "0",              Fsm, GRAY,       rv_x0-4, mid_y, anchor="midright")
    max_shift = DOPPLER_SCALE * RV_YLIM
    txt(surf, f"+{max_shift:.1f}nm", Fsm, BLUE_COL, rv_x0-4, rv_y0+4,       anchor="midright")
    txt(surf, f"-{max_shift:.1f}nm", Fsm, RED_COL,  rv_x0-4, rv_y0+rv_h_-4, anchor="midright")
    order = [(p + i) % HIST for i in range(HIST)]
    prev  = None
    for xi, idx in enumerate(order):
        xp_ = rv_x0 + int(xi / (HIST-1) * rv_w)
        yp_ = mid_y  - int((rv_h[idx] / RV_YLIM) * (rv_h_//2 - 4))
        pt  = (xp_, yp_)
        if prev is not None:
            col = BLUE_COL if rv_h[idx] > 0.001 else \
                  (RED_COL if rv_h[idx] < -0.001 else L_GRAY)
            pygame.draw.line(surf, col, prev, pt, 2)
        prev = pt
    cur_y = mid_y - int((doppler_norm / RV_YLIM) * (rv_h_//2 - 4))
    pygame.draw.circle(surf, WHITE, (rv_x0+rv_w, cur_y), 5)

t0       = pygame.time.get_ticks() / 1000.0
paused   = False
t_frozen = 0.0

while True:
    clock.tick(60)

    for ev in pygame.event.get():
        if ev.type == pygame.QUIT:
            pygame.quit(); sys.exit()
        if ev.type == pygame.KEYDOWN:
            if ev.key == pygame.K_ESCAPE:
                pygame.quit(); sys.exit()
            if ev.key == pygame.K_SPACE:
                paused = not paused
                if paused:
                    t_frozen = pygame.time.get_ticks() / 1000.0 - t0
                else:
                    t0 = pygame.time.get_ticks() / 1000.0 - t_frozen
            if ev.key == pygame.K_UP:
                R_PLAN_PX = min(R_PLAN_MAX, R_PLAN_PX + 1)
            if ev.key == pygame.K_DOWN:
                R_PLAN_PX = max(R_PLAN_MIN, R_PLAN_PX - 1)
        if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
            mx, my = ev.pos
            if SL_X - 12 <= mx <= SL_X + SL_W + 12 and SL_Y - 16 <= my <= SL_Y + 16:
                SL_DRAG = True
        if ev.type == pygame.MOUSEBUTTONUP and ev.button == 1:
            SL_DRAG = False
        if ev.type == pygame.MOUSEMOTION and SL_DRAG:
            mx = ev.pos[0]
            t  = (mx - SL_X) / SL_W
            DENSITY_RATIO = round(max(DENS_MIN, min(DENS_MAX,
                             DENS_MIN + t * (DENS_MAX - DENS_MIN))), 2)

    t = t_frozen if paused else pygame.time.get_ticks() / 1000.0 - t0
    theta = OMEGA * t + np.pi

    R_S = R_P * DENSITY_RATIO * (R_PLAN_PX / R_STAR_PX) ** 3

    xp = R_P * np.cos(theta)
    yp = R_P * np.sin(theta)
    xs = -R_S * np.cos(theta)
    ys = -R_S * np.sin(theta)

    doppler_norm = np.sin(theta)

    trans_sep  = abs(yp - ys) * SCALE
    overlap    = trans_sep < (R_STAR_PX + R_PLAN_PX)
    in_transit = overlap and (xp > xs)
    in_occ     = overlap and (xp < xs)

    if in_transit:
        depth_full = (R_PLAN_PX / R_STAR_PX) ** 2
        ovlp = (R_STAR_PX + R_PLAN_PX) - trans_sep
        frac = np.clip(ovlp / (2 * R_PLAN_PX), 0, 1)
        flux = 1.0 - depth_full * frac
    else:
        flux = 1.0

    if not paused:
        lc_hist[ptr] = flux
        rv_hist[ptr] = doppler_norm * (R_S / R_P)
        ptr = (ptr + 1) % HIST

    screen.fill(BG)
    draw_edge_on    (screen, theta, in_transit, in_occ)
    draw_top_down   (screen, theta)
    draw_light_curve(screen, flux, lc_hist, ptr)
    draw_spectrum   (screen, doppler_norm * (R_S / R_P), rv_hist, ptr)
    pygame.draw.line(screen, BORDER, (HW, 0),  (HW, H),  2)
    pygame.draw.line(screen, BORDER, (0,  HH), (W,  HH), 2)

    ph_deg    = np.degrees(theta % (2*np.pi))
    depth_pct = (R_PLAN_PX / R_STAR_PX) ** 2 * 100
    status = (f"  th={ph_deg:5.1f}  flux={flux:.4f}  "
              f"R_p={R_PLAN_PX}px  rho={DENSITY_RATIO:.2f}x  "
              f"depth={depth_pct:.2f}%  "
              f"[UP/DOWN] radius  [drag] density  [SPACE] pause  [ESC] quit")
    txt(screen, status, Fsm, GRAY, 4, H-15)
    pygame.display.flip()
