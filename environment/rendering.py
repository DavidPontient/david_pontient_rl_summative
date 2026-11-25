"""2D pygame view of the construction site.

Left side: the plot, the house growing phase by phase, material piles, workers,
delivery truck and the architect. Right side: a dashboard with the numbers the
agent sees (budget, prices, stock, labour owed, architect request).
"""
from __future__ import annotations

import math
import os

import numpy as np
import pygame

from environment.custom_env import (
    ACTION_NAMES, MATERIALS, PHASES, PHASE_BUDGET, PHASE_BUILDS, TOTAL_BUDGET, STOCK_CAP,
)

W, H = 1180, 700
SCENE_W = 780
GROUND_Y = 470

C_BG = (20, 28, 44)
C_PANEL = (30, 41, 64)
C_TEXT = (226, 232, 240)
C_MUTED = (148, 163, 184)
C_GOOD = (74, 222, 128)
C_BAD = (248, 113, 113)
C_WARN = (250, 204, 21)
MAT_COLORS = [(156, 163, 175), (96, 165, 250), (217, 181, 124), (180, 110, 60)]


def _lerp(a, b, t):
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


class ConstructionRenderer:
    def __init__(self, env, show_window: bool = True):
        self.env = env
        self.show_window = show_window
        if not show_window:
            os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        pygame.display.init()
        pygame.font.init()
        if show_window:
            self.screen = pygame.display.set_mode((W, H))
            pygame.display.set_caption("Kigali Home Build, RL agent")
        else:
            self.screen = pygame.Surface((W, H))
        self.clock = pygame.time.Clock()
        self.f_big = pygame.font.SysFont("arial,dejavusans,segoeui", 26, bold=True)
        self.f_mid = pygame.font.SysFont("arial,dejavusans,segoeui", 18, bold=True)
        self.f_sm = pygame.font.SysFont("arial,dejavusans,segoeui", 15)
        self.f_xs = pygame.font.SysFont("arial,dejavusans,segoeui", 13)
        self.log = []
        self._last_logged = None
        self.background = self._make_background()

    # ------------------------------------------------------------------
    def _make_background(self):
        s = pygame.Surface((SCENE_W, H))
        top, bottom = (110, 170, 230), (214, 234, 248)
        for y in range(0, GROUND_Y):
            pygame.draw.line(s, _lerp(top, bottom, y / GROUND_Y), (0, y), (SCENE_W, y))
        pygame.draw.circle(s, (255, 236, 170), (680, 90), 38)
        # far hills, Kigali style
        pygame.draw.polygon(s, (120, 170, 140), [(0, GROUND_Y), (0, 360), (140, 320), (300, 380), (460, 330), (620, 390), (780, 340), (780, GROUND_Y)])
        pygame.draw.polygon(s, (92, 150, 112), [(0, GROUND_Y), (0, 420), (200, 390), (380, 430), (560, 400), (780, 430), (780, GROUND_Y)])
        pygame.draw.rect(s, (110, 160, 80), (0, GROUND_Y, SCENE_W, H - GROUND_Y))
        pygame.draw.rect(s, (176, 140, 98), (0, GROUND_Y + 70, SCENE_W, 70))        # dirt road
        for x in range(0, SCENE_W, 60):
            pygame.draw.rect(s, (200, 170, 130), (x + 10, GROUND_Y + 102, 30, 4))
        # fence
        for x in range(10, SCENE_W, 34):
            pygame.draw.rect(s, (120, 90, 60), (x, GROUND_Y + 8, 5, 32))
        pygame.draw.rect(s, (140, 105, 70), (10, GROUND_Y + 16, SCENE_W - 20, 4))
        pygame.draw.rect(s, (140, 105, 70), (10, GROUND_Y + 28, SCENE_W - 20, 4))
        return s

    # ------------------------------------------------------------------
    def _text(self, txt, pos, font=None, color=C_TEXT, center=False, right=False):
        font = font or self.f_sm
        surf = font.render(str(txt), True, color)
        r = surf.get_rect()
        if center:
            r.center = pos
        elif right:
            r.topright = pos
        else:
            r.topleft = pos
        self.screen.blit(surf, r)
        return r

    def _bar(self, x, y, w, h, frac, color, back=(51, 65, 85)):
        pygame.draw.rect(self.screen, back, (x, y, w, h), border_radius=4)
        frac = float(np.clip(frac, 0.0, 1.0))
        if frac > 0:
            pygame.draw.rect(self.screen, color, (x, y, max(int(w * frac), 3), h), border_radius=4)

    # ------------------------------------------------------------------
    def _draw_house(self, t):
        e = self.env
        x0, x1 = 300, 640
        base = GROUND_Y
        ph, pb = e.phase, e.phase_builds
        frac = [1.0 if ph > i else (pb / PHASE_BUILDS[i] if ph == i else 0.0) for i in range(5)]

        # preliminary: pegs and string line
        pf = frac[0]
        pygame.draw.rect(self.screen, (150, 115, 80), (x0 - 20, base - 4, (x1 - x0 + 40), 12))
        pegs = int(2 + 8 * pf)
        for i in range(pegs):
            px = x0 - 20 + i * (x1 - x0 + 40) / 9
            pygame.draw.rect(self.screen, (240, 200, 60), (px, base - 14, 4, 16))
        if pf > 0.4:
            pygame.draw.line(self.screen, (240, 240, 240), (x0 - 20, base - 12), (min(x1 + 20, x0 - 20 + (x1 - x0 + 40) * pf), base - 12), 2)

        # foundation
        ff = frac[1]
        if ff > 0:
            fw = int((x1 - x0 + 20) * ff)
            pygame.draw.rect(self.screen, (105, 105, 112), (x0 - 10, base - 22, fw, 26))
            for i in range(0, fw, 22):
                pygame.draw.rect(self.screen, (82, 82, 90), (x0 - 10 + i, base - 22, 20, 12), 1)

        # elevation: walls for two floors with columns and a slab line
        ef = frac[2]
        wall_h = 230
        h = int(wall_h * ef)
        wall_col = (190, 182, 170)
        if h > 0:
            pygame.draw.rect(self.screen, wall_col, (x0, base - 22 - h, x1 - x0, h))
            for yy in range(0, h, 14):
                pygame.draw.line(self.screen, (160, 152, 140), (x0, base - 22 - yy), (x1, base - 22 - yy), 1)
            for cx in (x0, (x0 + x1) // 2, x1 - 10):
                pygame.draw.rect(self.screen, (120, 118, 120), (cx, base - 22 - h, 10, h))
            if h > wall_h // 2:
                pygame.draw.rect(self.screen, (90, 90, 96), (x0 - 6, base - 22 - wall_h // 2 - 8, x1 - x0 + 12, 8))
            if ef < 1.0:  # rebar sticking out of the top
                for cx in (x0 + 3, (x0 + x1) // 2 + 3, x1 - 7):
                    pygame.draw.line(self.screen, (150, 90, 60), (cx, base - 22 - h), (cx, base - 22 - h - 22), 2)

        # roofing
        rf = frac[3]
        top_y = base - 22 - wall_h
        if rf > 0:
            half = int((x1 - x0 + 40) * rf)
            pts = [(x0 - 20, top_y), (x0 + 40, top_y - 62), (x1 - 40, top_y - 62), (x1 + 20, top_y)]
            roof = pygame.Surface((SCENE_W, H), pygame.SRCALPHA)
            pygame.draw.polygon(roof, (150, 70, 60, 255), pts)
            mask = pygame.Surface((SCENE_W, H), pygame.SRCALPHA)
            pygame.draw.rect(mask, (255, 255, 255, 255), (0, 0, x0 - 20 + half, H))
            roof.blit(mask, (0, 0), special_flags=pygame.BLEND_RGBA_MIN)
            self.screen.blit(roof, (0, 0))
            for i in range(1, 7):
                lx = x0 - 20 + i * (x1 - x0 + 40) / 7
                if lx < x0 - 20 + half:
                    pygame.draw.line(self.screen, (120, 50, 45), (lx, top_y), (lx, top_y - 60), 1)

        # finishes
        nf = frac[4]
        if nf > 0 and h > 0:
            paint = _lerp(wall_col, (244, 226, 190), min(nf * 1.4, 1.0))
            pygame.draw.rect(self.screen, paint, (x0, base - 22 - wall_h, x1 - x0, wall_h))
            if nf > 0.3:
                for wx in (x0 + 40, x0 + 140, x0 + 240):
                    for wy in (base - 22 - 90, base - 22 - 200):
                        if wx == x0 + 140 and wy == base - 22 - 90:
                            continue
                        pygame.draw.rect(self.screen, (60, 90, 130), (wx, wy, 44, 52))
                        pygame.draw.rect(self.screen, (230, 230, 235), (wx, wy, 44, 52), 3)
                        pygame.draw.line(self.screen, (230, 230, 235), (wx + 22, wy), (wx + 22, wy + 52), 2)
            if nf > 0.55:
                pygame.draw.rect(self.screen, (110, 70, 40), ((x0 + x1) // 2 - 22, base - 22 - 88, 44, 88))
                pygame.draw.circle(self.screen, (230, 200, 80), ((x0 + x1) // 2 + 14, base - 22 - 44), 3)
            if nf > 0.8:
                pygame.draw.rect(self.screen, (170, 150, 130), (x0 - 12, base - 24, x1 - x0 + 24, 6))
            if nf > 0.95:
                pygame.draw.circle(self.screen, (255, 240, 150), ((x0 + x1) // 2, base - 22 - 104), 6)

        if ph >= 5:
            self._text("HOUSE COMPLETE", (x0 + 170, top_y - 95), self.f_big, C_GOOD, center=True)

    def _draw_workers(self, t):
        e = self.env
        working = e.last_event == "build"
        for i, wx in enumerate((330, 400, 470)):
            bob = math.sin((t * 6.0 + i) * math.pi) * 4 if working else 0
            y = GROUND_Y + 30 - bob
            pygame.draw.circle(self.screen, (232, 190, 150), (wx, y - 40), 8)
            pygame.draw.rect(self.screen, (250, 204, 21), (wx - 9, y - 52, 18, 7), border_radius=3)     # helmet
            pygame.draw.rect(self.screen, (234, 88, 12), (wx - 8, y - 32, 16, 26), border_radius=3)     # vest
            pygame.draw.line(self.screen, (50, 60, 90), (wx - 4, y - 6), (wx - 4, y + 12), 4)
            pygame.draw.line(self.screen, (50, 60, 90), (wx + 4, y - 6), (wx + 4, y + 12), 4)
            if working:
                ang = math.sin((t * 8.0 + i) * math.pi) * 0.7
                pygame.draw.line(self.screen, (120, 120, 130), (wx + 8, y - 24), (wx + 8 + 16 * math.cos(ang), y - 24 - 16 * math.sin(ang)), 3)

    def _draw_stock(self):
        e = self.env
        # cement bags
        n = int(round(e.stock[0] * 4))
        for i in range(n):
            bx, by = 30 + (i % 4) * 30, GROUND_Y + 30 - (i // 4) * 14
            pygame.draw.rect(self.screen, MAT_COLORS[0], (bx, by, 28, 13), border_radius=3)
            pygame.draw.rect(self.screen, (110, 116, 126), (bx, by, 28, 13), 1, border_radius=3)
        # steel rebar rack
        n = int(round(e.stock[1] * 5))
        for i in range(n):
            pygame.draw.line(self.screen, MAT_COLORS[1], (160, GROUND_Y + 36 - i * 4), (270, GROUND_Y + 36 - i * 4), 3)
        # sand and stones
        s = e.stock[2] / STOCK_CAP
        if s > 0.02:
            pw = int(40 + 70 * s)
            ph = int(10 + 50 * s)
            pygame.draw.polygon(self.screen, MAT_COLORS[2], [(30, GROUND_Y + 110), (30 + pw, GROUND_Y + 110), (30 + pw // 2, GROUND_Y + 110 - ph)])
        # timber stack
        n = int(round(e.stock[3] * 5))
        for i in range(n):
            pygame.draw.rect(self.screen, MAT_COLORS[3], (160, GROUND_Y + 110 - i * 7, 110, 6))
            pygame.draw.rect(self.screen, (130, 80, 40), (160, GROUND_Y + 110 - i * 7, 110, 6), 1)
        self._text("cement", (30, GROUND_Y + 36 + 2), self.f_xs, C_TEXT)
        self._text("steel", (160, GROUND_Y + 42), self.f_xs, C_TEXT)
        self._text("sand + stone", (30, GROUND_Y + 114), self.f_xs, C_TEXT)
        self._text("timber", (160, GROUND_Y + 114), self.f_xs, C_TEXT)

    def _draw_truck(self, t):
        e = self.env
        if not e.last_event.startswith("buy:"):
            return
        m = MATERIALS.index(e.last_event.split(":")[1])
        k = min(t * 1.4, 1.0)
        x = 840 - (840 - 120) * (1 - (1 - k) ** 2)
        y = GROUND_Y + 126
        pygame.draw.rect(self.screen, (60, 80, 140), (x, y - 40, 100, 40), border_radius=4)
        pygame.draw.rect(self.screen, (60, 80, 140), (x - 34, y - 28, 34, 28), border_radius=4)
        pygame.draw.rect(self.screen, (200, 225, 245), (x - 28, y - 24, 18, 12))
        pygame.draw.rect(self.screen, MAT_COLORS[m], (x + 8, y - 54, 84, 16), border_radius=3)
        for wx in (x - 18, x + 20, x + 76):
            pygame.draw.circle(self.screen, (25, 25, 30), (wx, y + 2), 10)
            pygame.draw.circle(self.screen, (120, 120, 130), (wx, y + 2), 4)

    def _draw_architect(self):
        e = self.env
        ax, ay = 700, GROUND_Y + 30
        pygame.draw.circle(self.screen, (232, 190, 150), (ax, ay - 52), 10)
        pygame.draw.rect(self.screen, (40, 60, 100), (ax - 11, ay - 40, 22, 36), border_radius=4)
        pygame.draw.line(self.screen, (30, 30, 40), (ax - 5, ay - 4), (ax - 5, ay + 14), 4)
        pygame.draw.line(self.screen, (30, 30, 40), (ax + 5, ay - 4), (ax + 5, ay + 14), 4)
        pygame.draw.rect(self.screen, (120, 90, 60), (ax + 12, ay - 26, 18, 14), border_radius=2)
        self._text("architect", (ax, ay + 20), self.f_xs, C_MUTED, center=True)
        if e.req_pending:
            col = C_GOOD if e.req_verified else C_WARN
            r = pygame.Rect(ax - 105, ay - 130, 150, 54)
            pygame.draw.rect(self.screen, (255, 255, 255), r, border_radius=8)
            pygame.draw.rect(self.screen, col, r, 3, border_radius=8)
            pygame.draw.polygon(self.screen, (255, 255, 255), [(ax - 14, ay - 76), (ax + 2, ay - 76), (ax - 2, ay - 64)])
            label = "Verified" if e.req_verified else "Payment request"
            self._text(label, (r.x + 10, r.y + 6), self.f_xs, (40, 40, 40))
            self._text(f"{e.req_amount / 1e6:.2f}M RWF", (r.x + 10, r.y + 24), self.f_mid, (20, 20, 20))

    # ------------------------------------------------------------------
    def _draw_panel(self):
        e = self.env
        x = SCENE_W
        pygame.draw.rect(self.screen, C_PANEL, (x, 0, W - SCENE_W, H))
        px = x + 18
        self._text("Kigali Home Build", (px, 18), self.f_mid)
        self._text(f"Week {e.week} of {e.max_weeks}", (W - 18, 20), self.f_mid, C_MUTED, right=True)

        # phase
        phase_name = PHASES[e.phase] if e.phase < 5 else "Complete"
        self._text(f"Phase: {phase_name}", (px, 56), self.f_mid)
        self._bar(px, 82, 360, 12, e.progress(), C_GOOD)
        self._text(f"overall progress {e.progress() * 100:.0f}%", (px, 98), self.f_xs, C_MUTED)

        # budget
        self._text("Budget left", (px, 126), self.f_mid)
        bfrac = e.budget / TOTAL_BUDGET
        self._bar(px, 152, 360, 14, bfrac, C_GOOD if bfrac > 0.3 else (C_WARN if bfrac > 0.12 else C_BAD))
        self._text(f"{e.budget / 1e6:.1f}M of {TOTAL_BUDGET / 1e6:.1f}M RWF", (px, 170), self.f_xs, C_MUTED)

        # prices and stock
        self._text("Material", (px, 198), self.f_xs, C_MUTED)
        self._text("price vs base", (px + 100, 198), self.f_xs, C_MUTED)
        self._text("stock (lots)", (px + 215, 198), self.f_xs, C_MUTED)
        ratio = e._ratio()
        mom = ratio - e.price_hist[0]
        for i, name in enumerate(MATERIALS):
            y = 220 + i * 30
            pygame.draw.rect(self.screen, MAT_COLORS[i], (px, y + 3, 10, 10))
            self._text(name, (px + 16, y), self.f_sm)
            col = C_BAD if ratio[i] > 1.08 else (C_GOOD if ratio[i] < 0.95 else C_TEXT)
            arrow = "up" if mom[i] > 0.01 else ("down" if mom[i] < -0.01 else "flat")
            self._text(f"{ratio[i]:.2f}x {arrow}", (px + 100, y), self.f_sm, col)
            self._bar(px + 215, y + 3, 100, 10, e.stock[i] / STOCK_CAP, MAT_COLORS[i])
            self._text(f"{e.stock[i]:.1f}", (px + 322, y), self.f_sm)

        # labour, request, quote
        y = 350
        wage = e._wage()
        self._text("Labour owed", (px, y), self.f_mid)
        self._bar(px, y + 26, 360, 12, (e.labour_owed / (2 * wage)) if wage > 0 else 0, C_WARN)
        self._text(f"{e.labour_owed / 1e6:.2f}M RWF", (px, y + 42), self.f_xs, C_MUTED)
        if e.discount > 0:
            self._text(f"Supplier quote ready: {e.discount * 100:.1f}% off next buy", (px, y + 64), self.f_sm, C_GOOD)
        else:
            self._text("No supplier quote in hand", (px, y + 64), self.f_sm, C_MUTED)
        if e.phase >= 5:
            self._text("Build finished", (px, y + 86), self.f_sm, C_GOOD)
        elif e._hold():
            self._text("Architect has put the site on hold", (px, y + 86), self.f_sm, C_BAD)
        elif e._can_build():
            self._text("Crew is ready to build", (px, y + 86), self.f_sm, C_GOOD)
        else:
            self._text("Crew cannot build right now", (px, y + 86), self.f_sm, C_WARN)

        # last action
        y = 472
        self._text("Last action", (px, y), self.f_mid)
        name = ACTION_NAMES[e.last_action] if e.last_action is not None else "none yet"
        col = C_BAD if e.last_event == "invalid" else C_TEXT
        self._text(name + ("  (not allowed)" if e.last_event == "invalid" else ""), (px, y + 26), self.f_sm, col)
        self._text(f"reward {e.last_reward:+.2f}", (px + 200, y + 26), self.f_sm, C_GOOD if e.last_reward >= 0 else C_BAD)
        self._text(f"episode return {e.total_reward:+.1f}", (px, y + 48), self.f_sm, C_MUTED)

        # log
        y = 552
        self._text("Recent steps", (px, y), self.f_mid)
        for i, line in enumerate(self.log[-4:]):
            self._text(line, (px, y + 26 + i * 18), self.f_xs, C_MUTED)

    # ------------------------------------------------------------------
    def _update_log(self):
        e = self.env
        key = (e.week, e.last_action)
        if e.last_action is not None and key != self._last_logged:
            self._last_logged = key
            self.log.append(f"wk {e.week:>3}  {ACTION_NAMES[e.last_action]:<18} {e.last_reward:+.2f}")
        if e.week == 0:
            self.log = []

    def _draw_frame(self, t):
        self.screen.fill(C_BG)
        self.screen.blit(self.background, (0, 0))
        self._draw_stock()
        self._draw_house(t)
        self._draw_workers(t)
        self._draw_architect()
        self._draw_truck(t)
        self._draw_panel()

    def render(self, animate: bool = True):
        self._update_log()
        if self.show_window:
            frames = 14 if animate else 1
            for f in range(frames):
                for ev in pygame.event.get():
                    if ev.type == pygame.QUIT:
                        self.close()
                        raise SystemExit
                self._draw_frame(f / max(frames - 1, 1))
                pygame.display.flip()
                self.clock.tick(30)
            return None
        self._draw_frame(1.0)
        return np.transpose(np.array(pygame.surfarray.pixels3d(self.screen)), (1, 0, 2)).copy()

    def save_frame(self, path: str):
        self._draw_frame(1.0)
        pygame.image.save(self.screen, path)

    def close(self):
        if pygame.get_init():
            pygame.display.quit()
            pygame.quit()
