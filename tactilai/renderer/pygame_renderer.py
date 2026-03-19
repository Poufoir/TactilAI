"""
tactilai/renderer/pygame_renderer.py

Pygame renderer for TactilAI — 32x32 sprites on a 16x16 grid.

Layout
──────
  ┌─────────────────────────────────┬──────────────┐
  │  Grid (512x512)                 │  Side panel  │
  │  16x16 tiles × 32px             │  (256px)     │
  │                                 │              │
  │  Terrain colors                 │  Turn info   │
  │  Unit sprites (shape + letter)  │  Unit stats  │
  │  HP bars                        │  Controls    │
  └─────────────────────────────────┴──────────────┘
  Total window : 768 × 560

Terrain colors
──────────────
  Plain    : #8DB360  (green)
  Forest   : #2D6A4F  (dark green)
  Mountain : #6B6570  (grey-purple)

Unit colors (by team)
─────────────────────
  Blue team : #4A90D9
  Red team  : #E05C5C

Unit shapes (by class)
──────────────────────
  Knight   : square
  Archer   : triangle (up)
  Mage     : diamond
  Healer   : cross (+)
  Cavalier : circle
  Barbarian: hexagon

Controls
────────
  SPACE     : toggle step/realtime mode
  RIGHT / N : next step (step mode only)
  Q / ESC   : quit
  +/-       : speed up / slow down (realtime mode)
"""

from __future__ import annotations

import math

import pygame

from tactilai.env.grid import Grid
from tactilai.env.terrain import TerrainType
from tactilai.env.unit import Team, Unit, UnitClass

# ── Layout constants ──────────────────────────────────────────────────────────

TILE_SIZE = 32
GRID_PX = TILE_SIZE * 16  # 512
PANEL_WIDTH = 256
WIN_WIDTH = GRID_PX + PANEL_WIDTH  # 768
WIN_HEIGHT = GRID_PX + 48  # 560  (grid + bottom bar)
FPS_DEFAULT = 10

# ── Colors ────────────────────────────────────────────────────────────────────

C_BG = (18, 18, 24)
C_PLAIN = (141, 179, 96)
C_FOREST = (45, 106, 79)
C_MOUNTAIN = (107, 101, 112)
C_GRID_LINE = (0, 0, 0, 60)  # semi-transparent overlay (not used directly)
C_BLUE_LIGHT = (140, 190, 255)  # blue ready
C_BLUE_DARK = (40, 80, 150)  # blue has acted
C_RED_LIGHT = (255, 140, 140)  # red ready (pink)
C_RED_DARK = (150, 30, 30)  # red has acted
C_HP_GREEN = (80, 200, 80)
C_HP_YELLOW = (220, 200, 40)
C_HP_RED = (220, 60, 60)
C_HP_BG = (40, 40, 40)
C_PANEL_BG = (28, 28, 36)
C_TEXT = (220, 220, 220)
C_TEXT_DIM = (140, 140, 140)
C_HIGHLIGHT = (255, 220, 80)
C_BLUE = C_BLUE_LIGHT  # alias pour le panel
C_RED = C_RED_LIGHT  # alias pour le panel

TERRAIN_COLORS: dict[TerrainType, tuple[int, int, int]] = {
    TerrainType.PLAIN: C_PLAIN,
    TerrainType.FOREST: C_FOREST,
    TerrainType.MOUNTAIN: C_MOUNTAIN,
}


# ── Sprite drawing ────────────────────────────────────────────────────────────


def _unit_color(unit: Unit) -> tuple[int, int, int]:
    if unit.team == Team.BLUE:
        return C_BLUE_DARK if unit.has_acted else C_BLUE_LIGHT
    else:
        return C_RED_DARK if unit.has_acted else C_RED_LIGHT


def _draw_sprite(
    surface: pygame.Surface,
    unit: Unit,
    cx: int,
    cy: int,
    size: int = 20,
) -> None:
    """
    Draws a class-specific shape centered at (cx, cy).
    Size is the bounding box half-width.
    """
    color = _unit_color(unit)
    s = size

    match unit.unit_class:
        case UnitClass.KNIGHT:
            # Filled square
            rect = pygame.Rect(cx - s // 2, cy - s // 2, s, s)
            pygame.draw.rect(surface, color, rect)
            pygame.draw.rect(surface, C_BG, rect, 1)

        case UnitClass.ARCHER:
            # Upward triangle
            pts = [
                (cx, cy - s // 2),
                (cx - s // 2, cy + s // 2),
                (cx + s // 2, cy + s // 2),
            ]
            pygame.draw.polygon(surface, color, pts)
            pygame.draw.polygon(surface, C_BG, pts, 1)

        case UnitClass.MAGE:
            # Diamond
            pts = [
                (cx, cy - s // 2),
                (cx + s // 2, cy),
                (cx, cy + s // 2),
                (cx - s // 2, cy),
            ]
            pygame.draw.polygon(surface, color, pts)
            pygame.draw.polygon(surface, C_BG, pts, 1)

        case UnitClass.HEALER:
            # Cross (+)
            t = s // 4
            rects = [
                pygame.Rect(cx - t, cy - s // 2, t * 2, s),
                pygame.Rect(cx - s // 2, cy - t, s, t * 2),
            ]
            for r in rects:
                pygame.draw.rect(surface, color, r)
            pygame.draw.rect(
                surface, C_BG, pygame.Rect(cx - s // 2, cy - s // 2, s, s), 1
            )

        case UnitClass.CAVALIER:
            # Circle
            pygame.draw.circle(surface, color, (cx, cy), s // 2)
            pygame.draw.circle(surface, C_BG, (cx, cy), s // 2, 1)

        case UnitClass.BARBARIAN:
            # Hexagon
            pts = [
                (
                    cx + int(s // 2 * math.cos(math.radians(60 * i - 30))),
                    cy + int(s // 2 * math.sin(math.radians(60 * i - 30))),
                )
                for i in range(6)
            ]
            pygame.draw.polygon(surface, color, pts)
            pygame.draw.polygon(surface, C_BG, pts, 1)


def _draw_hp_bar(
    surface: pygame.Surface,
    unit: Unit,
    tile_x: int,
    tile_y: int,
) -> None:
    """Draws a small HP bar at the bottom of the tile."""
    bar_w = TILE_SIZE - 4
    bar_h = 4
    x = tile_x + 2
    y = tile_y + TILE_SIZE - bar_h - 2

    # Background
    pygame.draw.rect(surface, C_HP_BG, (x, y, bar_w, bar_h))

    # Fill
    ratio = unit.hp_ratio
    fill_w = max(1, int(bar_w * ratio))
    if ratio > 0.5:
        color = C_HP_GREEN
    elif ratio > 0.25:
        color = C_HP_YELLOW
    else:
        color = C_HP_RED
    pygame.draw.rect(surface, color, (x, y, fill_w, bar_h))


# ── Renderer ──────────────────────────────────────────────────────────────────


class PygameRenderer:
    """
    Pygame renderer for a TactilAI Grid.

    Parameters
    ----------
    grid       : Grid       the game grid to render
    fps        : int        frames per second in realtime mode
    step_mode  : bool       start in step-by-step mode

    Usage
    -----
    renderer = PygameRenderer(grid)

    # In your game/eval loop:
    while not done:
        action = agent.select_action(obs, mask)
        obs, reward, done, _, info = env.step(action)
        should_continue = renderer.render(env._grid)
        if not should_continue:
            break

    renderer.close()
    """

    def __init__(
        self,
        grid: Grid,
        fps: int = FPS_DEFAULT,
        step_mode: bool = False,
    ) -> None:
        pygame.init()
        pygame.display.set_caption("TactilAI")

        self.screen = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.fps = fps
        self.step_mode = step_mode
        self._running = True
        self._advance = False  # flag for step mode

        # Fonts
        self._font_sm = pygame.font.SysFont("monospace", 11, bold=True)
        self._font_md = pygame.font.SysFont("monospace", 13, bold=True)
        self._font_lg = pygame.font.SysFont("monospace", 15, bold=True)

        # Initial draw
        self.render(grid)

    # ── Public API ────────────────────────────────────────────────────────────

    def render(self, grid: Grid) -> bool:
        """
        Renders the current grid state.

        In step mode, blocks until SPACE or RIGHT is pressed.
        In realtime mode, advances at self.fps.

        Returns
        -------
        bool : False if the user requested quit, True otherwise.
        """
        if not self._running:
            return False

        self._handle_events()
        if not self._running:
            return False

        # Draw everything
        self.screen.fill(C_BG)
        self._draw_terrain(grid)
        self._draw_grid_lines()
        self._draw_units(grid)
        self._draw_panel(grid)
        self._draw_bottom_bar(grid)
        pygame.display.flip()

        # Timing
        if self.step_mode:
            self._wait_for_step()
        else:
            self.clock.tick(self.fps)

        return self._running

    def close(self) -> None:
        pygame.quit()

    # ── Drawing ───────────────────────────────────────────────────────────────

    def _draw_terrain(self, grid: Grid) -> None:
        for r in range(16):
            for c in range(16):
                terrain = grid.terrain_at((r, c))
                color = TERRAIN_COLORS[terrain]
                rect = pygame.Rect(c * TILE_SIZE, r * TILE_SIZE, TILE_SIZE, TILE_SIZE)
                pygame.draw.rect(self.screen, color, rect)

    def _draw_grid_lines(self) -> None:
        line_color = (0, 0, 0)
        for i in range(17):
            # Vertical
            pygame.draw.line(
                self.screen,
                line_color,
                (i * TILE_SIZE, 0),
                (i * TILE_SIZE, GRID_PX),
                1,
            )
            # Horizontal
            pygame.draw.line(
                self.screen,
                line_color,
                (0, i * TILE_SIZE),
                (GRID_PX, i * TILE_SIZE),
                1,
            )

    def _draw_units(self, grid: Grid) -> None:
        for unit in grid.alive_units():
            r, c = unit.pos
            tile_x = c * TILE_SIZE
            tile_y = r * TILE_SIZE
            cx = tile_x + TILE_SIZE // 2
            cy = tile_y + TILE_SIZE // 2 - 2

            # Sprite shape
            _draw_sprite(self.screen, unit, cx, cy, size=20)

            # Class letter
            letter = unit.unit_class.name[0]
            surf = self._font_sm.render(letter, True, C_BG)
            rect = surf.get_rect(center=(cx, cy))
            self.screen.blit(surf, rect)

            # HP bar
            _draw_hp_bar(self.screen, unit, tile_x, tile_y)

    def _draw_panel(self, grid: Grid) -> None:
        panel_x = GRID_PX
        panel_rect = pygame.Rect(panel_x, 0, PANEL_WIDTH, WIN_HEIGHT)
        pygame.draw.rect(self.screen, C_PANEL_BG, panel_rect)

        y = 12

        # Turn info
        turn_text = f"Turn {grid.turn:3d} / 100"
        self._blit(turn_text, panel_x + 10, y, self._font_lg, C_HIGHLIGHT)
        y += 22

        team_text = f"Active: {grid.active_team.name}"
        team_col = C_BLUE if grid.active_team == Team.BLUE else C_RED
        self._blit(team_text, panel_x + 10, y, self._font_md, team_col)
        y += 28

        # Separator
        pygame.draw.line(
            self.screen, C_TEXT_DIM, (panel_x + 8, y), (panel_x + PANEL_WIDTH - 8, y), 1
        )
        y += 10

        # Units list
        for team, color in ((Team.BLUE, C_BLUE), (Team.RED, C_RED)):
            label = "BLUE" if team == Team.BLUE else "RED "
            self._blit(f"── {label} ──", panel_x + 10, y, self._font_md, color)
            y += 18

            for unit in sorted(grid.alive_units(team), key=lambda u: u.uid):
                # Class name + HP
                name = unit.unit_class.name[:3]
                hp = f"{unit.current_hp:3d}/{unit.stats.hp:3d}"
                moved = "·" if unit.has_acted else "▶"
                line = f" {moved} {name}  HP:{hp}"
                col = C_TEXT_DIM if unit.has_acted else C_TEXT
                self._blit(line, panel_x + 8, y, self._font_sm, col)
                y += 15

            # Dead units count
            dead = NUM_UNITS - len(grid.alive_units(team))
            if dead > 0:
                self._blit(
                    f"   ({dead} dead)", panel_x + 8, y, self._font_sm, C_TEXT_DIM
                )
                y += 15
            y += 6

        # Separator
        pygame.draw.line(
            self.screen, C_TEXT_DIM, (panel_x + 8, y), (panel_x + PANEL_WIDTH - 8, y), 1
        )
        y += 10

        # Controls
        controls = [
            "SPACE  step/realtime",
            "→ / N  next step",
            "+/-    speed",
            "ESC    quit",
        ]
        self._blit("Controls:", panel_x + 10, y, self._font_sm, C_TEXT_DIM)
        y += 16
        for line in controls:
            self._blit(line, panel_x + 14, y, self._font_sm, C_TEXT_DIM)
            y += 14

        # Mode indicator
        mode = "STEP" if self.step_mode else f"LIVE {self.fps}fps"
        self._blit(
            f"Mode: {mode}",
            panel_x + 10,
            WIN_HEIGHT - 56,
            self._font_md,
            C_HIGHLIGHT,
        )

    def _draw_bottom_bar(self, grid: Grid) -> None:
        bar_rect = pygame.Rect(0, GRID_PX, GRID_PX, 48)
        pygame.draw.rect(self.screen, C_PANEL_BG, bar_rect)

        winner = grid.winner
        if winner is not None:
            text = f"{'BLUE' if winner == Team.BLUE else 'RED'} WINS!"
            color = C_BLUE if winner == Team.BLUE else C_RED
        elif grid.turn >= 100:
            text, color = "DRAW — max turns reached", C_TEXT_DIM
        else:
            alive_b = len(grid.alive_units(Team.BLUE))
            alive_r = len(grid.alive_units(Team.RED))
            text = f"Blue: {alive_b} units alive    Red: {alive_r} units alive"
            color = C_TEXT

        surf = self._font_lg.render(text, True, color)
        self.screen.blit(surf, (10, GRID_PX + 14))

    # ── Events ────────────────────────────────────────────────────────────────

    def _handle_events(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._running = False

            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    self._running = False

                elif event.key == pygame.K_SPACE:
                    self.step_mode = not self.step_mode

                elif event.key in (pygame.K_RIGHT, pygame.K_n):
                    self._advance = True

                elif event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
                    self.fps = min(60, self.fps + 2)

                elif event.key == pygame.K_MINUS:
                    self.fps = max(1, self.fps - 2)

    def _wait_for_step(self) -> None:
        """Blocks until the user presses SPACE, RIGHT or N."""
        self._advance = False
        while not self._advance and self._running:
            self._handle_events()
            if self.step_mode is False:
                break  # toggled back to realtime
            self.clock.tick(30)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _blit(
        self,
        text: str,
        x: int,
        y: int,
        font: pygame.font.Font,
        color: tuple[int, int, int],
    ) -> None:
        surf = font.render(text, True, color)
        self.screen.blit(surf, (x, y))


# ── Import guard ──────────────────────────────────────────────────────────────

from tactilai.env.grid import NUM_UNITS  # noqa: E402
