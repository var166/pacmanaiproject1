import pygame
import math
import random
from collections import deque
from queue import PriorityQueue
from board import boards
import copy

pygame.init()

width = 900
height = 950
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Pac-Man AI")
timer = pygame.time.Clock()
fps = 60
font = pygame.font.Font("freesansbold.ttf", 20)
small_font = pygame.font.Font("freesansbold.ttf", 16)
game_over_font = pygame.font.Font("freesansbold.ttf", 60)

pacman_move_interval = 6
ghost_move_interval = 6

num1 = (height - 50) // 32
num2 = width // 30

show_paths = True

fruit_active = False
fruit_x = 15
fruit_y = 18
fruit_spawn_interval = 30000
fruit_lifetime = 10000

next_fruit_time = pygame.time.get_ticks() + fruit_spawn_interval
fruit_end_time = 0

fruit = pygame.Surface((num2, num1), pygame.SRCALPHA)
pygame.draw.circle(fruit, 'red', (num2 // 2, num1 // 2), min(num1, num2) // 2 - 2)

level = [row[:] for row in boards]
PI = math.pi


STATE_PLAYING = 1
STATE_RESPAWNING = 2
STATE_GAMEOVER = 3
STATE_VICTORY = 4
current_state = STATE_PLAYING
respawn_start_time = 0

MODE_MANUAL = "MANUAL"
MODE_BFS    = "BFS"
MODE_DFS    = "DFS"
MODE_ASTAR  = "ASTAR"

AGENT_MANUAL     = "MANUAL"
AGENT_REFLEX     = "REFLEX"
AGENT_MINIMAX    = "MINIMAX"
AGENT_ALPHABETA  = "ALPHA-BETA"

current_ghost_algo = MODE_BFS
current_pacman_agent = AGENT_MANUAL

button_w, button_h = 160, 30
button_x = width - button_w - 10

label_ghosts = small_font.render("Ghost AI:", True, "white")
bfs_button    = pygame.Rect(button_x, 70,  button_w, button_h)
dfs_button    = pygame.Rect(button_x, 110, button_w, button_h)
astar_button  = pygame.Rect(button_x, 150, button_w, button_h)

label_pacman = small_font.render("Pacman Mode:", True, "white")
manual_button   = pygame.Rect(button_x, 240, button_w, button_h)
reflex_button   = pygame.Rect(button_x, 280, button_w, button_h)
minimax_button  = pygame.Rect(button_x, 320, button_w, button_h)
ab_button       = pygame.Rect(button_x, 360, button_w, button_h)


player_images = []
for i in range(1, 2 + 1):
    player_images.append(
        pygame.transform.scale(
            pygame.image.load(f"assets/pacman_photos/{i}.png"),
            (num2, num1)
        )
    )

PACMAN_START = (14, 18)
player_x, player_y = PACMAN_START
direction = None
next_direction = None
counter = 0
score = 0
lives = 3
life_image = pygame.transform.scale(
    pygame.image.load("assets/pacman_photos/1.png"),
    (num2, num1)
)

power_up = False
power_up_duration = 8000
power_up_end_time = 0

ghost_size = (num2, num1)

def load_ghost(path: str) -> pygame.Surface:
    return pygame.transform.scale(pygame.image.load(path), ghost_size)

blinky_imgs = {
    "normal": load_ghost("assets/ghosts_photos/red.png"),
    "scared": load_ghost("assets/ghosts_photos/spooked.png"),
    "dead":   load_ghost("assets/ghosts_photos/dead.png"),
}
pinky_imgs = {
    "normal": load_ghost("assets/ghosts_photos/pinky.png"),
    "scared": load_ghost("assets/ghosts_photos/spooked.png"),
    "dead":   load_ghost("assets/ghosts_photos/dead.png"),
}
inky_imgs = {
    "normal": load_ghost("assets/ghosts_photos/incky.png"),
    "scared": load_ghost("assets/ghosts_photos/spooked.png"),
    "dead":   load_ghost("assets/ghosts_photos/dead.png"),
}
clyde_imgs = {
    "normal": load_ghost("assets/ghosts_photos/yelo.png"),
    "scared": load_ghost("assets/ghosts_photos/spooked.png"),
    "dead":   load_ghost("assets/ghosts_photos/dead.png"),
}

def manhattan(p1, p2) -> int:
    x1, y1 = p1
    x2, y2 = p2
    dx = abs(x1 - x2)
    dy = abs(y1 - y2)
    
    board_width = len(level[0])
    if dx > board_width / 2:
        dx = board_width - dx
        
    return dx + dy

def reconstruct_path(came_from, start, end):
    if end not in came_from:
        return [start]

    path = [end]
    cur = end
    while cur != start:
        cur = came_from[cur]
        path.append(cur)
    path.reverse()
    return path

def bfs_path(start, end, neighbor_fn):
    if start is None or end is None:
        return [start]

    queue = deque([start])
    visited = {start}
    came_from = {}

    while queue:
        current = queue.popleft()

        if current == end:
            return reconstruct_path(came_from, start, end)

        for nb in neighbor_fn(current):
            if nb not in visited:
                visited.add(nb)
                came_from[nb] = current
                queue.append(nb)

    return [start]

def dfs_path(start, end, neighbor_fn):
    if start is None or end is None:
        return [start]

    stack = [start]
    visited = {start}
    came_from = {}

    while stack:
        current = stack.pop()

        if current == end:
            return reconstruct_path(came_from, start, end)

        for nb in neighbor_fn(current):
            if nb not in visited:
                visited.add(nb)
                came_from[nb] = current
                stack.append(nb)

    return [start]

def astar_path(start, end, neighbor_fn):
    if start is None or end is None:
        return [start]

    open_heap = PriorityQueue()
    count = 0
    open_heap.put((0, count, start))

    came_from = {}
    g_score = {start: 0}
    f_score = {start: manhattan(start, end)}
    in_open = {start}

    while not open_heap.empty():
        _, _, current = open_heap.get()
        in_open.remove(current)

        if current == end:
            return reconstruct_path(came_from, start, end)

        for nb in neighbor_fn(current):
            tentative_g = g_score[current] + 1

            if nb not in g_score or tentative_g < g_score[nb]:
                came_from[nb] = current
                g_score[nb] = tentative_g
                f_score[nb] = tentative_g + manhattan(nb, end)

                if nb not in in_open:
                    count += 1
                    open_heap.put((f_score[nb], count, nb))
                    in_open.add(nb)

    return [start]

def find_path(algo, start, goal, neighbor_fn):
    if algo == MODE_BFS:
        return bfs_path(start, goal, neighbor_fn)
    elif algo == MODE_DFS:
        return dfs_path(start, goal, neighbor_fn)
    else:
        return astar_path(start, goal, neighbor_fn)

def ghost_tile_walkable(x: int, y: int) -> bool:
    if not (0 <= y < len(level) and 0 <= x < len(level[0])):
        return False
    tile = level[y][x]
    return tile in (0, 1, 2, 9)

def nearest_walkable(target):
    tx = target[0] % len(level[0])
    ty = target[1]
    
    start = (tx, ty)
    if ghost_tile_walkable(tx, ty):
        return start

    queue = deque([start])
    visited = {start}

    while queue:
        x, y = queue.popleft()
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx = (x + dx) % len(level[0])
            ny = y + dy
            
            if (nx, ny) in visited:
                continue
            visited.add((nx, ny))
            if ghost_tile_walkable(nx, ny):
                return (nx, ny)
            queue.append((nx, ny))

    return start

ghost_mode_sequence = [
    ("scatter", 2000),
    ("chase", 20000),
    ("scatter", 2000),
    ("chase", 20000),
    ("scatter", 3000),
    ("chase", -1)
]

ghost_mode_index = 0
ghost_mode = ghost_mode_sequence[0][0]
ghost_mode_start_time = pygame.time.get_ticks()

def update_ghost_global_mode():
    global ghost_mode, ghost_mode_index, ghost_mode_start_time
    mode, duration = ghost_mode_sequence[ghost_mode_index]
    if duration == -1:
        return
    now = pygame.time.get_ticks()
    if now - ghost_mode_start_time >= duration:
        ghost_mode_index = min(
            ghost_mode_index + 1,
            len(ghost_mode_sequence) - 1
        )
        ghost_mode = ghost_mode_sequence[ghost_mode_index][0]
        ghost_mode_start_time = now

RAW_SCATTER_TL = (1, 1)
RAW_SCATTER_TR = (28, 1)
RAW_SCATTER_BR = (28, 30)
RAW_SCATTER_BL = (1, 30)

scatter_tl = nearest_walkable(RAW_SCATTER_TL)
scatter_tr = nearest_walkable(RAW_SCATTER_TR)
scatter_br = nearest_walkable(RAW_SCATTER_BR)
scatter_bl = nearest_walkable(RAW_SCATTER_BL)

GHOST_HOME = (14, 15)

def ghost_neighbors(node, ghost):
    x, y = node
    neigh = []
    directions = {
        0: (1, 0),
        1: (-1, 0),
        2: (0, -1),
        3: (0, 1)
    }
    reverse_map = {0: 1, 1: 0, 2: 3, 3: 2}
    forbidden = reverse_map.get(ghost.direction, -1)
    is_start = (x == ghost.x and y == ghost.y)

    for d_code, (dx, dy) in directions.items():
        if is_start and d_code == forbidden:
            continue

        nx, ny = (x + dx) % len(level[0]), y + dy
        
        if ghost_tile_walkable(nx, ny):
            if level[ny][nx] == 9 and ghost.state == "scatter" and ny > y:
                continue
            neigh.append((nx, ny))

    if is_start and not neigh:
        back_dx, back_dy = directions[forbidden]
        bx, by = (x + back_dx) % len(level[0]), y + back_dy
        if ghost_tile_walkable(bx, by):
            neigh.append((bx, by))

    return neigh

class Ghost:
    def __init__(self, name, start_tile, images, scatter_cycle):
        self.name = name
        self.start = start_tile
        self.x, self.y = start_tile
        self.images = images
        self.scatter_cycle = scatter_cycle
        self.scatter_index = 0
        self.state = "chase"
        self.image = self.images["normal"]
        self.path = []
        self.direction = 0
        self.eaten = False
        self.spawn_delay = 0  

    def reset(self):
        self.x, self.y = self.start
        self.state = "chase"
        self.image = self.images["normal"]
        self.path = []
        self.direction = 0
        self.scatter_index = 0
        self.eaten = False
        
        now = pygame.time.get_ticks()
        
       
        if self.name == "pinky":
            self.spawn_delay = now + 500
        elif self.name == "inky":
            self.spawn_delay = now + 1000
        elif self.name == "clyde":
            self.spawn_delay = now + 1500
        else:
            self.spawn_delay = 0 
    def draw(self):
        screen.blit(self.image, (self.x * num2, self.y * num1))

    def update_state(self):
        if self.state == "dead":
            self.image = self.images["dead"]
            if (self.x, self.y) == GHOST_HOME:
                self.state = ghost_mode
                self.image = self.images["normal"]
               
                self.spawn_delay = pygame.time.get_ticks() + 1000
            return

        if power_up and not self.eaten:
            self.state = "frightened"
            self.image = self.images["scared"]
        else:
            self.state = ghost_mode
            self.image = self.images["normal"]

    def get_chase_target(self, pac_pos, pac_dir, blinky_pos):
        px, py = pac_pos
        if self.name == "blinky":
            return (px, py)

        elif self.name == "pinky":
            dx = dy = 0
            if pac_dir == 0: dx = 4
            elif pac_dir == 1: dx = -4
            elif pac_dir == 2: dy = -4
            elif pac_dir == 3: dy = 4
            return (px + dx, py + dy)

        elif self.name == "inky":
            bx, by = blinky_pos
            current_dir = pac_dir if pac_dir is not None else 0

            pivot_x, pivot_y = px, py
            if current_dir == 0: pivot_x += 2
            elif current_dir == 1: pivot_x -= 2
            elif current_dir == 2: pivot_y -= 2
            elif current_dir == 3: pivot_y += 2

            vector_x = pivot_x - bx
            vector_y = pivot_y - by

            return (bx + 2 * vector_x, by + 2 * vector_y)

        elif self.name == "clyde":
            gx, gy = self.x, self.y
            dist = manhattan((gx, gy), (px, py))
            if dist <= 8:
                return (px, py)
            else:
                return self.scatter_cycle[self.scatter_index]

    def frightened_move(self, pac_pos):
        self.path = []
        start = (self.x, self.y)
        cur_dist = manhattan(start, pac_pos)
        candidates = []
        for nb in ghost_neighbors(start, self):
            if manhattan(nb, pac_pos) > cur_dist:
                candidates.append(nb)
        if not candidates:
            candidates = ghost_neighbors(start, self)
        if candidates:
            nx, ny = random.choice(candidates)
            dx, dy = nx - self.x, ny - self.y
            
            if dx == 1 or dx < -1: self.direction = 0
            elif dx == -1 or dx > 1: self.direction = 1
            elif dy == -1: self.direction = 2
            elif dy == 1: self.direction = 3
            self.x, self.y = nx, ny

    def move(self, pac_pos, pac_dir, blinky_pos):
        
        if pygame.time.get_ticks() < self.spawn_delay:
            return

        start = (self.x, self.y)

        if self.state == "frightened":
            self.frightened_move(pac_pos)
            return

        if self.state == "dead":
            target = GHOST_HOME
        elif self.state == "scatter":
            target = self.scatter_cycle[self.scatter_index]

            if (self.x, self.y) == target:
                self.scatter_index = (self.scatter_index + 1) % len(self.scatter_cycle)
                target = self.scatter_cycle[self.scatter_index]

        else:
            target = self.get_chase_target(pac_pos, pac_dir, blinky_pos)

        max_y = len(level) - 1
        max_x = len(level[0])
        
        tx = int(target[0]) % max_x
        ty = max(0, min(max_y, int(target[1])))
        target = (tx, ty)

        path = find_path(
            current_ghost_algo,
            start,
            target,
            lambda n: ghost_neighbors(n, self)
        )
        self.path = path
        
        next_pos = None
        if len(path) > 1:
            next_pos = path[1]
        else:
            neighbors = ghost_neighbors(start, self)
            if neighbors:
                next_pos = min(neighbors, key=lambda n: manhattan(n, target))

        if next_pos:
            nx, ny = next_pos
            dx, dy = nx - self.x, ny - self.y
            
            if dx == 1 or dx < -1: self.direction = 0
            elif dx == -1 or dx > 1: self.direction = 1
            elif dy == -1: self.direction = 2
            elif dy == 1: self.direction = 3
            self.x, self.y = nx, ny

scatter_corners_tl = [
    nearest_walkable((2, 2)),
    nearest_walkable((2, 7)),
    nearest_walkable((7, 7)),
    nearest_walkable((7, 2))
]

scatter_corners_tr = [
    nearest_walkable((27, 2)),
    nearest_walkable((27, 7)),
    nearest_walkable((22, 7)),
    nearest_walkable((22, 2))
]

scatter_corners_br = [
    nearest_walkable((27, 29)),
    nearest_walkable((27, 23)),
    nearest_walkable((22, 23)),
    nearest_walkable((22, 29))
]

scatter_corners_bl = [
    nearest_walkable((2, 29)),
    nearest_walkable((2, 23)),
    nearest_walkable((7, 23)),
    nearest_walkable((7, 29))
]

blinky = Ghost("blinky", (13, 16), blinky_imgs, scatter_corners_tr)
pinky  = Ghost("pinky",  (13, 15), pinky_imgs,  scatter_corners_tl)
inky   = Ghost("inky",   (15, 15), inky_imgs,   scatter_corners_br)
clyde  = Ghost("clyde",  (14, 15), clyde_imgs,  scatter_corners_bl)
ghosts = [blinky, pinky, inky, clyde]

def draw_board():
    for i in range(len(level)):
        for j in range(len(level[i])):
            if level[i][j] == 1:
                pygame.draw.circle(screen, 'white', ((j*num2+ (0.5*num2)), (i*num1 + (0.5*num1))), 4)
            if level[i][j] == 2:
                pygame.draw.circle(screen, 'white', ((j*num2+ (0.5*num2)), (i*num1 + (0.5*num1))), 10)
            if level[i][j] == 3:
                pygame.draw.line(screen, 'blue', ((j*num2+ (0.5*num2)), (i*num1)), ((j*num2+ (0.5*num2)), (i+1)*num1), 5)
            if level[i][j] == 4:
                pygame.draw.line(screen, 'blue', ((j*num2), (i*num1+0.5*num1)), (((j+1)*num2), i*num1+ 0.5*num1),5)
            if level[i][j] == 9:
                pygame.draw.line(screen, 'white', ((j*num2), (i*num1+0.5*num1)), (((j+1)*num2), i*num1+ 0.5*num1),5)
            if level[i][j] == 5:
                pygame.draw.arc(screen, 'blue', [(j*num2 - 0.5*num2), (i*num1 + 0.5*num1),num2, num1],0,PI/2,5)
            if level[i][j] == 6:
                pygame.draw.arc(screen, 'blue', [(j*num2 + 0.5*num2), (i*num1 + 0.5*num1),num2, num1],PI/2,PI,5)
            if level[i][j] == 7:
                pygame.draw.arc(screen, 'blue', [(j*num2 + 0.5*num2), (i*num1 - 0.5*num1),num2, num1],PI,3*PI/2,5)
            if level[i][j] == 8:
                pygame.draw.arc(screen, 'blue', [(j*num2 - 0.5*num2), (i*num1 - 0.5*num1),num2, num1],3*PI/2, 2*PI,5)

def modify_board():
    global score, power_up, power_up_end_time
    tile = level[player_y][player_x]
    if tile == 1:
        score += 10
        level[player_y][player_x] = 0
    elif tile == 2:
        score += 50
        level[player_y][player_x] = 0
        now = pygame.time.get_ticks()
        
        power_up = True
        power_up_end_time = now + power_up_duration
        for g in ghosts:
            if g.state != "dead":
                g.eaten = False
                g.state = "frightened"
                g.image = g.images["scared"]

def update_power_up():
    global power_up
    if power_up and pygame.time.get_ticks() >= power_up_end_time:
        power_up = False

def draw_power_up_text():
    if power_up:
        txt = font.render("POWER UP!", True, "yellow")
        screen.blit(txt, (width - 220, height - 30))

def draw_score():
    txt = font.render(f"Score: {score}", True, "white")
    screen.blit(txt, (20, height - 30))

def draw_lives():
    for i in range(lives):
        screen.blit(life_image, (20 + i * (num2 + 5), height - 60))

def draw_fruit():
    if fruit_active:
        screen.blit(fruit, (fruit_x * num2, fruit_y * num1))

def update_fruit():
    global fruit_active, next_fruit_time, fruit_end_time, score
    now = pygame.time.get_ticks()
    if not fruit_active and now >= next_fruit_time:
        fruit_active = True
        fruit_end_time = now + fruit_lifetime
    elif fruit_active and now >= fruit_end_time:
        fruit_active = False
        next_fruit_time = now + fruit_spawn_interval

    if fruit_active and player_x == fruit_x and player_y == fruit_y:
        score += 500
        fruit_active = False
        next_fruit_time = now + fruit_spawn_interval

def draw_buttons():
    screen.blit(label_ghosts, (button_x, 40))
    ghost_buttons = [
        ("BFS",    bfs_button,    MODE_BFS),
        ("DFS",    dfs_button,    MODE_DFS),
        ("A*",     astar_button,  MODE_ASTAR),
    ]
    for name, rect, mode in ghost_buttons:
        selected = (current_ghost_algo == mode)
        color = (200, 200, 0) if selected else (40, 40, 40)
        pygame.draw.rect(screen, color, rect)
        pygame.draw.rect(screen, "white", rect, 2)
        text = small_font.render(name, True, "white")
        screen.blit(
            text,
            (rect.x + (rect.width - text.get_width()) // 2,
             rect.y + (rect.height - text.get_height()) // 2)
        )

    screen.blit(label_pacman, (button_x, 210))
    pacman_buttons = [
        ("Manual",     manual_button,   AGENT_MANUAL),
        ("Reflex",     reflex_button,   AGENT_REFLEX),
        ("Minimax",    minimax_button,  AGENT_MINIMAX),
        ("AlphaBeta",  ab_button,       AGENT_ALPHABETA)
    ]
    for name, rect, agent in pacman_buttons:
        selected = (current_pacman_agent == agent)
        color = (200, 200, 0) if selected else (40, 40, 40)
        pygame.draw.rect(screen, color, rect)
        pygame.draw.rect(screen, "white", rect, 2)
        text = small_font.render(name, True, "white")
        screen.blit(
            text,
            (rect.x + (rect.width - text.get_width()) // 2,
             rect.y + (rect.height - text.get_height()) // 2)
        )


def draw_player():
    img = player_images[(counter // 20) % 2]
    if direction == 0:
        screen.blit(img, (num2 * player_x, num1 * player_y))
    elif direction == 1:
        screen.blit(pygame.transform.flip(img, True, False),
                    (num2 * player_x, num1 * player_y))
    elif direction == 2:
        screen.blit(pygame.transform.rotate(img, 90),
                    (num2 * player_x, num1 * player_y))
    elif direction == 3:
        screen.blit(pygame.transform.rotate(img, 270),
                    (num2 * player_x, num1 * player_y))
    else:
        screen.blit(img, (num2 * player_x, num1 * player_y))

def draw_paths():
    if not show_paths:
        return

    for g in ghosts:
        if not g.path or len(g.path) < 2:
            continue

        if g.name == "blinky":
            color = (255, 0, 0, 80)
        elif g.name == "pinky":
            color = (255, 105, 180, 80)
        elif g.name == "inky":
            color = (0, 255, 255, 80)
        elif g.name == "clyde":
            color = (255, 165, 0, 80)
        else:
            color = (255, 255, 255, 80)

        tile_surf = pygame.Surface((num2, num1), pygame.SRCALPHA)
        tile_surf.fill(color)

        for (tx, ty) in g.path:
            screen.blit(tile_surf, (tx * num2, ty * num1))

def can_move(x, y, dir_code):
    max_y = len(level) - 1

    if dir_code == 0:
        nx = (x + 1) % 30
        return level[y][nx] not in (3, 4, 5, 6, 7, 8, 9)
    elif dir_code == 1:
        nx = (x - 1) % 30
        return level[y][nx] not in (3, 4, 5, 6, 7, 8, 9)
    elif dir_code == 2:
        if y <= 0:
            return False
        return level[y - 1][x] not in (3, 4, 5, 6, 7, 8, 9)
    elif dir_code == 3:
        if y >= max_y:
            return False
        return level[y + 1][x] not in (3, 4, 5, 6, 7, 8, 9)
    return False

def is_opposite_direction(d1, d2):
    if d1 is None or d2 is None: return False
    if (d1 == 0 and d2 == 1) or (d1 == 1 and d2 == 0): return True
    if (d1 == 2 and d2 == 3) or (d1 == 3 and d2 == 2): return True
    return False

def get_valid_moves(x, y):
    moves = []
    for d in range(4):
        if can_move(x, y, d):
            moves.append(d)
    return moves

def get_next_pos(x, y, move):
    if move == 0: return ((x + 1) % 30, y)
    if move == 1: return ((x - 1) % 30, y)
    if move == 2: return (x, y - 1)
    if move == 3: return (x, y + 1)
    return (x, y)

def get_nearest_food_dist(start_pos, board_grid, frightened_ghosts_pos):
   
    dangerous_tiles = set()
    if not power_up:
        for ghost_p in ghosts:
            if ghost_p.state not in ["dead", "frightened"]:
                dangerous_tiles.add((ghost_p.x, ghost_p.y))

    targets = []
    
    for fg_pos in frightened_ghosts_pos:
        targets.append(fg_pos)
        
    for r in range(len(board_grid)):
        for c in range(len(board_grid[0])):
            if board_grid[r][c] in [1, 2]:
                targets.append((c, r))
    
    
    if not targets and not fruit_active:
        return 9999

    closest_target_dist = float('inf')
    
    
    queue = deque([(start_pos, 0)])
    visited = {start_pos}
    
    found_food = False
    
    while queue:
        curr, dist = queue.popleft()
        
    
        cx, cy = curr
        if board_grid[cy][cx] in [1, 2] or curr in frightened_ghosts_pos:
            return dist
        
    
        if fruit_active and curr == (fruit_x, fruit_y):
            return dist

        for d_idx in range(4):
            mx, my = -1, -1
            if d_idx == 0: mx, my = (cx + 1) % 30, cy
            elif d_idx == 1: mx, my = (cx - 1) % 30, cy
            elif d_idx == 2: mx, my = cx, cy - 1
            elif d_idx == 3: mx, my = cx, cy + 1
            
            if 0 <= my < len(board_grid):
               
                if board_grid[my][mx] not in [3,4,5,6,7,8,9]:
               
                    if (mx, my) not in dangerous_tiles:
                        if (mx, my) not in visited:
                            visited.add((mx, my))
                            queue.append(((mx, my), dist + 1))
    
    return 9999 

def reflex_agent_move():
    valid_moves = get_valid_moves(player_x, player_y)
    
    if not valid_moves:
        return random.choice([0,1,2,3])

    best_score = -float('inf')
    best_move = valid_moves[0]
    
    remaining_power = 0
    if power_up:
        remaining_power = power_up_end_time - pygame.time.get_ticks()

    frightened_positions = set()
    if power_up and remaining_power > 2000:
        for g in ghosts:
            if g.state != "dead" and not g.eaten:
                frightened_positions.add((g.x, g.y))

    for move in valid_moves:
        nx, ny = get_next_pos(player_x, player_y, move)
        score = 0
        
        
        dist_to_food = get_nearest_food_dist((nx, ny), level, frightened_positions)
        
        
        score += 50 * (100 - dist_to_food)
        
        if direction is not None and is_opposite_direction(move, direction):
            score -= 200

       
        if fruit_active:
            d_fruit = manhattan((nx, ny), (fruit_x, fruit_y))
            score += 20 * (50 - d_fruit)

        for g in ghosts:
            dist = manhattan((nx, ny), (g.x, g.y))
            if g.state == "dead":
                continue
            
            is_scared = False
            if power_up and not g.eaten:
                if remaining_power > 2000:
                    is_scared = True
            
            if is_scared:
                if dist < 15:
                    score += 2000 / (dist + 1)
            else:
              
                if dist < 5:
                    score -= 50000 / (dist + 0.1)

        if level[ny][nx] == 1: score += 100
        if level[ny][nx] == 2: score += 500
        
        if score > best_score:
            best_score = score
            best_move = move
            
    return best_move

def evaluate_state(p_pos, g_positions, food_grid, current_score, ghost_states):
    val = current_score
    
    remaining_power = 0
    if power_up:
        remaining_power = power_up_end_time - pygame.time.get_ticks()

    frightened_positions = set()
    for i, g_pos in enumerate(g_positions):
        if ghost_states[i] == "dead":
            continue

        g_eaten = ghosts[i].eaten

        is_scared = False
        if power_up and not g_eaten:
            if remaining_power > 2000:
                is_scared = True
        
        if is_scared:
            frightened_positions.add(g_pos)
            dist = manhattan(p_pos, g_pos)
            if dist < 15: val += 5000 / (dist + 1)
        else:
             dist = manhattan(p_pos, g_pos)
            
             if dist < 5: val -= 50000 / (dist + 1)
    
    dist_to_food = get_nearest_food_dist(p_pos, food_grid, frightened_positions)
    val += 100 * (100 - dist_to_food)

    
    if fruit_active:
        d_fruit = manhattan(p_pos, (fruit_x, fruit_y))
        val += 50 * (50 - d_fruit)
    
    return val

def minimax(depth, agent_index, p_pos, g_positions, current_score, ghost_states, is_maximizing):
    if depth == 0:
        return evaluate_state(p_pos, g_positions, level, current_score, ghost_states)
    
    if is_maximizing:
        valid_moves = []
        for d in range(4):
            if can_move(p_pos[0], p_pos[1], d):
                valid_moves.append(d)
        
        if not valid_moves: return evaluate_state(p_pos, g_positions, level, current_score, ghost_states)
        
        max_eval = -float('inf')
        for move in valid_moves:
            nx, ny = get_next_pos(p_pos[0], p_pos[1], move)
            new_score = current_score
            if level[ny][nx] == 1: new_score += 10
            
            eval = minimax(depth - 1, 0, (nx, ny), g_positions, new_score, ghost_states, False)
            max_eval = max(max_eval, eval)
        return max_eval
    
    else:
        new_g_positions = list(g_positions)
        
        for i in range(len(new_g_positions)):
            gx, gy = new_g_positions[i]
            best_g_dist = float('inf')
            best_g_pos = (gx, gy)
            
            possible_g_moves = []
            for dx, dy in [(0,1), (0,-1), (1,0), (-1,0)]:
                ngx, ngy = (gx + dx) % 30, gy + dy
                if 0 <= ngy < len(level) and level[ngy][ngx] not in [3,4,5,6,7,8]:
                    dist = manhattan((ngx, ngy), p_pos)
                    if dist < best_g_dist:
                        best_g_dist = dist
                        best_g_pos = (ngx, ngy)
            new_g_positions[i] = best_g_pos
            
            if best_g_pos == p_pos:
                if ghost_states[i] != "frightened" and ghost_states[i] != "dead":
                    return -10000
        
        return minimax(depth - 1, 0, p_pos, new_g_positions, current_score, ghost_states, True)

def minimax_agent_move():
    depth = 10
    valid_moves = get_valid_moves(player_x, player_y)
    
    best_move = random.choice(valid_moves) if valid_moves else 0
    max_eval = -float('inf')
    
    g_positions = [(g.x, g.y) for g in ghosts]
    g_states = [g.state for g in ghosts]
    
    for move in valid_moves:
        nx, ny = get_next_pos(player_x, player_y, move)
        collision = False
        for i, pos in enumerate(g_positions):
            if pos == (nx, ny) and g_states[i] not in ["frightened", "dead"]:
                collision = True
        if collision: continue
            
        eval_val = minimax(depth, 0, (nx, ny), g_positions, score, g_states, False)
        
        if eval_val > max_eval:
            max_eval = eval_val
            best_move = move
            
    return best_move

def alphabeta(depth, p_pos, g_positions, current_score, ghost_states, alpha, beta, is_maximizing):
    if depth == 0:
        return evaluate_state(p_pos, g_positions, level, current_score, ghost_states)
    
    if is_maximizing:
        valid_moves = []
        for d in range(4):
            if can_move(p_pos[0], p_pos[1], d):
                valid_moves.append(d)
        
        if not valid_moves: return evaluate_state(p_pos, g_positions, level, current_score, ghost_states)
        
        max_eval = -float('inf')
        for move in valid_moves:
            nx, ny = get_next_pos(p_pos[0], p_pos[1], move)
            new_score = current_score
            if level[ny][nx] == 1: new_score += 10
            
            eval = alphabeta(depth - 1, (nx, ny), g_positions, new_score, ghost_states, alpha, beta, False)
            max_eval = max(max_eval, eval)
            alpha = max(alpha, eval)
            if beta <= alpha:
                break
        return max_eval
    else:
        new_g_positions = list(g_positions)
        min_eval = float('inf')
        
        for i in range(len(new_g_positions)):
            gx, gy = new_g_positions[i]
            best_g_dist = float('inf')
            best_g_pos = (gx, gy)
            for dx, dy in [(0,1), (0,-1), (1,0), (-1,0)]:
                ngx, ngy = (gx + dx) % 30, gy + dy
                if 0 <= ngy < len(level) and level[ngy][ngx] not in [3,4,5,6,7,8]:
                    dist = manhattan((ngx, ngy), p_pos)
                    if dist < best_g_dist:
                        best_g_dist = dist
                        best_g_pos = (ngx, ngy)
            new_g_positions[i] = best_g_pos
            if best_g_pos == p_pos and ghost_states[i] not in ["frightened", "dead"]:
                return -10000
        
        eval = alphabeta(depth - 1, p_pos, new_g_positions, current_score, ghost_states, alpha, beta, True)
        min_eval = min(min_eval, eval)
        beta = min(beta, eval)
        
        return min_eval

def alphabeta_agent_move():
    depth = 10
    valid_moves = get_valid_moves(player_x, player_y)
    
    best_move = random.choice(valid_moves) if valid_moves else 0
    max_eval = -float('inf')
    alpha = -float('inf')
    beta = float('inf')
    
    g_positions = [(g.x, g.y) for g in ghosts]
    g_states = [g.state for g in ghosts]
    
    for move in valid_moves:
        nx, ny = get_next_pos(player_x, player_y, move)
        collision = False
        for i, pos in enumerate(g_positions):
            if pos == (nx, ny) and g_states[i] not in ["frightened", "dead"]:
                collision = True
        if collision: continue
        
        eval_val = alphabeta(depth, (nx, ny), g_positions, score, g_states, alpha, beta, False)
        
        if eval_val > max_eval:
            max_eval = eval_val
            best_move = move
        alpha = max(alpha, eval_val)
            
    return best_move

def update_pacman():
    global player_x, player_y, direction, next_direction

    if current_pacman_agent == AGENT_REFLEX:
        next_direction = reflex_agent_move()
    elif current_pacman_agent == AGENT_MINIMAX:
        next_direction = minimax_agent_move()
    elif current_pacman_agent == AGENT_ALPHABETA:
        next_direction = alphabeta_agent_move()
    
    if next_direction is not None and can_move(player_x, player_y, next_direction):
        direction = next_direction

    if direction is None:
        return

    if can_move(player_x, player_y, direction):
        if direction == 0:
            player_x = (player_x + 1) % 30
        elif direction == 1:
            player_x = (player_x - 1) % 30
        elif direction == 2:
            player_y -= 1
        elif direction == 3:
            player_y += 1

def reset_positions():
    global player_x, player_y, direction, power_up, respawn_start_time
    player_x, player_y = PACMAN_START
    direction = None
    power_up = False
    for g in ghosts:
        g.reset()

def reset_game():
    global score, lives, current_state, level
    score = 0
    lives = 3
    current_state = STATE_PLAYING
    level = [row[:] for row in boards]
    reset_positions()

def check_victory():
    global current_state
    won = True
    for i in range(len(level)):
        if 1 in level[i] or 2 in level[i]:
            won = False
            break
    if won:
        current_state = STATE_VICTORY

def handle_collisions():
    global score, lives, current_state, respawn_start_time

   
    if current_state != STATE_PLAYING:
        return

    for g in ghosts:
        if g.x == player_x and g.y == player_y:
            if g.state == "frightened":
                score += 200
                g.state = "dead"
                g.eaten = True
                return

            if g.state in ("chase", "scatter"):
                score = max(0, score - 500)
                lives -= 1

                if lives <= 0:
                    current_state = STATE_GAMEOVER
                else:
                    current_state = STATE_RESPAWNING
                    respawn_start_time = pygame.time.get_ticks()
                return

run = True
while run:
    timer.tick(fps)
    counter += 1
    screen.fill("black")

    if current_state == STATE_PLAYING:
        if not power_up:
            update_ghost_global_mode()

        if counter % pacman_move_interval == 0:
            update_pacman()
            handle_collisions() 
        pac_pos = (player_x, player_y)

        for g in ghosts:
            g.update_state()
            
            should_move = False
            if g.state == "dead":
                if counter % 2 == 0: should_move = True
            elif g.state == "frightened":
                if counter % 15 == 0 or counter % 15 == 8: should_move = True
            else:
                if counter % ghost_move_interval == 0: should_move = True

            if should_move:
                g.move(pac_pos, direction if direction is not None else 0, (blinky.x, blinky.y))
                handle_collisions() 

        modify_board()
        check_victory()
        update_power_up()
        update_fruit()
        handle_collisions()

    elif current_state == STATE_RESPAWNING:
        if pygame.time.get_ticks() - respawn_start_time > 2000:
            reset_positions()
            current_state = STATE_PLAYING

    draw_board()
    draw_fruit()
    draw_score()
    draw_lives()
    draw_power_up_text()
    draw_buttons()
    draw_paths()

    for g in ghosts:
        g.draw()
    draw_player()

    if current_state == STATE_GAMEOVER:
        game_over_text = game_over_font.render("GAME OVER", True, "red")
        restart_text = font.render("Press SPACE to Restart", True, "white")
        screen.blit(game_over_text, (width // 2 - 150, height // 2 - 50))
        screen.blit(restart_text, (width // 2 - 120, height // 2 + 20))
    elif current_state == STATE_VICTORY:
        victory_text = game_over_font.render("VICTORY!", True, "green")
        restart_text = font.render("Press SPACE to Restart", True, "white")
        screen.blit(victory_text, (width // 2 - 130, height // 2 - 50))
        screen.blit(restart_text, (width // 2 - 120, height // 2 + 20))

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False

        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            mx, my = event.pos
            if bfs_button.collidepoint(mx, my):
                current_ghost_algo = MODE_BFS
                current_algo = MODE_BFS
            elif dfs_button.collidepoint(mx, my):
                current_ghost_algo = MODE_DFS
                current_algo = MODE_DFS
            elif astar_button.collidepoint(mx, my):
                current_ghost_algo = MODE_ASTAR
                current_algo = MODE_ASTAR
            
            elif manual_button.collidepoint(mx, my):
                current_pacman_agent = AGENT_MANUAL
            elif reflex_button.collidepoint(mx, my):
                current_pacman_agent = AGENT_REFLEX
            elif minimax_button.collidepoint(mx, my):
                current_pacman_agent = AGENT_MINIMAX
            elif ab_button.collidepoint(mx, my):
                current_pacman_agent = AGENT_ALPHABETA

        elif event.type == pygame.KEYDOWN:
            if current_state == STATE_GAMEOVER or current_state == STATE_VICTORY:
                if event.key == pygame.K_SPACE:
                    reset_game()
            else:
                if event.key == pygame.K_p:
                    show_paths = not show_paths
                if current_pacman_agent == AGENT_MANUAL:
                    if event.key == pygame.K_RIGHT:
                        next_direction = 0
                    elif event.key == pygame.K_LEFT:
                        next_direction = 1
                    elif event.key == pygame.K_UP:
                        next_direction = 2
                    elif event.key == pygame.K_DOWN:
                        next_direction = 3

    pygame.display.flip()

pygame.quit()