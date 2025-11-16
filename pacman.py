import pygame
import math
import random
from collections import deque
from queue import PriorityQueue
from board import boards

pygame.init()

width = 900
height = 950
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Pac-Man AI")
timer = pygame.time.Clock()
fps = 60
font = pygame.font.Font("freesansbold.ttf", 20)
ghost_move_interval = 8
ghost_move_counter = 0
num1 = (height - 50) // 32   # tile height
num2 = width // 30           # tile width

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


MODE_MANUAL = "MANUAL"
MODE_BFS    = "BFS"
MODE_DFS    = "DFS"
MODE_ASTAR  = "ASTAR"


current_mode = MODE_MANUAL
current_algo = MODE_BFS

#butoane
button_w, button_h = 160, 35
button_x = width - button_w - 10
manual_button = pygame.Rect(button_x, 40,  button_w, button_h)
bfs_button    = pygame.Rect(button_x, 85,  button_w, button_h)
dfs_button    = pygame.Rect(button_x, 130, button_w, button_h)
astar_button  = pygame.Rect(button_x, 175, button_w, button_h)


player_images = []
for i in range(1, 2 + 1):
    player_images.append(
        pygame.transform.scale(
            pygame.image.load(f"assets/pacman_photos/{i}.png"),
            (num2, num1)
        )
    )

PACMAN_START = (2, 2)
player_x, player_y = PACMAN_START
direction = None          # 0=right,1=left,2=up,3=down, None = stopped
next_direction = None     
counter = 0
score = 0
lives =3
life_image = pygame.transform.scale(
    pygame.image.load("assets/pacman_photos/1.png"),
    (num2, num1)
)
pacman_move_interval = 6  


power_up = False
power_up_duration = 8000  #
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


#algoritmi


def manhattan(p1, p2) -> int:
    x1, y1 = p1
    x2, y2 = p2
    return abs(x1 - x2) + abs(y1 - y2)


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


def ghost_neighbors(node, ghost):
    x, y = node
    neigh = []
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nx, ny = x + dx, y + dy
        if ghost_tile_walkable(nx, ny):
            neigh.append((nx, ny))
    return neigh


def nearest_walkable(target):
    tx, ty = target
    start = (tx, ty)
    if ghost_tile_walkable(tx, ty):
        return start

    queue = deque([start])
    visited = {start}

    while queue:
        x, y = queue.popleft()
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = x + dx, y + dy
            if (nx, ny) in visited:
                continue
            visited.add((nx, ny))
            if ghost_tile_walkable(nx, ny):
                return (nx, ny)
            queue.append((nx, ny))

    return start  



ghost_mode_sequence = [
   
    ("scatter", 1000),
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


#corners
RAW_SCATTER_TL = (2, 2)
RAW_SCATTER_TR = (27, 2)
RAW_SCATTER_BR = (27, 29)
RAW_SCATTER_BL = (2, 29)

scatter_tl = nearest_walkable(RAW_SCATTER_TL)
scatter_tr = nearest_walkable(RAW_SCATTER_TR)
scatter_br = nearest_walkable(RAW_SCATTER_BR)
scatter_bl = nearest_walkable(RAW_SCATTER_BL)

GHOST_HOME = (14, 15)


class Ghost:
    def __init__(self, name, start_tile, images, scatter_target):
        self.name = name
        self.start = start_tile
        self.x, self.y = start_tile
        self.images = images
        self.scatter_target = scatter_target
        self.state = "chase"   
        self.image = self.images["normal"]
        self.path = []         

    def reset(self):
        self.x, self.y = self.start
        self.state = "chase"
        self.image = self.images["normal"]
        self.path = []

    def draw(self):
        screen.blit(self.image, (self.x * num2, self.y * num1))

    def update_state(self):
       
        if self.state == "dead":
            self.image = self.images["dead"]
            if (self.x, self.y) == GHOST_HOME:
                self.state = "chase"
                self.image = self.images["normal"]
            return

        if power_up:
            self.state = "frightened"
            self.image = self.images["scared"]
        else:
            if self.state == "frightened":
                self.image = self.images["normal"]
            self.state = ghost_mode
            self.image = self.images["normal"]

    def get_chase_target(self, pac_pos, pac_dir, blinky_pos):
        px, py = pac_pos
        if self.name == "blinky":
            # direct chase
            return (px, py)

        elif self.name == "pinky":
            # 4 tiles ahead of Pac-Man
            dx = dy = 0
            if pac_dir == 0:
                dx = 4
            elif pac_dir == 1:
                dx = -4
            elif pac_dir == 2:
                dy = -4
            elif pac_dir == 3:
                dy = 4
            return (px + dx, py + dy)

        elif self.name == "inky":
            # depend on Blinky's position and Pac-Man's Location
            bx, by = blinky_pos
            dx = dy = 0
            if pac_dir == 0:
                dx = 2
            elif pac_dir == 1:
                dx = -2
            elif pac_dir == 2:
                dy = -2
            elif pac_dir == 3:
                dy = 2
            target_x = px + dx
            target_y = py + dy
            vector_x = target_x - bx
            vector_y = target_y - by
            return (bx + 2 * vector_x, by + 2 * vector_y)
        elif self.name == "clyde":
            # chase Pac-Man only when <= 8 tiles, else scatter corner
            gx, gy = self.x, self.y
            dist = manhattan((gx, gy), (px, py))
            if dist <= 8:
                return (px, py)
            else:
                return self.scatter_target

        

   
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
            self.x, self.y = random.choice(candidates)

   
    def move(self, pac_pos, pac_dir, blinky_pos):
        start = (self.x, self.y)

        if self.state == "frightened":
            self.frightened_move(pac_pos)
            return

        
        if self.state == "dead":
            target = GHOST_HOME
        elif self.state == "scatter":
           
            target = self.scatter_target
        else:  
            target = self.get_chase_target(pac_pos, pac_dir, blinky_pos)

        
        max_y = len(level) - 1
        max_x = len(level[0]) - 1
        tx = max(0, min(max_x, int(target[0])))
        ty = max(0, min(max_y, int(target[1])))
        target = (tx, ty)

        path = find_path(
            current_algo,
            start,
            target,
            lambda n: ghost_neighbors(n, self)
        )
        self.path = path 
        if len(path) > 1:
            nx, ny = path[1]
            self.x, self.y = nx, ny



blinky = Ghost("blinky", (13, 16), blinky_imgs, scatter_tr)  
pinky  = Ghost("pinky",  (13, 15), pinky_imgs,  scatter_tl) 
inky   = Ghost("inky",   (15, 15), inky_imgs,   scatter_br) 
clyde  = Ghost("clyde",  (14, 15), clyde_imgs,  scatter_bl)  
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
        if not power_up:
            power_up = True
            power_up_end_time = now + power_up_duration
        else:
            power_up_end_time += power_up_duration


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
    buttons = [
        ("MANUAL", manual_button, MODE_MANUAL),
        ("BFS",    bfs_button,    MODE_BFS),
        ("DFS",    dfs_button,    MODE_DFS),
        ("A*",     astar_button,  MODE_ASTAR),
    ]
    for name, rect, mode in buttons:
        selected = (
            (current_mode == mode) or
            (
                mode in (MODE_BFS, MODE_DFS, MODE_ASTAR)
                and current_mode != MODE_MANUAL
                and current_algo == mode
            )
        )
        color = (200, 200, 0) if selected else (40, 40, 40)
        pygame.draw.rect(screen, color, rect)
        pygame.draw.rect(screen, "white", rect, 2)
        text = font.render(name, True, "white")
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


def update_pacman():
    
    global player_x, player_y, direction

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
    global player_x, player_y, direction, power_up
    player_x, player_y = PACMAN_START
    direction = None
    power_up = False
    for g in ghosts:
        g.reset()


def handle_collisions():
    global score, run, lives

    for g in ghosts:
        if g.x == player_x and g.y == player_y:

            
            if g.state == "frightened":
                score += 200
                g.state = "dead"
                return

           
            if g.state in ("chase", "scatter"):
                score = max(0, score - 500)
                lives -= 1

                if lives <= 0:
                    run = False 
                else:
                    reset_positions()
                return


run = True
while run:
    timer.tick(fps)
    counter += 1
    screen.fill("black")

    if not power_up:
        update_ghost_global_mode()

    draw_board()
    draw_fruit()
   
    if counter % pacman_move_interval == 0:
        update_pacman()
        pac_pos = (player_x, player_y)

       
        for g in ghosts:
            g.update_state()
       
        ghost_move_counter += 1
        if ghost_move_counter >= ghost_move_interval:
            ghost_move_counter = 0
            for g in ghosts:
                g.move(pac_pos, direction if direction is not None else 0,(blinky.x, blinky.y))

    modify_board()
    update_power_up()
    update_fruit()
    draw_score()
    draw_lives()
    draw_power_up_text()
    draw_buttons()
    draw_paths()

    for g in ghosts:
        g.draw()
    draw_player()
    handle_collisions()

   
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False

        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            mx, my = event.pos
            if manual_button.collidepoint(mx, my):
                current_mode = MODE_MANUAL
            elif bfs_button.collidepoint(mx, my):
                current_mode = MODE_BFS
                current_algo = MODE_BFS
            elif dfs_button.collidepoint(mx, my):
                current_mode = MODE_DFS
                current_algo = MODE_DFS
            elif astar_button.collidepoint(mx, my):
                current_mode = MODE_ASTAR
                current_algo = MODE_ASTAR

        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_p:
                show_paths = not show_paths
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
