import pygame
from board import boards
import math

pygame.init()

width = 900
height = 950
screen = pygame.display.set_mode([width, height])
timer = pygame.time.Clock()
fps = 60
font = pygame.font.Font('freesansbold.ttf', 20)
level = boards
PI = math.pi
player_images = []
turns_allowed = [False, False, False, False]
num1 = ((height - 50) // 32)
num2 = (width // 30)

for i in range(1,3):
    player_images.append(pygame.transform.scale(pygame.image.load(f'assets/pacman_photos/{i}.png'), (num1, num2)))

player_x = 2
player_y = 2
direction = 0
counter = 0
center_x = player_x 
center_y = player_y + 24
#turns_allowed = check_position(center_x, center_y):


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
    if(level[player_y][player_x] == 1 or level[player_y][player_x] == 2):
        level[player_y][player_x] =0
def reset_game():
    player_x = 2
    player_y = 2
    level = boards 



def draw_player():
    
    if direction == 0:
        screen.blit(player_images[(counter//20)%2], (num2 * player_x , num1*player_y ))
    elif direction == 1:
        screen.blit(pygame.transform.flip(player_images[(counter//20)%2],True,False), (num2 * player_x , num1*player_y ))
    elif direction == 2:
        screen.blit(pygame.transform.rotate(player_images[(counter//20)%2],90), (num2* player_x , num1*player_y ))
    elif direction == 3:
        screen.blit(pygame.transform.rotate(player_images[(counter//20)%2],270), (num2 * player_x, num1*player_y ))
    
run = True
while run:
    timer.tick(fps)
    counter = counter+1
    screen.fill('black')
    draw_board()
    draw_player()
    modify_board()
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RIGHT:
                direction = 0
                if player_x != 29:
                    if(level[player_y][player_x+1] not in [3,4,5,6,7,8,9]):
                        player_x = player_x + 1
                else:
                    player_x = 0
            if event.key == pygame.K_LEFT:
                direction = 1
                if(level[player_y][player_x-1] not in [3,4,5,6,7,8,9]):
                    if player_x != 0:
                        player_x = player_x -1
                    else:
                        player_x = 29
            if event.key == pygame.K_UP:
                direction = 2
                if(level[player_y-1][player_x] not in [3,4,5,6,7,8,9]):
                    player_y = player_y - 1
            if event.key == pygame.K_DOWN:
                direction = 3
                if(level[player_y+1][player_x] not in [3,4,5,6,7,8,9]):
                    player_y = player_y+1
            if event.key == pygame.K_BACKSLASH:
                reset_game()
                draw_board()
                draw_player()
    
    pygame.display.flip()

pygame.quit()