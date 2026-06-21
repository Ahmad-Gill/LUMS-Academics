import pygame
import os
import sys

# Initialize Pygame
pygame.init()

# Set up some constants
WIDTH, HEIGHT = 800, 400
GROUND_Y = HEIGHT - 50
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)

# Set up the display
SCREEN = pygame.display.set_mode((WIDTH, HEIGHT))

# Set up the font
FONT = pygame.font.Font(None, 36)

# Set up the clock
CLOCK = pygame.time.Clock()

class Dinosaur:
    def __init__(self):
        self.x = 50
        self.y = GROUND_Y - 50
        self.width = 50
        self.height = 50
        self.velocity = 0
        self.is_jumping = False
        self.is_ducking = False

    def draw(self):
        if self.is_ducking:
            # Draw ducking dinosaur
            pygame.draw.ellipse(SCREEN, GREEN, (self.x, self.y, self.width, self.height // 2))
            pygame.draw.ellipse(SCREEN, BLACK, (self.x + self.width // 2 - 5, self.y + self.height // 4, 10, 10))  # Eye
            pygame.draw.line(SCREEN, BLACK, (self.x + self.width // 2 - 5, self.y + self.height // 2), (self.x + self.width // 2 + 5, self.y + self.height // 2), 2)  # Mouth
            pygame.draw.line(SCREEN, GREEN, (self.x + self.width, self.y + self.height // 2), (self.x + self.width + 20, self.y + self.height // 2), 2)  # Tail
            pygame.draw.line(SCREEN, GREEN, (self.x, self.y + self.height // 2), (self.x - 20, self.y + self.height // 2), 2)  # Arm
        else:
            # Draw standing dinosaur
            pygame.draw.ellipse(SCREEN, GREEN, (self.x, self.y, self.width, self.height))
            pygame.draw.ellipse(SCREEN, BLACK, (self.x + self.width // 2 - 5, self.y + self.height // 4, 10, 10))  # Eye
            pygame.draw.line(SCREEN, BLACK, (self.x + self.width // 2 - 5, self.y + self.height // 2), (self.x + self.width // 2 + 5, self.y + self.height // 2), 2)  # Mouth
            pygame.draw.line(SCREEN, GREEN, (self.x + self.width, self.y + self.height // 2), (self.x + self.width + 20, self.y + self.height // 2), 2)  # Tail
            pygame.draw.line(SCREEN, GREEN, (self.x, self.y + self.height // 2), (self.x - 20, self.y + self.height // 2), 2)  # Arm
            pygame.draw.line(SCREEN, GREEN, (self.x, self.y + self.height), (self.x - 10, self.y + self.height + 20), 2)  # Leg
            pygame.draw.line(SCREEN, GREEN, (self.x + self.width, self.y + self.height), (self.x + self.width + 10, self.y + self.height + 20), 2)  # Leg

    def update(self):
        if self.is_jumping:
            self.velocity += 1
            self.y += self.velocity
            if self.y > GROUND_Y - 50:
                self.y = GROUND_Y - 50
                self.is_jumping = False
                self.velocity = 0

class Cactus:
    def __init__(self, x):
        self.x = x
        self.y = GROUND_Y - 50
        self.width = 20
        self.height = 50

    def draw(self):
        pygame.draw.rect(SCREEN, RED, (self.x, self.y, self.width, self.height))
        pygame.draw.line(SCREEN, RED, (self.x, self.y), (self.x + self.width // 2, self.y - 20), 2)  # Branch
        pygame.draw.line(SCREEN, RED, (self.x + self.width, self.y), (self.x + self.width // 2, self.y - 20), 2)  # Branch

class Pterodactyl:
    def __init__(self, x):
        self.x = x
        self.y = GROUND_Y - 120
        self.width = 50
        self.height = 50

    def draw(self):
        pygame.draw.ellipse(SCREEN, RED, (self.x, self.y, self.width, self.height))  # Body
        pygame.draw.ellipse(SCREEN, BLACK, (self.x + self.width // 2 - 5, self.y + self.height // 4, 10, 10))  # Eye
        pygame.draw.line(SCREEN, BLACK, (self.x + self.width // 2 - 5, self.y + self.height // 2), (self.x + self.width // 2 + 5, self.y + self.height // 2), 2)  # Beak
        pygame.draw.line(SCREEN, RED, (self.x, self.y + self.height // 2), (self.x - 20, self.y + self.height // 2), 2)  # Wing
        pygame.draw.line(SCREEN, RED, (self.x + self.width, self.y + self.height // 2), (self.x + self.width + 20, self.y + self.height // 2), 2)  # Wing

def main():
    try:
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        BASE_DIR = os.getcwd()

    HIGH_SCORE_FILE = os.path.join(BASE_DIR, "high_score.txt")

    try:
        with open(HIGH_SCORE_FILE, 'r') as f:
            high_score = int(f.read())
    except (FileNotFoundError, ValueError):
        high_score = 0

    dinosaur = Dinosaur()
    obstacles = []
    score = 0
    game_over = False
    obstacle_type = 'cactus'

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE and not dinosaur.is_jumping:
                    dinosaur.is_jumping = True
                    dinosaur.velocity = -20
                elif event.key == pygame.K_DOWN:
                    dinosaur.is_ducking = True
                elif event.key == pygame.K_r and game_over:
                    game_over = False
                    score = 0
                    obstacles = []
                    dinosaur = Dinosaur()
                    obstacle_type = 'cactus'
                elif event.key == pygame.K_q and game_over:
                    pygame.quit()
                    sys.exit()
            elif event.type == pygame.KEYUP:
                if event.key == pygame.K_DOWN:
                    dinosaur.is_ducking = False

        SCREEN.fill(WHITE)

        if not game_over:
            dinosaur.update()
            dinosaur.draw()

            if obstacle_type == 'cactus':
                if not obstacles or obstacles[-1].x < WIDTH - 550:
                    obstacles.append(Cactus(WIDTH))
                    obstacle_type = 'pterodactyl'
            else:
                if not obstacles or obstacles[-1].x < WIDTH - 550:
                    obstacles.append(Pterodactyl(WIDTH))
                    obstacle_type = 'cactus'

            for obstacle in obstacles:
                if isinstance(obstacle, Cactus):
                    obstacle.x -= 5
                    obstacle.draw()
                    if obstacle.x < -obstacle.width:
                        obstacles.remove(obstacle)
                    elif (obstacle.x < dinosaur.x + dinosaur.width and
                          obstacle.x + obstacle.width > dinosaur.x and
                          obstacle.y < dinosaur.y + dinosaur.height and
                          obstacle.y + obstacle.height > dinosaur.y):
                        game_over = True
                elif isinstance(obstacle, Pterodactyl):
                    obstacle.x -= 5
                    obstacle.draw()
                    if obstacle.x < -obstacle.width:
                        obstacles.remove(obstacle)
                    elif (obstacle.x < dinosaur.x + dinosaur.width and
                          obstacle.x + obstacle.width > dinosaur.x and
                          obstacle.y < dinosaur.y + dinosaur.height and
                          obstacle.y + obstacle.height > dinosaur.y):
                        game_over = True

            score += 1
            high_score_text = FONT.render(f'High Score: {high_score}', True, BLACK)
            score_text = FONT.render(f'Score: {score}', True, BLACK)
            SCREEN.blit(high_score_text, (10, 10))
            SCREEN.blit(score_text, (10, 50))

            if score > high_score:
                high_score = score
                with open(HIGH_SCORE_FILE, 'w') as f:
                    f.write(str(high_score))

        else:
            game_over_text = FONT.render('Game Over', True, BLACK)
            restart_text = FONT.render('Press R to restart', True, BLACK)
            quit_text = FONT.render('Press Q to quit', True, BLACK)
            SCREEN.blit(game_over_text, (WIDTH // 2 - 50, HEIGHT // 2 - 50))
            SCREEN.blit(restart_text, (WIDTH // 2 - 100, HEIGHT // 2))
            SCREEN.blit(quit_text, (WIDTH // 2 - 50, HEIGHT // 2 + 50))
            high_score_text = FONT.render(f'High Score: {high_score}', True, BLACK)
            score_text = FONT.render(f'Final Score: {score}', True, BLACK)
            SCREEN.blit(high_score_text, (10, 10))
            SCREEN.blit(score_text, (10, 50))

        pygame.draw.line(SCREEN, BLACK, (0, GROUND_Y), (WIDTH, GROUND_Y), 2)

        pygame.display.update()
        CLOCK.tick(60)

if __name__ == '__main__':
    main()