import pygame
import sys
import random

# Initialize Pygame
pygame.init()

# Set up some constants
WIDTH, HEIGHT = 800, 400
BASE_SPEED = 5
MIN_OBSTACLE_DISTANCE = 320
SPAWN_TIMER = 1300
JUMP_POWER = -18
GRAVITY = 0.9
GROUND_Y = HEIGHT - 50

# Set up some variables
score = 0
speed = BASE_SPEED
night_mode = False
game_over = False
first_flying_spawned = False
high_score = 0

# Set up the display
screen = pygame.display.set_mode((WIDTH, HEIGHT))

# Set up the font
font = pygame.font.Font(None, 36)

# Set up the clock
clock = pygame.time.Clock()

# Set up the Dino class
class Dino:
    def __init__(self):
        self.rect = pygame.Rect(50, HEIGHT // 2, 50, 50)
        self.velocity = 0
        self.on_ground = True

    def update(self):
        if not self.on_ground:
            self.velocity += GRAVITY
            self.rect.y += self.velocity
            if self.rect.bottom >= GROUND_Y:
                self.rect.bottom = GROUND_Y
                self.on_ground = True
                self.velocity = 0

    def jump(self):
        if self.on_ground:
            self.velocity = JUMP_POWER
            self.on_ground = False

    def duck(self):
        self.rect.height = 20
        self.rect.y = GROUND_Y - 20

    def stand_up(self):
        self.rect.height = 50
        self.rect.y = GROUND_Y - 50

# Set up the GroundObstacle class
class GroundObstacle:
    def __init__(self, x):
        self.rect = pygame.Rect(x, GROUND_Y - 20, 20, 20)
        self.kind = "ground"

    def update(self):
        self.rect.x -= speed

# Set up the FlyingObstacle class
class FlyingObstacle:
    def __init__(self, x):
        self.rect = pygame.Rect(x, random.randint(0, HEIGHT // 2), 20, 20)
        self.kind = "flying"

    def update(self):
        self.rect.x -= speed

# Set up the obstacles list
obstacles = [GroundObstacle(WIDTH + 250), GroundObstacle(WIDTH + 650)]

# Set up the dino object
dino = Dino()

try:
    with open("high_score.txt", "r") as file:
        high_score = int(file.read())
except FileNotFoundError:
    pass

# Game loop
while True:
    # Event handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE and not game_over:
                dino.jump()
            elif event.key == pygame.K_DOWN and not game_over:
                dino.duck()
            elif event.key == pygame.K_UP and not game_over:
                dino.stand_up()
            elif event.key == pygame.K_SPACE and game_over:
                game_over = False
                score = 0
                speed = BASE_SPEED
                night_mode = False
                first_flying_spawned = False
                obstacles = [GroundObstacle(WIDTH + 250), GroundObstacle(WIDTH + 650)]
                dino = Dino()

    # Game logic
    if not game_over:
        # Update the dino
        dino.update()

        # Update the obstacles
        for obstacle in obstacles:
            obstacle.update()
            if obstacle.rect.right < 0:
                obstacles.remove(obstacle)

        # Spawn new obstacles
        if obstacles and obstacles[-1].rect.right < WIDTH + 120:
            if random.random() < 0.4 and score >= 30 and not first_flying_spawned:
                obstacles.append(FlyingObstacle(WIDTH + random.randint(250, 450)))
                first_flying_spawned = True
            elif random.random() < 0.4 and score >= 30:
                if random.random() < 0.4:
                    obstacles.append(FlyingObstacle(WIDTH + random.randint(250, 450)))
                else:
                    obstacles.append(GroundObstacle(WIDTH + random.randint(250, 450)))
            else:
                obstacles.append(GroundObstacle(WIDTH + random.randint(250, 450)))
        elif not obstacles:
            obstacles.append(GroundObstacle(WIDTH + random.randint(250, 450)))

        # Check for collisions
        for obstacle in obstacles:
            if dino.rect.colliderect(obstacle.rect):
                game_over = True

        # Update the score
        score += 1
        speed = BASE_SPEED + score // 250

        # Update the night mode
        night_mode = (score // 100) % 2 == 1

        # Save high score
        if score > high_score:
            high_score = score
            with open("high_score.txt", "w") as file:
                file.write(str(high_score))

    # Draw everything
    if night_mode:
        screen.fill((0, 0, 0))
    else:
        screen.fill((255, 255, 255))

    # Draw ground
    pygame.draw.rect(screen, (139, 69, 19), (0, GROUND_Y, WIDTH, 50))

    # Draw dino
    pygame.draw.rect(screen, (0, 255, 0), dino.rect)

    # Draw obstacles
    for obstacle in obstacles:
        if obstacle.kind == "ground":
            pygame.draw.rect(screen, (0, 255, 0), obstacle.rect)
        else:
            pygame.draw.rect(screen, (255, 0, 0), obstacle.rect)

    # Draw score and high score
    text = font.render(f"Score: {score}", True, (0, 0, 0) if not night_mode else (255, 255, 255))
    screen.blit(text, (10, 10))
    text = font.render(f"High Score: {high_score}", True, (0, 0, 0) if not night_mode else (255, 255, 255))
    screen.blit(text, (10, 50))

    # Draw game over text
    if game_over:
        text = font.render("Game Over - Press SPACE to Restart", True, (0, 0, 0) if not night_mode else (255, 255, 255))
        screen.blit(text, (WIDTH // 2 - 150, HEIGHT // 2))

    # Update the display
    pygame.display.flip()

    # Cap the frame rate
    clock.tick(60)

if __name__ == '__main__':
    main()