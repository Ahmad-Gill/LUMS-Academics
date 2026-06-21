import pygame
import random
import sys

# Initialize Pygame
pygame.init()

# Define constants
WIDTH, HEIGHT = 800, 600
BASE_SPEED = 5
MIN_OBSTACLE_DISTANCE = 320
GROUND_Y = HEIGHT - 50
JUMP_POWER = -18
GRAVITY = 0.9

# Define colors
DAY_COLOR = (255, 255, 255)
NIGHT_COLOR = (0, 0, 0)
BLUE = (135, 206, 235)
DARK_BLUE = (25, 25, 112)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
RED = (255, 0, 0)

# Define obstacle classes
class Obstacle(pygame.Rect):
    def __init__(self, x, y, kind):
        super().__init__(x, y, 50, 50)
        self.kind = kind

    def update(self, speed):
        self.x -= speed

    def draw(self, screen):
        if self.kind == 'ground':
            pygame.draw.rect(screen, RED, self)
        elif self.kind == 'flying':
            pygame.draw.ellipse(screen, RED, self)

# Define Dino class
class Dino(pygame.Rect):
    def __init__(self):
        super().__init__(100, GROUND_Y - 50, 50, 50)
        self.velocity = 0
        self.on_ground = True

    def update(self):
        if not self.on_ground:
            self.velocity += GRAVITY
            self.y += self.velocity
            if self.bottom >= GROUND_Y:
                self.bottom = GROUND_Y
                self.on_ground = True
                self.velocity = 0

    def jump(self):
        if self.on_ground:
            self.velocity = JUMP_POWER
            self.on_ground = False

    def duck(self):
        if self.on_ground:
            self.height = 25
        else:
            self.height = 50

# Define Game class
class Game:
    def __init__(self):
        self.dino = Dino()
        self.obstacles = [Obstacle(WIDTH + 250, GROUND_Y - 50, 'ground'), Obstacle(WIDTH + 650, GROUND_Y - 50, 'ground')]
        self.score = 0
        self.speed = BASE_SPEED
        self.night_mode = False
        self.first_flying_spawned = False
        self.high_score = self.load_high_score()

    def update(self):
        # Update dino position
        self.dino.update()

        # Update obstacle positions
        for obstacle in self.obstacles:
            obstacle.update(self.speed)
            if obstacle.x < -obstacle.width:
                self.obstacles.remove(obstacle)

        # Check for collisions
        for obstacle in self.obstacles:
            if self.dino.colliderect(obstacle):
                # Game over
                return False

        # Spawn new obstacles
        rightmost_x = max(obstacle.x for obstacle in self.obstacles) if self.obstacles else -1000
        if rightmost_x < WIDTH + 120 or not self.obstacles:
            if random.random() < 0.4 and self.score >= 30:
                self.obstacles.append(Obstacle(WIDTH + random.randint(250, 450), random.randint(0, HEIGHT // 2), 'flying'))
            else:
                self.obstacles.append(Obstacle(WIDTH + random.randint(250, 450), GROUND_Y - 50, 'ground'))

        # Update score and speed
        self.score += 1
        if self.score % 100 == 0:
            self.speed += 1

        return True

    def render(self, screen):
        # Render background
        if self.night_mode:
            pygame.draw.rect(screen, DARK_BLUE, (0, 0, WIDTH, HEIGHT))
        else:
            pygame.draw.rect(screen, BLUE, (0, 0, WIDTH, HEIGHT))

        # Render dino
        pygame.draw.rect(screen, GREEN, self.dino)
        pygame.draw.ellipse(screen, BLACK, (self.dino.x + 10, self.dino.y + 10, 10, 10))
        pygame.draw.polygon(screen, GREEN, [(self.dino.x + 10, self.dino.y + 30), (self.dino.x + 40, self.dino.y + 30), (self.dino.x + 25, self.dino.y + 50)])
        pygame.draw.line(screen, BLACK, (self.dino.x + 20, self.dino.y + 20), (self.dino.x + 30, self.dino.y + 20), 2)

        # Render obstacles
        for obstacle in self.obstacles:
            obstacle.draw(screen)

        # Render score and high score
        font = pygame.font.Font(None, 36)
        text = font.render('Score: ' + str(self.score), True, WHITE)
        screen.blit(text, (10, 10))
        text = font.render('High Score: ' + str(self.high_score), True, WHITE)
        screen.blit(text, (10, 50))

    def load_high_score(self):
        try:
            with open('high_score.txt', 'r') as file:
                content = file.read()
                if content == '':
                    return 0
                return int(content)
        except FileNotFoundError:
            return 0

    def save_high_score(self):
        with open('high_score.txt', 'w') as file:
            file.write(str(self.high_score))

# Initialize game
def main():
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()
    game = Game()
    game_over = False

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE and game_over:
                    game = Game()
                    game_over = False
                elif event.key == pygame.K_UP:
                    game.dino.jump()
                elif event.key == pygame.K_DOWN:
                    game.dino.duck()

        if not game_over:
            game_over = not game.update()
            if game.score > game.high_score:
                game.high_score = game.score
                game.save_high_score()

        game.night_mode = (game.score // 200) % 2 == 1

        screen.fill((0, 0, 0))
        game.render(screen)
        if game_over:
            font = pygame.font.Font(None, 36)
            text = font.render('Game Over - Press SPACE to Restart', True, WHITE)
            screen.blit(text, (WIDTH // 2 - 150, HEIGHT // 2))
        pygame.display.flip()
        clock.tick(60)

if __name__ == '__main__':
    main()