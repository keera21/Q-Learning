import pygame
import numpy as np

GRID_SIZE = 10
CELL_SIZE = 60
WINDOW_SIZE = GRID_SIZE * CELL_SIZE

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (200, 200, 200)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)

class GridWorld:
    def __init__(self):
        self.grid_size = GRID_SIZE
        self.state = (0, 0)
        self.goal = (9, 9)
        self.obstacles = set([
            (3, 3), (3, 4), (4, 3), (5, 5), (6, 5)
        ])
    
    def reset(self):
        self.state = (0, 0)
        return self.state

    def step(self, action):
        x, y = self.state
        if action == 0: y -= 1  # up
        elif action == 1: y += 1  # down
        elif action == 2: x -= 1  # left
        elif action == 3: x += 1  # right

        # Boundaries
        x = max(0, min(x, self.grid_size - 1))
        y = max(0, min(y, self.grid_size - 1))

        new_state = (x, y)
        reward = -1
        done = False

        if new_state in self.obstacles:
            reward = -10
            new_state = self.state  # stay in same place

        elif new_state == self.goal:
            reward = 100
            done = True

        self.state = new_state
        return new_state, reward, done

    def render(self, screen):
        screen.fill(WHITE)
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                pygame.draw.rect(screen, GRAY, rect, 1)

                if (x, y) in self.obstacles:
                    pygame.draw.rect(screen, BLACK, rect)

        # Draw goal
        goal_rect = pygame.Rect(self.goal[0] * CELL_SIZE, self.goal[1] * CELL_SIZE, CELL_SIZE, CELL_SIZE)
        pygame.draw.rect(screen, GREEN, goal_rect)

        # Draw agent
        agent_rect = pygame.Rect(self.state[0] * CELL_SIZE, self.state[1] * CELL_SIZE, CELL_SIZE, CELL_SIZE)
        pygame.draw.rect(screen, BLUE, agent_rect)

        pygame.display.flip()
