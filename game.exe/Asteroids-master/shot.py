from constants import WHITE
import pygame
from circleshape import CircleShape
class Shot(CircleShape):
 def __init__(self,x ,y ,radius):
    super().__init__(x,y,radius)
    for group in getattr(self, "containers", []):
        group.add(self)

 def draw(self, screen):
    pygame.draw.circle(screen, WHITE, (self.position.x, self.position.y), self.radius, 2)
 def update(self,dt):
   self.position += self.velocity * dt
