from constants import ASTEROID_MIN_RADIUS
from constants import WHITE
import random
import pygame
from logger import log_event
from circleshape import CircleShape
class Asteroid(CircleShape):
 def __init__(self, x, y, radius):
     super().__init__(x,  y, radius)

 def draw(self, screen):
    pygame.draw.circle(screen, WHITE, (self.position), self.radius, 2)
 def update(self,dt):
   self.position += self.velocity * dt
 def split(self):
  self.kill()
  if self.radius <= ASTEROID_MIN_RADIUS:
      return
  else:
     log_event("asteroid_split")
     random_angle =  random.uniform(20.0,50.0)
     prime_vel = self.velocity.rotate(random_angle)
     alternate_vel =  self.velocity.rotate(-random_angle)
     new_radius =  self.radius - ASTEROID_MIN_RADIUS
     asteroid1 = Asteroid(self.position.x,self.position.y,new_radius)
     asteroid2 = Asteroid(self.position.x,self.position.y,new_radius)
     asteroid1.velocity = prime_vel *1.2
     asteroid2.velocity = alternate_vel *1.2
     return asteroid1, asteroid2
