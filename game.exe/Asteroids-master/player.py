import pygame
from shot import Shot
from constants import PLAYER_SHOOT_COOLDOWN_SECONDS
from constants import PLAYER_SHOOT_SPEED
from constants import SHOT_RADIUS
from constants import PLAYER_RADIUS
from circleshape import CircleShape
from constants import PLAYER_TURN_SPEED
from constants import PLAYER_SPEED
class Player(CircleShape):
  def __init__(self,x,y,):
    super().__init__(x,y,PLAYER_RADIUS)
    self.rotation = 0
   
    self.bullet_time = 0
  def rotate(self, dt):
    self.rotation += PLAYER_TURN_SPEED * dt
  def update(self, dt):
    keys = pygame.key.get_pressed()
    if keys[pygame.K_a]:
       self.rotate(-dt)
    if keys[pygame.K_d]:
       self.rotate(dt)
    if keys[pygame.K_w]:
       self.move(dt)
    if keys[pygame.K_s]:
       self.move(-dt)
    if keys[pygame.K_SPACE] and self.bullet_time <= 0:
       self.shoot()
       self.bullet_time = PLAYER_SHOOT_COOLDOWN_SECONDS
    self.bullet_time -= dt 
    if self.bullet_time < 0: 
       self.bullet = 0
       
  def move(self, dt):
     forward = pygame.Vector2(0, 1).rotate(self.rotation)
     self.position += forward * PLAYER_SPEED * dt

 # in the player class
  def triangle(self):
    forward = pygame.Vector2(0, 1).rotate(self.rotation)
    right = pygame.Vector2(0, 1).rotate(self.rotation + 90) * self.radius / 1.5
    a = self.position + forward * self.radius
    b = self.position - forward * self.radius - right
    c = self.position - forward * self.radius + right
    return [a, b, c]
  def draw(self, screen):
    pygame.draw.polygon(screen, "white", self.triangle(), 2)
  def shoot(self):
    x, y = self.position.x, self.position.y
    shot = Shot(x,y,SHOT_RADIUS)
    direction = pygame.Vector2(0, 1).rotate(self.rotation)
    shot.velocity = direction * PLAYER_SHOOT_SPEED
    
