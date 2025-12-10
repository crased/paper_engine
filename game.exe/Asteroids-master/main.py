from shot import Shot
from asteroidfield import AsteroidField
from asteroid import Asteroid
import sys
import pygame
from logger import log_state
from logger import log_event
from player import Player
from constants import *
def main():
    pygame.init()
    print("Starting Asteroids!")
    print("Screen width:", SCREEN_WIDTH)
    print("Screen height:", SCREEN_HEIGHT)
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    dt = 0
    # player containers and sprites
    updatable = pygame.sprite.Group()
    drawable = pygame.sprite.Group()
    Player.containers = (updatable, drawable)
    # shot container and sprites 
    shots = pygame.sprite.Group()
    Shot.containers = (shots, updatable, drawable) 
    # Asteroid container and sprite 
    asteroids = pygame.sprite.Group()
    Asteroid.containers = (asteroids, updatable, drawable)
    # astroidfield container and sprite
    AsteroidField.containers = (updatable,)
    field = AsteroidField()
    player = Player(SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2)
    while True:
       for event in pygame.event.get():
     	    if event.type == pygame.QUIT:
               return
       dt = clock.tick(60) / 1000
       updatable.update(dt)
       for shot in shots:
           for asteroid in asteroids:
               if player.collides_with(asteroid) == True:
               	  log_event("player_hit")
                  print("Game over!")
                  return sys.exit(" you lost")
               if shot.collides_with(asteroid) == True:
                  log_event("asteroid_shot")
                  shot.kill
                  asteroid.split()
       log_state()
       screen.fill("black")
       for sprite in drawable:
          sprite.draw(screen)
       pygame.display.flip()
if __name__ == "__main__":
    main()




