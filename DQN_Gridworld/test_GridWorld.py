# -*- coding: utf-8 -*-
"""
Created on Mon Jan  1 22:38:12 2024

@author: aunko
"""

from Gridworld import Gridworld

game = Gridworld(size = 4, mode = 'static')

game.display() # <- in Console ausführen!

game.makeMove('d') # <- move down
game.makeMove('l') # <- move left

game.reward() # <- in Consolte ausführen!

game.board.render_np() # <- in Console ausführen!

game.board.render_np().shape # <- in Console ausführen!  <= 4 x 4 x 4 Tensor! Jede Schicht repräsentiert jeweils Spieler, Ziel, Grube, Mauer!






