from collections import defaultdict, namedtuple
from enum import Enum
from typing import Tuple, List
import random
from IPython.display import clear_output
Point = namedtuple('Point', ['x', 'y'])
class Direction(Enum):
  NORTH = "⬆"
  EAST = "⮕"
  SOUTH = "⬇"
  WEST = "⬅"
  def values(self):
    return [v for v in self]
class SimpleGridWorld(object):
  def __init__(self, width: int = 5, height: int = 5, debug: bool = False):
    self.width = width
    self.height = height
    self.debug = debug
    self.action_space = [d for d in Direction]
    self.reset()
class SimpleGridWorld(SimpleGridWorld):
  def reset(self):
    self.cur_pos = Point(x=0, y=(self.height - 1))
    self.goal = Point(x=(self.width - 1), y=0)

    if self.debug:
      print(self)
    return self.cur_pos, 0, False
class SimpleGridWorld(SimpleGridWorld):
  def __repr__(self):
    res = ""
    for y in reversed(range(self.height)):
      for x in range(self.width):
        if self.goal.x == x and self.goal.y == y:
          if self.cur_pos.x == x and self.cur_pos.y == y:
            res += "@"
          else:
            res += "o"
          continue
        if self.cur_pos.x == x and self.cur_pos.y == y:
          res += "x"
        else:
          res += "_"
      res += "\n"
    return res
s = SimpleGridWorld(debug=True)
print(" The visualisation of the environment state.\n")
s.step(Direction.SOUTH)
print(s.step(Direction.SOUTH), "⬅ This displays the state and reward from the environment 𝐀𝐅𝐓𝐄𝐑 moving.\n")
s.step(Direction.SOUTH)
s.step(Direction.SOUTH)
s.step(Direction.EAST)
s.step(Direction.EAST)
s.step(Direction.EAST)
s.step(Direction.EAST)
