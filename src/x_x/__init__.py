import random


EYES = [
    "x",
    "X",
    "O",
    "o",
    "0",
    "U",
    "u",
    "@",
    "*",
    "^",
    "-",
]


def x_x():
    """Return a random face with an eye."""
    return random.choice(EYES) + "_" + random.choice(EYES)
