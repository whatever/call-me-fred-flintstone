import argparse
import os.path


from x_x import x_x


def splash(choice):
    """Return a splash screen"""

    fname = os.path.join(
        os.path.dirname(__file__),
        "splashes",
        f"{choice}.txt",
    )

    with open(fname, "r") as fi:
        return fi.read()


def main():
    """CLI stub"""

    parser = argparse.ArgumentParser()
    parser.add_argument("--splash", action="store_true", help="show splash screen")
    args = parser.parse_args()

    if args.splash:
        print(splash("sword"))
    else:
        print(x_x())
