"""Take a file or folder as an argument and get it using scp to a destination
"""

import os
import sys


def main():
    if len(sys.argv) < 3:
        print("Usage: python synchronise.py <file> <destination>")
        sys.exit(1)

    file = sys.argv[1]
    destination = sys.argv[2]
    source = "s1640204@mathsgpu2.maths.ed.ac.uk:~/Pufferfish_own/"+file

    os.system(f"scp -r {source} {destination}")

    print("Synchronisation completed.")


if __name__ == "__main__":
    main()
