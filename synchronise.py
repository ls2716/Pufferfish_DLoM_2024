"""Take a file or folder as an argument and send it using scp to a destination
"""

import os
import sys


def main():
    if len(sys.argv) < 2:
        print("Usage: python synchronise.py <source>")
        sys.exit(1)

    source = sys.argv[1]
    destination = "s1640204@mathsgpu2.maths.ed.ac.uk:~/Pufferfish_own/"

    if os.path.isfile(source):
        os.system(f"scp {source} {destination}")
    elif os.path.isdir(source):
        os.system(f"scp -r {source} {destination}/{source}")
    else:
        print("The source does not exist.")

    print("Synchronisation completed.")


if __name__ == "__main__":
    main()
