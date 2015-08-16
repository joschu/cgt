#!/usr/bin/env python

import os
import subprocess
import unittest
import webbrowser


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--cov", action="store_true")
    args = parser.parse_args()

    filename = os.path.realpath(__file__)
    if args.cov:
        # Run this same script with coverage monitor
        if os.path.exists(".coverage"):
            os.remove(".coverage")
        subprocess.check_call("coverage run -a %s" % filename, shell=True)
        subprocess.check_call("coverage html", shell=True)
        webbrowser.open("htmlcov/index.html")
    else:
        # Run all tests in the directory of this script
        tests = unittest.TestLoader().discover(os.path.dirname(filename))
        unittest.TextTestRunner().run(tests)


if __name__ == '__main__':
    main()
