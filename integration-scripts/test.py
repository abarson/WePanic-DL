import sys

if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "pass":
            sys.exit(0) # pass code
        else:
            sys.exit(1) 
