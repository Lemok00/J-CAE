import sys


def info(*msg):
    s = "[INFO] " + " ".join(list(map(str, msg)))
    print(s)



def debug(*msg):
    s = "[DEBUG] " + " ".join(list(map(str, msg)))
    print(s)
    sys.stdout.flush()


def error(*msg):
    s = "[ERROR] " + " ".join(list(map(str, msg)))
    print(s)
