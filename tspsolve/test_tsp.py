import time


def test():
    x = 1
    while x < 5:
        x += 2
        print("Before yield")
        yield x
        time.sleep(1)
        print("After")

    print("Done test")


