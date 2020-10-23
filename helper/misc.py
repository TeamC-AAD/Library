""" Miscallaneous helper functions"""

def binSearch(wheel, num):
    mid = len(wheel)//2
    low, high, answer = wheel[mid]
    if low<=num<=high:
        return answer
    elif high < num:
        return binSearch(wheel[mid+1:], num)
    else:
        return binSearch(wheel[:mid], num)


def makeWheel(population):
    wheel = []
    total = sum([p for p in population])
    top = 0
    
    for i in range(len(population)):
        p = population[i]
        f = p/total
        wheel.append((top, top+f, i))
        top += f
    return wheel