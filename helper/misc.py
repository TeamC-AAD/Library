""" Miscallaneous helper functions"""

def binSearch(wheel, num):
    """Standard binary search
    Args:
        wheel(list): An ordered list of floats/ints
        num(int/float): Element to be searched
    
    Returns:
        int: value
    """
    mid = len(wheel)//2
    low, high, answer = wheel[mid]
    if low<=num<=high:
        return answer
    elif high < num:
        return binSearch(wheel[mid+1:], num)
    else:
        return binSearch(wheel[:mid], num)


def makeWheel(population):
    """Helper for the RWS in creating a proportional
    distribution among elements in a given array

    Args:
        population(list): List of int/float
    
    Returns:
        list: The generated wheel
    """
    wheel = []
    total = sum([p for p in population])
    top = 0
    
    for i in range(len(population)):
        p = population[i]
        f = p/total
        wheel.append((top, top+f, i))
        top += f
    return wheel