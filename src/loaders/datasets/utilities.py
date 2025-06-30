
MAX_LEN = 512  # Maximum length for tokenization

def process(elem):
    if elem in ['Negative','unknown']:
        return 0
    else:
        return 1

def process2(elem):
    if elem == 'Positive':
        return 1
    else:
        return 0 

def process3(elem):
    elem = float(elem)
    if elem<=2:
        return 0
    elif elem>2 and elem<=3:
        return 1
    else:
        return 2