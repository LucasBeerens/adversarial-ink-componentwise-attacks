def matrix(a):
    """
    Convert a 2D matrix to LaTeX representation.

    Args:
    - a: The input 2D matrix.

    Returns:
    - None (prints the LaTeX representation of the matrix).
    """
    text = r'\begin{array}{*{'
    text += str(len(a[0]))
    text += r'}c}'
    text += '\n'

    for x in range(len(a)):
        for y in range(len(a[x])):
            text += str(a[x][y])
            text += r' & '
        text = text[:-2]
        text += r'\\'
        text += '\n'
    text += r'\end{array}'

    print(text)
