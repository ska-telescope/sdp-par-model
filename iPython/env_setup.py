'''
When running this code block, make sure that it finishes *before* running any of the subsequent blocks. 
The Cell nr is shown as "In [*]" (in the left-hand margin) when it is executing and updates to "In [1]" when done.
Only then should you run subsequent code block. 
Selecting Cell -> "Run All Below" from the iPy Notebook menu, you can run all of the later code blocks in a single command.
'''

from sympy import init_session
init_session(
    use_latex=True,
    use_unicode=True,
    pretty_print=True,
    quiet=True,
)
from sympy import Max, ceiling

def divide_elementwise(A,B):
    '''
    Does an element-wise division A/B between two sympy matrices A and B
    '''
    m, n = A.shape
    m2, n2 = B.shape
    C = A.copy()
    assert (m, n) == (m2, n2)
    for i in range(m):
        for j in range(n):
            C[i,j] = A[i,j] / B[i,j]
    return C

def ceiling_elementwise(A): 
    '''
    Does an element-wise division A/B between two sympy matrices A and B
    '''
    m, n = A.shape
    B = A.copy()
    for i in range(m):
        for j in range(n):
            B[i,j] = ceiling(A[i,j])
    return B

def max_elementwise(A, B): 
    '''
    Does an element-wise division A/B between two sympy matrices A and B
    '''
    m, n = A.shape
    m2, n2 = B.shape
    C = A.copy()
    assert (m, n) == (m2, n2)
    for i in range(m):
        for j in range(n):
            C[i,j] = Max(A[i,j], B[i,j])
    return C