import numpy as np

def lvd(s, t):
    # LVD for transforming s into t
    # for all i and j, d[i,j] will hold the Levenshtein distance between
    # the first i characters of s and the first j characters of t
    # note that d has (m+1)*(n+1) values
    m, n = len(s), len(t)

    # L[i,j]: min no. edits transforming s[:i] to t[:j]
    L = np.zeros((m+1, n+1), dtype=int)
    L[0,1:] = np.arange(1,n+1) # insertions
    L[1:,0] = np.arange(1,m+1) # deletions

    # E[i,j]: last edit operator used in the min transformation 
    E = np.zeros((m+1, n+1), dtype=int) # 0,1,2,3 = no-op, insertion, deletion, substitution
    E[0,1:] = 1
    E[1:,0] = 2
    
    for j in range(n):
        for i in range(m):
            L[i+1,j+1] = min(
                L[i+1, j] + 1, # insertion
                L[i, j+1] + 1, # deletion
                L[i, j] + (s[i] != t[j]), # substitution
            )
            if s[i] == t[j]:
                E[i+1,j+1] = [
                    L[i, j], # no-op
                    L[i+1, j] + 1, # insertion
                    L[i, j+1] + 1, # deletion
                ].index(L[i+1,j+1])
            else:
                E[i+1,j+1] = [
                    L[i+1, j] + 1, # insertion
                    L[i, j+1] + 1, # deletion
                    L[i, j] + 1, # substitution
                ].index(L[i+1,j+1]) + 1

    # extract transformation
    i, j = m, n
    edits = []
    while i > 0 or j > 0:
        if E[i,j] > 0:
            # get operand
            if E[i,j] == 1: arg = t[j-1]
            if E[i,j] == 2: arg = s[i-1]
            if E[i,j] == 3: arg = s[i-1]+t[j-1]
            # record edit
            edits.insert(0, [
                E[i,j], # operation
                i-1, # position in s (insert after, delete at, subtitute at)
                arg, # what to insert/delete/substitute
                ])
            # update edit positions
            for e in edits[1:]:
                e[1] += int(E[i,j] == 1) - int(E[i,j] == 2)
        i, j = i - (E[i,j] != 1), j - (E[i,j] != 2)

    ni = sum([e[0]==1 for e in edits])
    nd = sum([e[0]==2 for e in edits])
    ns = sum([e[0]==3 for e in edits])

    return L[m, n], ni, nd, ns, L, E, edits

if __name__ == "__main__":

    # s, t = "fatr", "catty"
    # s, t = ["1B","2C","1C"], ["1C","1B","2C"]
    s, t = [(1,"B"),(2,"C"),(1,"C")], [(1,"C"),(1,"B"),(2,"C")]

    l, ni, nd, ns, L, E, edits = lvd(s,t)
    print(L)
    print(E)
    print("s: "+str(s))
    print("t: "+str(t))
    print("op, pos, arg")
    for edit in edits: print(edit)
    print('l,i,d,s:')
    print(l,ni,nd,ns)
    
