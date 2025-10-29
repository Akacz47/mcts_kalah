import numpy as np
class Kamienie():
    def __init__(self):
        # Gdy my mamy bonusowy ruch, to dodac jeden ruch dla przeciwnika, ktory jest pusty (taki jakiego nie da sie wykonac) i z powrotem ruch trafia do nas
        self.k = 4
        self.magazyn1 = 0
        self.magazyn2 = 0
        self.board = np.array([[self.k for i in range(6)],[self.k for i in range(6)]])
        self.bonus = False
        self.steal = False
    def __str__(self):
        s = ''
        for i in range(2):
            if i == 1:
                s += str(self.magazyn1) # magazyn
            else:
                s += ' '

            s += '   '

            for j in range(6):
                s += '|'
                s += str(self.board[i,j]) # kamienie w wierszach

            s += '|'
            s += '   '

            if i == 1:
                s += str(self.magazyn2) # magazyn
            else:
                s += ' '

            s += '\n'
        
        return s
    

kamienie = Kamienie()
print(kamienie)