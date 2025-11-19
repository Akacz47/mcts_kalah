import numpy as np
from mcts import State
from numba import jit
from numba import int8

__version__ = "1.0.0"
__author__ = ""
__email__ = "" 

class Kalah(State):
    # konstanty dla gry   
    POLE = 6
    KAMIENIE = 4
    PLAYER1_ROW = 1
    PLAYER2_ROW = 0
    
    def __init__(self, parent=None):
        super().__init__(parent)
        if self.parent:
            self.k = np.copy(self.parent.k)
            self.magazyn = np.copy(self.parent.magazyn)
            self.board = np.copy(self.parent.board)
            self.bonus1 = np.copy(self.parent.bonus1)
            self.bonus2 = np.copy(self.parent.bonus2)
            self.steal = np.copy(self.parent.steal)
        else:
            self.k = Kalah.KAMIENIE
            self.magazyn = np.zeros(2,dtype=np.int8)
            self.board = np.array([[self.k for i in range(Kalah.POLE)],[self.k for i in range(Kalah.POLE)]], dtype=np.int8)
            self.bonus1 = False
            self.bonus2 = False
            self.steal = False

    @staticmethod
    def class_repr():
        return f"{Kalah.__name__}_{Kalah.POLE}x{2}"
            
    def __str__(self): 
        if   self.bonus2: #pusty ruch
            return "ANOTHER PLAYER HAS BONUS MOVE" 
        s = ''
        for i in range(2):
            if i == 1:
                s += str(self.magazyn[0]) # magazyn
            else:
                if self.magazyn[0] < 10:
                    s += ' '

            s += '   '

            for j in range(Kalah.POLE):
                s += '|'
                s += str(self.board[i,j]) # kamienie w wierszach
                if self.board[i,j] < 10:
                    s += ' '

            s += '|'
            s += '   '

            if i == 1:
                s += str(self.magazyn[1]) # magazyn
            else:
                s += ' '

            s += '\n'

        return s
    
    def get_player_row(self):
        # turn moze byc 1 lub -1
        if self.turn == 1:
            return Kalah.PLAYER1_ROW
        return Kalah.PLAYER2_ROW
    

    
    def take_action_job(self, action_index):
        if self.bonus2: #dla wyświetlienia
            self.bonus2 = False

        if self.bonus1: #ruch pusty nie zależnie od indexu
            self.bonus1 = False
            self.bonus2 = True
            self.turn *= -1
            return True

        if  action_index > 5 or action_index < 0:
            return False
            
        player_row = self.get_player_row()
        current_row = self.get_player_row()
        stones = self.board[player_row, action_index]

        if stones == 0:
            return False

        self.board[player_row, action_index] = 0

        counter = int8(np.copy(self.turn))
        while stones != 0:
            idx = action_index + counter
            if idx == Kalah.POLE:
                idx = Kalah.POLE-1
                action_index = Kalah.POLE-1
                counter = 0
                current_row = 0
                if player_row == 1:
                    self.magazyn[player_row] += 1
                    stones -= 1
                    if stones == 0: #bonus ruch
                        self.bonus1 = True
                        self.turn *= -1
                        return True
                continue      
            elif idx < 0:
                idx = 0
                action_index = 0
                counter = 0
                current_row = 1
                if player_row == 0:
                    self.magazyn[player_row] += 1
                    stones -= 1
                    if stones == 0: #bonus ruch
                        self.bonus1 = True
                        self.turn *= -1
                        return True
                continue
            #print(idx)
            #idx = np.clip(idx,0,Kalah.POLE-1)

            # Predict if we can steal
            if self.board[current_row, idx] == 0 and player_row == current_row and stones == 1:
                if player_row == 1:
                    enemy_row = 0
                else:
                    enemy_row = 1
                if self.board[enemy_row, idx] > 0: 
                    #steal only if the enemy has what to steal
                    self.magazyn[player_row] += self.board[enemy_row, idx] + 1
                    self.board[enemy_row, idx] = 0
                    self.board[current_row, idx] -= 1
            self.board[current_row, idx] += 1
            if current_row == 1:
                counter += 1
            else:
                counter -= 1
            stones -= 1

        self.turn *= -1
        return True
    
    def compute_outcome_job(self):    
        """        
        Computes and returns the game outcome for this state in compliance with rules of Kalach game:
        {-1, 1} denoting a win for the minimizing or maximizing player;
        0 denoting a tie;  
        ``None`` when the game is ongoing.
       
        Returns:
            outcome ({-1, 0, 1} or ``None``)
                game outcome for this state.
        """  
        NUMBA = True
        if NUMBA: #faster
            numba_outcome = Kalah.compute_outcome_job_numba_jit(self.board,self.magazyn)
            if numba_outcome!=-2:
                return numba_outcome
        else: # pure Python
            if np.sum(self.board[1,:])==0:
                self.magazyn[0] += np.sum(self.board[0,:])
                self.board[0,:] = np.zeros_like(self.board[0,:])
                if self.magazyn[0]>self.magazyn[1]:
                    return -1
                elif self.magazyn[1]>self.magazyn[0]:
                    return 1
                elif self.magazyn[1] == self.magazyn[0]:
                    return 0
            elif np.sum(self.board[0,:])==0:
                self.magazyn[1] += np.sum(self.board[1,:])
                self.board[1,:] = np.zeros_like(self.board[1,:])
                if self.magazyn[0]>self.magazyn[1]:
                    return -1
                elif self.magazyn[1]>self.magazyn[0]:
                    return 1
                elif self.magazyn[1] == self.magazyn[0]:
                    return 0
        return None    
   
    @staticmethod
    @jit(int8(int8[:,:], int8[:]), nopython=True, cache=True)  
    def compute_outcome_job_numba_jit(board,magazyn):
        """Called by ``compute_outcome_job`` for faster outcomes."""  
        if np.sum(board[1,:])==0:
            magazyn[1] += np.sum(board[0,:])
            board[0,:] = np.zeros_like(board[0,:])
            if magazyn[0]>magazyn[1]:
                return -1
            elif magazyn[1]>magazyn[0]:
                return 1
            elif magazyn[1] == magazyn[0]:
                return 0
        elif np.sum(board[0,:])==0:
            magazyn[0] += np.sum(board[1,:])
            board[1,:] = np.zeros_like(board[1,:])
            if magazyn[0]>magazyn[1]:
                return -1
            elif magazyn[1]>magazyn[0]:
                return 1
            elif magazyn[1] == magazyn[0]:
                return 0
        return -2 #zamiast None
                        
    def take_random_action_playout(self):
        #TODO: sprawdzić
        """        
        Picks a uniformly random action from actions available in this state and returns the result of calling ``take_action`` with the action index as argument.
       
        Returns:
            child (State):
                result of ``take_action`` call for the random action.          
        """        
        j_indexes = np.where(self.board[self.get_player_row(),:] != 0)[0]
        j = np.random.choice(j_indexes)
        child = self.take_action(j)
        return child
    
    def get_board(self):
        #TODO
        """                
        Returns the board of this state (a two-dimensional array of bytes).
        
        Returns:
            board (ndarray[np.int8, ndim=2]):
                board of this state (a two-dimensional array of bytes).
        """        
        return self.board
    
    def get_extra_info(self):
        #TODO
        """
        Returns additional information associated with this state, as one-dimensional array of bytes,
        informing about fills of columns (how many discs have been dropped in each column). 
        
        Returns:
            extra_info (ndarray[np.int8, ndim=1] or ``None``):
                one-dimensional array with additional information associated with this state - fills of columns.        
        """
        return None  
    
    @staticmethod    
    def action_name_to_index(action_name):     
        return int(action_name)

    @staticmethod
    def action_index_to_name(action_index):      
        return str(action_index)
    
    @staticmethod
    def get_board_shape():      
        return (Kalah.POLE, 2)

    @staticmethod
    def get_extra_info_memory():      
        return 2*Kalah.POLE + 2

    @staticmethod
    def get_max_actions():
        """
        Returns the maximum number of actions (the largest branching factor) equal to the number of columns.
        
        Returns:
            max_actions (int):
                maximum number of actions (the largest branching factor) equal to the number of columns.
        """                
        return Kalah.POLE
    