import numpy as np
from mcts import MCTS
from mctsnc import MCTSNC
from c4 import C4
from game_runner import GameRunner
from kalah import Kalah
 

if __name__=="__main__":
    #AI = MCTSNC(Kalah.get_board_shape(), Kalah.get_extra_info_memory(), Kalah.get_max_actions(), search_time_limit=5.0, search_steps_limit=np.inf, n_trees=4, n_playouts=256, variant="acp_prodigal", action_index_to_name_function=Kalah.action_index_to_name)# MCTSNC(search_time_limit=5.0, search_steps_limit=np.inf)
    AI = MCTSNC(C4.get_board_shape(), C4.get_extra_info_memory(), C4.get_max_actions(), search_time_limit=1.0, search_steps_limit=np.inf, n_trees=1, n_playouts=32, variant="ocp_thrifty", action_index_to_name_function=C4.action_index_to_name)
    game_runner = GameRunner(C4,None,AI,0,1,None)
    #game_runner = GameRunner(Kalah,AI,AI,0,1,None) # None, AI
    outcome,game_info = game_runner.run()

    # plik mechanics
    # is action legal
    # device true po stronie karty
    # extra info dodatkowe informacje o grze
    # action to numer akcji
    # take action playout chyba też
    # 5 funkcji do implementacji, one są w mechanics

    # legal action
    # nie robimy return

    # Utils do zmiany ok. 86 linijka
    # mcts.py do poprawy

    # um2
    # dowolny 1 punkt można pominąć

    # non max suppression zrobić funkcje
    # w detekcji jest siła pewności okienka
    # w tablicy detected windows i detected responses wiadomo jakie okienko miało wskazanie
    # tą tablice trzeba posortować
    # iou cords2  j+h....
    # jeżeli iou >= jakiś próg, to okienka się nakładają i wtedy sprawdzać tą siłę
    # for windows malejąco:
    #   wg responses
    #       for ...
    
    # w scikit learn jest roc curve, na podstawie wykresu wybrać próg detekcyjny
    # DETECT_DECISION_THRESHOLD
    # acc = P(Y=+) * tpr + P(Y=-) * (1-FPR) wypisać to lub zaznaczyć na krzywej roc

    # 0 lepszy klasyfikator
    # Zwiększyć parametry haara S i P, oraz zwiększyć ilość rund boostingu T
    # negatives per image = 20
    # haar_s/p = 6, 6
    # T = 256
    # picklować to

    # 1. non max suppression
    # 2. roc i wybór progu o max acc test

    # wykonaj wsadowo funckje detekcyjną... nie robić

    # 3. przyśpieszenie zrównoleglenie | joblit spróbować luźna sugestia, zrobić jako ostatni punkt
    # 4. kamera
    # video capture w cv2, on daje klatki obrazu

    # wybrać 4 z tych 5 punktów i będzie ocena na 5
    # jakby się nie zdążyło to na teams albo na projekcie się odezwać
'''
if __name__=="__main__":
    print("hello")
    '''