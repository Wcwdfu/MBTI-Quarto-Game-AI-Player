import numpy as np

class QuartoEnv:
    def __init__(self):
        self.all_pieces = [(i, j, k, l) for i in range(2) for j in range(2)
                           for k in range(2) for l in range(2)]
        self.reset()


    def reset(self):
        self.board = np.zeros((4, 4), dtype=int)
        self.available_pieces = self.all_pieces.copy()
        self.selected_piece = None
        self.done = False
        self.winner = None
        self.phase = 'select_piece'  # 'select_piece' 또는 'place_piece'
        self.current_player = 1  # 플레이어 1부터 시작
        return self.get_state()


    def get_state(self):
        board_state_flattened = self.board.flatten()
        available_pieces_ids = np.array([self.all_pieces.index(p)
                                         for p in self.available_pieces])
        if len(available_pieces_ids) < 16:
            available_pieces_ids = np.pad(available_pieces_ids, (0, 16 - len(available_pieces_ids)),
                                          'constant', constant_values=-1)
        selected_piece_id = self.all_pieces.index(self.selected_piece) if self.selected_piece else -1
        phase_indicator = 1 if self.phase == 'select_piece' else 0

        state_vector = np.concatenate([
            board_state_flattened,
            available_pieces_ids,
            np.array([selected_piece_id]),
            np.array([phase_indicator])
        ])
        return state_vector


    def step(self, action):

        if self.phase == 'select_piece':
            # 말 선택 단계
            selected_piece = self.all_pieces[action]
            self.selected_piece = selected_piece
            self.available_pieces.remove(self.selected_piece)
            self.phase = 'place_piece'
            reward = 0
            done = False
            self.current_player = 2 if self.current_player == 1 else 1
        else:
            # 말 배치 단계
            row, col = divmod(action, 4)
            piece_id = self.all_pieces.index(self.selected_piece) + 1
            self.board[row][col] = piece_id
            self.selected_piece = None

            if self.check_win():
                self.done = True
                self.winner = self.current_player
                reward = 10
            elif np.all(self.board != 0):
                self.done = True
                self.winner = 0  # 무승부
                reward = 0
            else:
                reward = 0
                self.phase = 'select_piece'
            done = self.done

        next_state = self.get_state()
        return next_state, reward, done



    def check_win(self):
        def check_line(line):
            if 0 in line:  # 비어있는 셀이 있으면 승리 불가능
                return False
            characteristics = [self.all_pieces[idx - 1] for idx in line]
            for i in range(4):  # 각 특징에 대해
                if len(set(char[i] for char in characteristics)) == 1:
                    return True
            return False

        # 행과 열 검사
        for i in range(4):
            if check_line(self.board[i, :]) or check_line(self.board[:, i]):
                return True
        # 대각선 검사
        if check_line([self.board[i, i] for i in range(4)]) or check_line([self.board[i, 3 - i] for i in range(4)]):
            return True
        # 2x2 사각형 검사
        for i in range(3):
            for j in range(3):
                square = [
                    self.board[i][j], self.board[i][j+1],
                    self.board[i+1][j], self.board[i+1][j+1]
                ]
                if check_line(square):
                    return True
        return False

    
