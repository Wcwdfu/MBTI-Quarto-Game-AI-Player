# utils.py

import numpy as np
import copy

class QuartoEnv:
    def __init__(self):
        self.reset()

    def reset(self):
        self.board = np.zeros((4, 4), dtype=int)
        self.available_pieces = [(i, j, k, l) for i in range(2) for j in range(2)
                                 for k in range(2) for l in range(2)]
        self.selected_piece = None
        self.done = False
        self.winner = None
        self.phase = 'select_piece'
        self.current_player = 2
        return self.get_state_vector()

    def get_state_vector(self):
        board_state_flattened = self.board.flatten()
        available_pieces_ids = np.array([self.get_piece_id(p)
                                        for p in self.available_pieces])
        if len(available_pieces_ids) < 16:
            available_pieces_ids = np.pad(available_pieces_ids, (0, 16 - len(available_pieces_ids)),
                                        'constant', constant_values=-1)
        selected_piece_id = self.get_piece_id(self.selected_piece) if self.selected_piece else -1
        phase_indicator = 1 if self.phase == 'select_piece' else 0
        current_player_indicator = self.current_player  # 현재 플레이어 (1 또는 2)

        state_vector = np.concatenate([
            board_state_flattened,
            available_pieces_ids,
            np.array([selected_piece_id]),
            np.array([phase_indicator]),
            np.array([current_player_indicator])
        ])
        return state_vector


    def get_valid_actions(self):
        if self.phase == 'select_piece':
            return [self.get_piece_id(p) for p in self.available_pieces]
        else:
            return [i for i in range(16) if self.board[i // 4][i % 4] == 0]


    def step(self, action):
        if self.phase == 'select_piece':
            piece_id = action
            # Find the piece with the matching piece_id in available_pieces
            for idx, p in enumerate(self.available_pieces):
                if self.get_piece_id(p) == piece_id:
                    self.selected_piece = p
                    del self.available_pieces[idx]
                    break
            else:
                raise ValueError(f"Piece with ID {piece_id} not found in available_pieces")
            self.phase = 'place_piece'
            self.current_player = 2 if self.current_player == 1 else 1
            reward = 0
            done = False
        else:
            row, col = divmod(action, 4)
            piece_id = self.get_piece_id(self.selected_piece)
            self.board[row][col] = piece_id + 1  # Ensure piece IDs start from 1
            self.selected_piece = None
            if self.check_win():
                self.done = True
                self.winner = self.current_player
                reward = 1 if self.current_player == 1 else -1
            elif np.all(self.board != 0):
                self.done = True
                self.winner = 0  # Draw
                reward = 0
            else:
                self.phase = 'select_piece'
                reward = 0
                self.current_player = 2 if self.current_player == 1 else 1
            done = self.done
        return self.get_state_vector(), reward, done


    def check_win(self):
        def check_line(line):
            if 0 in line:
                return False
            characteristics = [self.get_piece_features(idx - 1) for idx in line]
            for i in range(4):
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
        # 2x2 사각형 검사 추가
        for row in range(3):
            for col in range(3):
                square = [
                    self.board[row][col],
                    self.board[row][col + 1],
                    self.board[row + 1][col],
                    self.board[row + 1][col + 1]
                ]
                if check_line(square):
                    return True
        return False

    def is_game_over(self):
        return self.done

    def get_winner_value(self):
        if self.winner == 1:
            return 1
        elif self.winner == 2:
            return -1
        else:
            return 0

    def copy(self):
        return copy.deepcopy(self)

    def do_move(self, action, phase):
        if self.phase == 'select_piece':
            piece_id = action
            for idx, p in enumerate(self.available_pieces):
                if self.get_piece_id(p) == piece_id:
                    self.selected_piece = p
                    del self.available_pieces[idx]
                    break
            else:
                print(f"Attempting to select piece ID: {piece_id}")
                print(f"Available piece IDs: {[self.get_piece_id(p) for p in self.available_pieces]}")
                raise ValueError(f"Piece with ID {piece_id} not found in available_pieces")
            self.phase = 'place_piece'
            self.current_player = 2 if self.current_player == 1 else 1
        else:
            row, col = divmod(action, 4)
            piece_id = self.get_piece_id(self.selected_piece)
            self.board[row][col] = piece_id + 1
            self.selected_piece = None
            if self.check_win():
                self.done = True
                self.winner = self.current_player
            elif np.all(self.board != 0):
                self.done = True
                self.winner = 0  # Draw
            else:
                self.phase = 'select_piece'
                self.current_player = 2 if self.current_player == 1 else 1



    def get_piece_id(self, piece):
        if piece is None:
            return -1
        return (piece[0] << 3) | (piece[1] << 2) | (piece[2] << 1) | piece[3]


    def get_piece_features(self, piece_id):
        return [
            (piece_id >> 3) & 1,
            (piece_id >> 2) & 1,
            (piece_id >> 1) & 1,
            piece_id & 1
        ]
