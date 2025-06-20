import os
import time
import random
from flask import Flask, request, jsonify
from flask_cors import CORS
import chess
import torch

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

app = Flask(__name__)
CORS(app)

@dataclass
class BotDifficulty:
    points: int
    skill_level: int
    time_limit: float  # seconds per move
    randomness: float  # 0-1, how often it makes suboptimal moves
    description: str

DIFFICULTY_LEVELS = [
    BotDifficulty(400, 0, 1.0, 0.5, "Beginner - Makes many mistakes"),
    BotDifficulty(800, 5, 1.5, 0.3, "Novice - Knows basic strategies"),
    BotDifficulty(1200, 10, 2.0, 0.2, "Intermediate - Solid play"),
    BotDifficulty(1800, 15, 2.5, 0.1, "Advanced - Strong tactical play"),
    BotDifficulty(2400, 20, 3.0, 0.05, "Master - Near-perfect play")
]

class ClassicChessAI:
    def __init__(self, bot_points: int):
        self.bot_points = bot_points
        self.difficulty = self._get_difficulty_level(bot_points)
        self.board = chess.Board()
        self.model = self._load_mml_model()

    def _get_difficulty_level(self, points: int) -> BotDifficulty:
        closest = min(DIFFICULTY_LEVELS, key=lambda x: abs(x.points - points))
        return closest

    def _load_mml_model(self):
        try:
            model_path = 'models/chess_mml_model.pth'
            if not os.path.exists(model_path):
                print(f"Model file not found at {model_path}. Using random move fallback.")
                return None
            model = torch.load(model_path, map_location=torch.device('cpu'))
            model.eval()
            return model
        except Exception as e:
            print(f"Failed to load MML model: {e}")
            return None

    def _fen_to_input(self, fen: str):
        board = chess.Board(fen)
        input_tensor = torch.zeros(12, 8, 8)
        for sq in chess.SQUARES:
            piece = board.piece_at(sq)
            if piece:
                piece_idx = (piece.piece_type - 1) + (6 if piece.color == chess.BLACK else 0)
                row = 7 - (sq // 8)
                col = sq % 8
                input_tensor[piece_idx, row, col] = 1
        return input_tensor.unsqueeze(0)

    def _output_to_move_probs(self, output, legal_moves):
        move_probs = {}
        for move in legal_moves:
            move_idx = self._move_to_index(move)
            prob = output[0, move_idx].item() if move_idx < output.shape[1] else 0
            move_probs[move] = prob
        return move_probs

    def _move_to_index(self, move):
        return move.from_square * 64 + move.to_square

    def _get_mml_move(self, legal_moves):
        if not self.model:
            return random.choice(legal_moves) if legal_moves else None

        input_data = self._fen_to_input(self.board.fen())
        with torch.no_grad():
            output = self.model(input_data)

        move_probs = self._output_to_move_probs(output, legal_moves)
        sorted_moves = sorted(move_probs.items(), key=lambda x: x[1], reverse=True)
        N = max(1, int(len(sorted_moves) * (1 - self.difficulty.randomness)))
        top_moves = sorted_moves[:N]
        selected_move, _ = random.choice(top_moves)
        return selected_move

    def set_board(self, fen: str):
        try:
            self.board = chess.Board(fen)
        except ValueError as e:
            print(f"Invalid FEN: {fen}, error: {e}")
            self.board = chess.Board()

    def make_move(self):
        legal_moves = list(self.board.legal_moves)
        if not legal_moves:
            return None

        base_time = self.difficulty.time_limit
        variation = random.uniform(0.5, 1.5)
        thinking_time = base_time * variation
        time.sleep(thinking_time)

        move = self._get_mml_move(legal_moves)
        from_sq = move.from_square
        to_sq = move.to_square
        from_row = 7 - (from_sq // 8)
        from_col = from_sq % 8
        to_row = 7 - (to_sq // 8)
        to_col = to_sq % 8
        return ((from_row, from_col), (to_row, to_col))

ai_instances = {}

def get_classic_ai(bot_points: int):
    if bot_points not in ai_instances:
        ai_instances[bot_points] = ClassicChessAI(bot_points)
    return ai_instances[bot_points]
