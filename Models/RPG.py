import json
import copy
import random
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

from chess.pgn import Game
from chess.svg import board


@dataclass
class RPGAIMove:
    from_pos: Tuple[int, int]
    to_pos: Tuple[int, int]
    piece_type: str
    piece_name: str
    move_type: str
    confidence: float
    reasoning: str


class RPGAIModel:
    def __init__(self, difficulty: int = 300):
        self.difficulty = difficulty
        self.piece_values = {
            'pawn': 1, 'knight': 3, 'bishop': 3, 'rook': 5,
            'queen': 9, 'king': 100,
            # Custom RPG pieces
            'goblin': 1.5, 'orc': 4, 'lich': 6, 'dragon': 12,
            'troll': 5, 'vampire': 4, 'demon': 7, 'angel': 8,
            'phoenix': 10, 'griffin': 6, 'wyvern': 5, 'elemental': 9,
            'warlock': 6, 'paladin': 5, 'ranger': 3, 'bard': 4,
            'joker': 2, 'forger': 4, 'necromancer': 7, 'berserker': 6,
            'alchemist': 5, 'time_mage': 8, 'summoner': 4, 'illusionist': 5
        }

    def calculate_move(self, board: List[List], enemy_pieces: List[Dict],
                      player_pieces: List[Dict], strategy: str,
                      board_size: int, round_num: int) -> Optional[RPGAIMove]:
        """Main RPG AI CALCULATION LOGIC """

        # 1. Get all possible moves
        all_moves = self.get_all_possible_moves(board, board_size)

        if not all_moves:
            return None

        # 2. Evaluate each move
        evaluated_moves = []
        for move in all_moves:
            score = self.evaluate_move(board, move, enemy_pieces, player_pieces, strategy)
            evaluated_moves.append((move, score))

        # 3. Select move based on difficulty
        return self.select_move_by_difficulty(evaluated_moves, strategy)

    def get_all_possible_moves(self, board: List[List], board_size: int) -> List[Dict]:
        """Get all possible moves for enemy pieces"""
        moves = []

        for row in range(board_size):
            for col in range(board_size):
                piece = board[row][col]
                if piece and piece.get('color') == 'black':
                    piece_moves = self.calculate_piece_moves(
                        piece, (row, col), board, board_size
                    )
                    for move in piece_moves:
                        moves.append({
                            'from': (row, col),
                            'to': move,
                            'piece': piece
                        })
        return moves

    def calculate_piece_moves(self, piece: Dict, pos: Tuple[int, int],
                              board: List[List], board_size: int) -> List[Tuple[int, int]]:
        """Calculate moves for a specific piece using your custom movement patterns"""
        row, col = pos
        piece_type = piece.get('type', '').lower()
        piece_name = piece.get('name', '').lower()
        moves = []

        # Use your existing movement logic from chessUtils.ts
        # This is a simplified version - you'll need to port the full logic

        if piece_type == 'pawn':
            moves = self.get_pawn_moves(row, col, piece, board, board_size)
        elif piece_type == 'knight':
            moves = self.get_knight_moves(row, col, piece, board, board_size)
        elif piece_type == 'bishop':
            moves = self.get_bishop_moves(row, col, piece, board, board_size)
        elif piece_type == 'rook':
            moves = self.get_rook_moves(row, col, piece, board, board_size)
        elif piece_type == 'queen':
            moves = self.get_queen_moves(row, col, piece, board, board_size)
        elif piece_type == 'king':
            moves = self.get_king_moves(row, col, piece, board, board_size)

        # Custom RPG piece movements
        elif 'goblin' in piece_name:
            moves = self.get_goblin_moves(row, col, piece, board, board_size)
        elif 'orc' in piece_name:
            moves = self.get_orc_moves(row, col, piece, board, board_size)
        elif 'dragon' in piece_name:
            moves = self.get_dragon_moves(row, col, piece, board, board_size)

        return moves

    def evaluate_move(self, board: List[List], move: Dict,
                      enemy_pieces: List[Dict], player_pieces: List[Dict],
                      strategy: str) -> float:
        """Evaluate a move and return a score"""
        score = 0.0
        from_pos = move['from']
        to_pos = move['to']
        piece = move['piece']

        # 1. Base piece value
        piece_type = piece.get('type', '').lower()
        score += self.piece_values.get(piece_type, 1)

        # 2. Capture bonus
        target_piece = board[to_pos[0]][to_pos[1]]
        if target_piece:
            target_type = target_piece.get('type', '').lower()
            capture_value = self.piece_values.get(target_type, 1)
            score += capture_value * 2  # Capture bonus

        # 3. Positional bonuses
        score += self.get_positional_bonus(to_pos, piece, board)

        # 4. Strategy-based adjustments
        if strategy == 'aggressive':
            score += self.get_aggressive_bonus(move, board)
        elif strategy == 'defensive':
            score += self.get_defensive_bonus(move, board)

        # 5. Special ability bonuses
        score += self.get_special_ability_bonus(piece, move, board)

        return score

    def select_move_by_difficulty(self, evaluated_moves: List[Tuple],
                                  strategy: str) -> Optional[RPGAIMove]:
        """Select move based on AI difficulty level"""
        if not evaluated_moves:
            return None

        # Sort moves by score (best first)
        evaluated_moves.sort(key=lambda x: x[1], reverse=True)

        # Difficulty-based selection
        if self.difficulty <= 300:  # Very weak
            # Pick from bottom 50% of moves
            bottom_half = evaluated_moves[len(evaluated_moves) // 2:]
            selected = random.choice(bottom_half)
        elif self.difficulty <= 600:  # Weak
            # Pick from bottom 25% of moves
            bottom_quarter = evaluated_moves[len(evaluated_moves) // 4:]
            selected = random.choice(bottom_quarter)
        elif self.difficulty <= 1000:  # Intermediate
            # Pick from top 75% of moves
            top_three_quarters = evaluated_moves[:len(evaluated_moves) * 3 // 4]
            selected = random.choice(top_three_quarters)
        else:  # Strong
            # Pick from top 25% of moves
            top_quarter = evaluated_moves[:len(evaluated_moves) // 4]
            selected = random.choice(top_quarter) if top_quarter else evaluated_moves[0]

        move, score = selected
        return RPGAIMove(
            from_pos=move['from'],
            to_pos=move['to'],
            piece_type=move['piece'].get('type', ''),
            piece_name=move['piece'].get('name', ''),
            move_type='capture' if board[move['to'][0]][move['to'][1]] else 'move',
            confidence=min(1.0, score / 20.0),
            reasoning=f"Selected {move['piece'].get('name', 'piece')} move with score {score:.2f}"
        )

