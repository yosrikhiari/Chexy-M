import random
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


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

    def calculate_move(self, board: List[List[Optional[Dict]]], enemy_pieces: List[Dict],
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

    def get_all_possible_moves(self, board: List[List[Optional[Dict]]], board_size: int) -> List[Dict]:
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

    def calculate_piece_moves(self, piece: Dict, pos: Tuple[int, int], board: List[List[Optional[Dict]]], board_size: int) -> List[Tuple[int, int]]:
        row, col = pos
        piece_type = str(piece.get('type', '')).lower()
        piece_name = str(piece.get('name', '')).lower()
        moves: List[Tuple[int, int]] = []

        def in_bounds(r: int, c: int) -> bool:
            return 0 <= r < board_size and 0 <= c < board_size

        def add_if_legal(rr: int, cc: int):
            if not in_bounds(rr, cc):
                return
            target_cell = board[rr][cc]
            if target_cell is None or target_cell.get('color') != piece.get('color'):
                moves.append((rr, cc))

        # Standard pieces
        if piece_type == 'pawn':
            direction = -1 if piece.get('color') == 'white' else 1
            f1 = (row + direction, col)
            if in_bounds(*f1) and board[f1[0]][f1[1]] is None:
                moves.append(f1)
                start_row = board_size - 2 if piece.get('color') == 'white' else 1
                if row == start_row and board[row + 2 * direction][col] is None:
                    moves.append((row + 2 * direction, col))
            for dc in (-1, 1):
                rr, cc = row + direction, col + dc
                if in_bounds(rr, cc) and board[rr][cc] is not None and board[rr][cc].get('color') != piece.get('color'):
                    moves.append((rr, cc))

        elif piece_type == 'knight':
            for dr, dc in [(-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1)]:
                add_if_legal(row + dr, col + dc)

        elif piece_type in ('rook', 'bishop', 'queen'):
            directions = []
            if piece_type in ('rook', 'queen'):
                directions += [(-1, 0), (1, 0), (0, -1), (0, 1)]
            if piece_type in ('bishop', 'queen'):
                directions += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
            for dr, dc in directions:
                rr, cc = row + dr, col + dc
                while in_bounds(rr, cc):
                    target_cell = board[rr][cc]
                    if target_cell is None:
                        moves.append((rr, cc))
                    else:
                        if target_cell.get('color') != piece.get('color'):
                            moves.append((rr, cc))
                        break
                    rr += dr
                    cc += dc

        elif piece_type == 'king':
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    add_if_legal(row + dr, col + dc)

        # Custom RPG pieces (minimal viable)
        if 'goblin' in piece_name:
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    add_if_legal(row + dr, col + dc)
            # 2 forward if path clear
            direction = -1 if piece.get('color') == 'white' else 1
            rr1, cc1 = row + direction, col
            rr2, cc2 = row + 2 * direction, col
            if in_bounds(rr1, cc1) and board[rr1][cc1] is None and in_bounds(rr2, cc2) and board[rr2][cc2] is None:
                moves.append((rr2, cc2))

        elif 'orc' in piece_name:
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                for dist in (1, 2, 3):
                    rr, cc = row + dr * dist, col + dc * dist
                    if not in_bounds(rr, cc):
                        break
                    target_cell = board[rr][cc]
                    if target_cell is None:
                        moves.append((rr, cc))
                    else:
                        if target_cell.get('color') != piece.get('color'):
                            moves.append((rr, cc))
                        break

        elif 'dragon' in piece_name:
            # Queen-like sliding
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
                rr, cc = row + dr, col + dc
                while in_bounds(rr, cc):
                    target_cell = board[rr][cc]
                    if target_cell is None:
                        moves.append((rr, cc))
                    else:
                        if target_cell.get('color') != piece.get('color'):
                            moves.append((rr, cc))
                        break
                    rr += dr
                    cc += dc
            # Flight radius 2 (ignores blockers): any square within 2 if empty or enemy
            for dr in (-2, -1, 0, 1, 2):
                for dc in (-2, -1, 0, 1, 2):
                    if dr == 0 and dc == 0:
                        continue
                    add_if_legal(row + dr, col + dc)

        elif 'lich' in piece_name:
            # Bishop-like + teleport anywhere empty
            for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                rr, cc = row + dr, col + dc
                while in_bounds(rr, cc):
                    target_cell = board[rr][cc]
                    if target_cell is None:
                        moves.append((rr, cc))
                    else:
                        if target_cell.get('color') != piece.get('color'):
                            moves.append((rr, cc))
                        break
                    rr += dr
                    cc += dc
            for rr in range(board_size):
                for cc in range(board_size):
                    if (rr, cc) != (row, col) and board[rr][cc] is None:
                        moves.append((rr, cc))

        return moves

    def evaluate_move(self, board: List[List[Optional[Dict]]], move: Dict,
                      enemy_pieces: List[Dict], player_pieces: List[Dict],
                      strategy: str) -> float:
        """Evaluate a move and return a score"""
        score = 0.0
        _from_pos = move['from']
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

    # --- Helper bonus calculators (stubs to satisfy lints; extend as needed) ---
    def get_positional_bonus(self, to_pos: Tuple[int, int], piece: Dict, board: List[List[Optional[Dict]]]) -> float:
        return 0.0

    def get_aggressive_bonus(self, move: Dict, board: List[List[Optional[Dict]]]) -> float:
        return 0.0

    def get_defensive_bonus(self, move: Dict, board: List[List[Optional[Dict]]]) -> float:
        return 0.0

    def get_special_ability_bonus(self, piece: Dict, move: Dict, board: List[List[Optional[Dict]]]) -> float:
        return 0.0

    def select_move_by_difficulty(self, evaluated_moves: List[Tuple],
                                  strategy: str) -> Optional[RPGAIMove]:
        """Select move based on AI difficulty level"""
        if not evaluated_moves:
            return None

        # Sort moves by score ( the best first)
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
            move_type='move',
            confidence=min(1.0, score / 20.0),
            reasoning=f"Selected {move['piece'].get('name', 'piece')} move with score {score:.2f}"
        )

