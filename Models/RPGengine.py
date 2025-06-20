import numpy as np
import tensorflow as tf
from keras import layers, models
import random
from typing import List, Tuple, Optional, Dict
import uuid
from dataclasses import dataclass

# Type definitions based on TypeScript interfaces
PieceType = str  # 'king', 'queen', 'rook', 'bishop', 'knight', 'pawn'
PieceColor = str  # 'white', 'black'
BoardPosition = Tuple[int, int]

@dataclass
class EnhancedRPGPiece:
    id: str
    type: PieceType
    color: PieceColor
    name: str
    description: str
    hp: int
    current_hp: int
    max_hp: int
    attack: int
    defense: int
    level: int
    experience: int
    rarity: str
    owner: str  # Added 'owner' field: 'player', 'teammate', 'opponent', 'free'
    pluscurrentHp: int
    plusmaxHp: int
    plusattack: int
    plusdefense: int
    pluslevel: int
    plusexperience: int
    position: Optional[BoardPosition] = None
    hasMoved: bool = False
    isJoker: bool = False
    specialAbility: str = ""

@dataclass
class RPGModifier:
    id: str
    name: str
    description: str
    effect: str
    rarity: str
    isActive: bool

@dataclass
class CapacityModifier:
    id: str
    name: str
    description: str
    type: str
    capacityBonus: int
    rarity: str
    isActive: bool
    pieceType: Optional[PieceType] = None

class MultiModalLearningSystem:
    def __init__(self, board_size: int):
        self.board_size = board_size
        self.model = self._build_neural_network()
        self.q_table: Dict[Tuple[int, int], Dict[str, float]] = {}
        self.strategies = ['defensive', 'aggressive', 'balanced', 'adaptive']  # Standardized to lowercase
        self.learning_rate = 0.1
        self.discount_factor = 0.9
        self.exploration_rate = 0.1
        self.piece_values = {'king': 1000, 'queen': 90, 'rook': 50, 'bishop': 30, 'knight': 30, 'pawn': 10}

    def _build_neural_network(self) -> tf.keras.Model:
        input_shape = (self.board_size * self.board_size * 3 + 4,)
        model = models.Sequential([
            layers.Input(shape=input_shape),
            layers.Dense(128, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dense(1, activation='linear')
        ])
        model.compile(optimizer='adam', loss='mse')
        return model

    def _determine_rarity(self, base_rarity: str, modifiers: List[Dict], capacity_modifiers: List[Dict]) -> str:
        total_modifier_strength = 0
        for modifier in modifiers:
            if not modifier.get('isActive', False):
                continue
            effect = modifier.get('effect', '')
            rarity = modifier.get('rarity', 'common')
            strength = {'common': 1, 'rare': 2, 'epic': 3, 'legendary': 5}.get(rarity, 1)
            if effect in ['pawn_promotion_boost', 'rook_explosion', 'knight_double_move', 'king_immunity']:
                total_modifier_strength += strength
        for cap_modifier in capacity_modifiers:
            if not cap_modifier.get('isActive', False):
                continue
            strength = {'common': 1, 'rare': 2, 'epic': 3, 'legendary': 5}.get(cap_modifier.get('rarity', 'common'), 1)
            total_modifier_strength += strength * max(1, cap_modifier.get('capacityBonus', 0) // 2)
        if total_modifier_strength >= 8:
            return 'legendary'
        elif total_modifier_strength >= 5:
            return 'epic'
        elif total_modifier_strength >= 2:
            return 'rare'
        return base_rarity

    def generate_enemy_army(self, round: int, board_size: int, modifiers: List[Dict] = [],
                           capacity_modifiers: List[Dict] = []) -> List[EnhancedRPGPiece]:
        difficulty = min(5, 1 + round // 3)
        base_stats = {'hp': 15 + (round * 3), 'attack': 8 + (round * 2), 'defense': 5 + round}
        army = []
        piece_count = min(16, 6 + round)

        king_rarity = self._determine_rarity('legendary', modifiers, capacity_modifiers)
        army.append(EnhancedRPGPiece(
            id=f"enemy_king_{round}", type='king', color='black', name=f"Dark King Lvl.{difficulty}",
            description="The enemy commander", hp=base_stats['hp'] + 10, current_hp=base_stats['hp'] + 10,
            max_hp=base_stats['hp'] + 10, attack=base_stats['attack'], defense=base_stats['defense'] + 3,
            level=difficulty, experience=0, rarity=king_rarity, owner='opponent',
            pluscurrentHp=base_stats['hp'] + 10, plusmaxHp=base_stats['hp'] + 10, plusattack=base_stats['attack'],
            plusdefense=base_stats['defense'] + 3, pluslevel=difficulty, plusexperience=0
        ))

        is_boss_round = round % 5 == 0
        expose_queen = is_boss_round and round >= 4
        queen_rarity = self._determine_rarity('epic', modifiers, capacity_modifiers)
        army.append(EnhancedRPGPiece(
            id=f"enemy_queen_{round}", type='queen', color='black', name=f"Dark Queen Lvl.{difficulty}",
            description="Exposed and vulnerable!" if expose_queen else "The enemy's most powerful piece",
            hp=base_stats['hp'] + 5, current_hp=int(base_stats['hp'] * 0.6) if expose_queen else base_stats['hp'] + 5,
            max_hp=base_stats['hp'] + 5, attack=base_stats['attack'] + 4,
            defense=int(base_stats['defense'] * 0.5) if expose_queen else base_stats['defense'] + 2,
            level=difficulty, experience=0, rarity=queen_rarity, owner='opponent',
            pluscurrentHp=int(base_stats['hp'] * 0.6) if expose_queen else base_stats['hp'] + 5,
            plusmaxHp=base_stats['hp'] + 5, plusattack=base_stats['attack'] + 4,
            plusdefense=int(base_stats['defense'] * 0.5) if expose_queen else base_stats['defense'] + 2,
            pluslevel=difficulty, plusexperience=0
        ))

        piece_types = ['rook', 'bishop', 'knight', 'pawn']
        for i in range(2, piece_count):
            piece_type = piece_types[i % len(piece_types)]
            stat_multiplier = 0.7 if piece_type == 'pawn' else 1.0
            base_rarity = 'rare' if i < 6 else 'common'
            rarity = self._determine_rarity(base_rarity, modifiers, capacity_modifiers)
            army.append(EnhancedRPGPiece(
                id=f"enemy_{piece_type}_{round}_{i}", type=piece_type, color='black',
                name=f"Dark {piece_type.capitalize()} Lvl.{difficulty}", description=f"Enhanced enemy {piece_type}",
                hp=int(base_stats['hp'] * stat_multiplier), current_hp=int(base_stats['hp'] * stat_multiplier),
                max_hp=int(base_stats['hp'] * stat_multiplier), attack=int(base_stats['attack'] * stat_multiplier),
                defense=int(base_stats['defense'] * stat_multiplier), level=difficulty, experience=0, rarity=rarity,
                owner='opponent', pluscurrentHp=int(base_stats['hp'] * stat_multiplier),
                plusmaxHp=int(base_stats['hp'] * stat_multiplier), plusattack=int(base_stats['attack'] * stat_multiplier),
                plusdefense=int(base_stats['defense'] * stat_multiplier), pluslevel=difficulty, plusexperience=0
            ))
        return army

    def get_ai_strategy(self, round: int, player_army_size: int) -> str:
        state = (round, player_army_size)
        if state not in self.q_table:
            self.q_table[state] = {s: 0.0 for s in self.strategies}
        if random.random() < self.exploration_rate:
            return random.choice(self.strategies)
        return max(self.q_table[state], key=self.q_table[state].get)

    def calculate_valid_moves(self, position: BoardPosition, board: List[List[Optional[EnhancedRPGPiece]]],
                             color: PieceColor, board_size: int, is_rpg: bool, game_id: str, game_mode: str,
                             player_id: Optional[str] = None, en_passant_target: Optional[BoardPosition] = None) -> List[BoardPosition]:
        piece = board[position[0]][position[1]]
        if not piece or piece.color != color:
            return []
        moves = []
        direction = 1 if piece.color == 'black' else -1
        start_row = 1 if piece.color == 'black' else board_size - 2

        if piece.type == 'pawn':
            single_move = (position[0] + direction, position[1])
            if self._is_valid_position(single_move, board_size) and not board[single_move[0]][single_move[1]]:
                moves.append(single_move)
                if position[0] == start_row:
                    double_move = (position[0] + 2 * direction, position[1])
                    if self._is_valid_position(double_move, board_size) and not board[double_move[0]][double_move[1]]:
                        moves.append(double_move)
            for dc in [-1, 1]:
                capture_pos = (position[0] + direction, position[1] + dc)
                if self._is_valid_position(capture_pos, board_size):
                    target = board[capture_pos[0]][capture_pos[1]]
                    if target and target.color != color and not self._is_teammate(target, player_id, game_mode):
                        moves.append(capture_pos)
                    elif en_passant_target and capture_pos == en_passant_target:
                        moves.append(capture_pos)
        elif piece.type == 'knight':
            knight_offsets = [(-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1)]
            for dr, dc in knight_offsets:
                move = (position[0] + dr, position[1] + dc)
                if self._is_valid_position(move, board_size):
                    target = board[move[0]][move[1]]
                    if not target or (target.color != color and not self._is_teammate(target, player_id, game_mode)):
                        moves.append(move)
        elif piece.type == 'king':
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    move = (position[0] + dr, position[1] + dc)
                    if self._is_valid_position(move, board_size):
                        target = board[move[0]][move[1]]
                        if not target or (target.color != color and not self._is_teammate(target, player_id, game_mode)):
                            moves.append(move)
        elif piece.type in ['rook', 'bishop', 'queen']:
            directions = []
            if piece.type in ['rook', 'queen']:
                directions.extend([(-1, 0), (1, 0), (0, -1), (0, 1)])
            if piece.type in ['bishop', 'queen']:
                directions.extend([(-1, -1), (-1, 1), (1, -1), (1, 1)])
            for dr, dc in directions:
                curr_pos = (position[0] + dr, position[1] + dc)
                while self._is_valid_position(curr_pos, board_size):
                    target = board[curr_pos[0]][curr_pos[1]]
                    if not target:
                        moves.append(curr_pos)
                    elif target.color != color and not self._is_teammate(target, player_id, game_mode):
                        moves.append(curr_pos)
                        break
                    else:
                        break
                    curr_pos = (curr_pos[0] + dr, curr_pos[1] + dc)
        if is_rpg and game_mode in ['MULTIPLAYER_RPG', 'ENHANCED_RPG']:
            moves = [move for move in moves if not self._would_be_in_check(position, move, board, color, board_size)]
        return moves

    def _is_valid_position(self, pos: BoardPosition, board_size: int) -> bool:
        return 0 <= pos[0] < board_size and 0 <= pos[1] < board_size

    def _is_teammate(self, piece: EnhancedRPGPiece, player_id: Optional[str], game_mode: str) -> bool:
        if game_mode != 'MULTIPLAYER_RPG' or not player_id:
            return False
        return piece.color == 'white' and piece.owner != 'player'  # Use 'owner' instead of id splitting

    def _would_be_in_check(self, from_pos: BoardPosition, to_pos: BoardPosition,
                          board: List[List[Optional[EnhancedRPGPiece]]], color: PieceColor, board_size: int) -> bool:
        test_board = [row[:] for row in board]
        test_board[to_pos[0]][to_pos[1]] = test_board[from_pos[0]][from_pos[1]]
        test_board[from_pos[0]][from_pos[1]] = None
        king_pos = self._find_king_position(test_board, color, board_size)
        if not king_pos:
            return False
        opponent_color = 'white' if color == 'black' else 'black'
        for row in range(board_size):
            for col in range(board_size):
                piece = test_board[row][col]
                if piece and piece.color == opponent_color:
                    moves = self.calculate_valid_moves((row, col), test_board, opponent_color, board_size, False, "", "")
                    if any(move == king_pos for move in moves):
                        return True
        return False

    def _find_king_position(self, board: List[List[Optional[EnhancedRPGPiece]]], color: PieceColor, board_size: int) -> Optional[BoardPosition]:
        for row in range(board_size):
            for col in range(board_size):
                piece = board[row][col]
                if piece and piece.type == 'king' and piece.color == color:
                    return (row, col)
        return None

    def _board_to_features(self, board: List[List[Optional[EnhancedRPGPiece]]], from_pos: BoardPosition,
                          to_pos: BoardPosition, strategy: str, round: int) -> np.ndarray:
        features = []
        for row in range(self.board_size):
            for col in range(self.board_size):
                piece = board[row][col]
                if piece:
                    features.extend([
                        {'king': 1, 'queen': 2, 'rook': 3, 'bishop': 4, 'knight': 5, 'pawn': 6}.get(piece.type, 0),
                        1 if piece.color == 'black' else -1,
                        piece.current_hp / max(1, piece.max_hp)  # Updated to current_hp/max_hp
                    ])
                else:
                    features.extend([0, 0, 0])
        features.extend([
            from_pos[0] / self.board_size, from_pos[1] / self.board_size,
            to_pos[0] / self.board_size, to_pos[1] / self.board_size
        ])
        return np.array(features)

    def calculate_move_priority(self, from_pos: BoardPosition, to_pos: BoardPosition,
                               board: List[List[Optional[EnhancedRPGPiece]]], strategy: str, round: int) -> float:
        features = self._board_to_features(board, from_pos, to_pos, strategy, round)
        priority = float(self.model.predict(features.reshape(1, -1), verbose=0)[0])
        target_piece = board[to_pos[0]][to_pos[1]]
        if target_piece and target_piece.color == 'white':
            priority += self.piece_values.get(target_piece.type, 10)
        if strategy == 'aggressive':
            if target_piece:
                priority *= 1.5
            if to_pos[0] > self.board_size / 2:
                priority += 10
        elif strategy == 'defensive':
            if from_pos[0] < self.board_size / 3:
                priority += 15
            if not target_piece:
                priority += 5
        elif strategy == 'balanced':
            priority += random.random() * 10
        elif strategy == 'adaptive':
            aggression_level = min(1.0, round * 0.15)
            if target_piece:
                priority *= (1 + aggression_level)
        priority += random.random() * 5
        return priority

    def calculate_ai_move(self, board: List[List[Optional[EnhancedRPGPiece]]], enemy_pieces: List[EnhancedRPGPiece],
                         strategy: str, board_size: int, round: int, game_id: str, game_mode: str,
                         player_id: Optional[str] = None) -> Optional[Dict[str, BoardPosition]]:
        all_moves = []
        for row in range(board_size):
            for col in range(board_size):
                piece = board[row][col]
                if piece and piece.color == 'black':
                    moves = self.calculate_valid_moves((row, col), board, 'black', board_size, True, game_id, game_mode, player_id)
                    for move in moves:
                        priority = self.calculate_move_priority((row, col), move, board, strategy, round)
                        all_moves.append({'from': (row, col), 'to': move, 'priority': priority})
        if not all_moves:
            return None
        all_moves.sort(key=lambda x: x['priority'], reverse=True)
        smartness_factor = min(0.9, round * 0.1)
        random_factor = 1 - smartness_factor
        if random.random() < random_factor:
            return random.choice(all_moves[:min(5, len(all_moves))])
        return all_moves[0]

    def update_q_table(self, state: Tuple[int, int], strategy: str, reward: float, next_state: Tuple[int, int]):
        if state not in self.q_table:
            self.q_table[state] = {s: 0.0 for s in self.strategies}
        if next_state not in self.q_table:
            self.q_table[next_state] = {s: 0.0 for s in self.strategies}
        current_q = self.q_table[state][strategy]
        max_future_q = max(self.q_table[next_state].values())
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_future_q - current_q)
        self.q_table[state][strategy] = new_q