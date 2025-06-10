import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import sys
from flask import Flask, request, jsonify
from flask_cors import CORS
from engine import MultiModalLearningSystem, EnhancedRPGPiece
from typing import List, Optional, Dict
import tensorflow as tf

tf.get_logger().setLevel('ERROR')

app = Flask(__name__)
CORS(app)

try:
    ai_system = MultiModalLearningSystem(board_size=8)
except Exception as e:
    print(f"❌ Failed to initialize AI system: {e}")
    sys.exit(1)

def serialize_piece(piece: EnhancedRPGPiece) -> Dict:
    return {
        'id': piece.id, 'type': piece.type, 'color': piece.color, 'name': piece.name,
        'description': piece.description, 'hp': piece.hp, 'currentHp': piece.current_hp,
        'maxHp': piece.max_hp, 'attack': piece.attack, 'defense': piece.defense,
        'level': piece.level, 'experience': piece.experience, 'rarity': piece.rarity,
        'owner': piece.owner,  # Now explicitly supported
        'pluscurrentHp': piece.pluscurrentHp, 'plusmaxHp': piece.plusmaxHp,
        'plusattack': piece.plusattack, 'plusdefense': piece.plusdefense,
        'pluslevel': piece.pluslevel, 'plusexperience': piece.plusexperience,
        'position': piece.position, 'hasMoved': piece.hasMoved, 'isJoker': piece.isJoker,
        'specialAbility': piece.specialAbility
    }

def create_piece_safely(piece_data: dict) -> Optional[EnhancedRPGPiece]:
    try:
        required_fields = [
            'id', 'type', 'color', 'name', 'description', 'hp', 'current_hp', 'max_hp',
            'attack', 'defense', 'level', 'experience', 'rarity', 'owner',
            'pluscurrentHp', 'plusmaxHp', 'plusattack', 'plusdefense', 'pluslevel', 'plusexperience'
        ]
        for field in required_fields:
            if field not in piece_data:
                piece_data[field] = '' if field in ['id', 'name', 'description', 'rarity', 'owner'] else 0
        piece_data.setdefault('position', None)
        piece_data.setdefault('hasMoved', False)
        piece_data.setdefault('isJoker', False)
        piece_data.setdefault('specialAbility', '')
        return EnhancedRPGPiece(**piece_data)
    except Exception as e:
        print(f"⚠️ Error creating piece from data {piece_data.get('id', 'unknown')}: {e}")
        return None

@app.route('/api/enemy-army', methods=['POST'])
def generate_enemy_army():
    try:
        data = request.get_json()
        round = data.get('round', 1)
        board_size = data.get('boardSize', 8)
        modifiers = data.get('activeModifiers', [])
        capacity_modifiers = data.get('activeCapacityModifiers', [])
        army = ai_system.generate_enemy_army(round, board_size, modifiers, capacity_modifiers)
        serialized_army = [serialize_piece(piece) for piece in army if piece]
        return jsonify({'pieces': serialized_army})
    except Exception as e:
        print(f"❌ Error in generate_enemy_army: {e}")
        return jsonify({'error': 'Failed to generate enemy army'}), 500

@app.route('/api/ai-strategy', methods=['POST'])
def get_ai_strategy():
    try:
        data = request.get_json()
        round = data.get('round', 1)
        player_army_size = data.get('playerArmySize', 8)
        strategy = ai_system.get_ai_strategy(round, player_army_size)
        return jsonify({'strategy': strategy.lower()})  # Ensure lowercase
    except Exception as e:
        print(f"❌ Error in get_ai_strategy: {e}")
        return jsonify({'error': 'Failed to get AI strategy'}), 500

@app.route('/api/ai-move', methods=['POST'])
def calculate_ai_move():
    try:
        data = request.get_json()
        board = data.get('board')
        enemy_pieces_data = data.get('enemyPieces', [])
        player_pieces_data = data.get('playerPieces', [])
        enemy_pieces = [create_piece_safely(piece) for piece in enemy_pieces_data]
        player_pieces = [create_piece_safely(piece) for piece in player_pieces_data]
        enemy_pieces = [piece for piece in enemy_pieces if piece is not None]
        player_pieces = [piece for piece in player_pieces if piece is not None]
        strategy = data.get('strategy', 'balanced').lower()  # Convert to lowercase
        board_size = data.get('boardSize', 8)
        round = data.get('round', 1)
        game_id = data.get('gameId', '')
        game_mode = data.get('gameMode', 'ENHANCED_RPG')
        player_id = data.get('playerId')

        board_array = [[None for _ in range(board_size)] for _ in range(board_size)]
        for row_idx, row in enumerate(board):
            for col_idx, piece_data in enumerate(row):
                if piece_data:
                    piece = create_piece_safely(piece_data)
                    if piece and 0 <= row_idx < board_size and 0 <= col_idx < board_size:
                        board_array[row_idx][col_idx] = piece

        move = ai_system.calculate_ai_move(board_array, enemy_pieces, strategy, board_size, round, game_id, game_mode, player_id)
        if move:
            return jsonify({
                'from': {'row': move['from'][0], 'col': move['from'][1]},
                'to': {'row': move['to'][0], 'col': move['to'][1]}
            })
        return jsonify({'move': None})
    except Exception as e:
        print(f"❌ Error in calculate_ai_move: {e}")
        return jsonify({'error': 'Failed to calculate AI move'}), 500

@app.route('/api/update-ai', methods=['POST'])
def update_ai():
    try:
        data = request.get_json()
        state = (data.get('round'), data.get('playerArmySize'))
        strategy = data.get('strategy').lower()  # Convert to lowercase
        reward = data.get('reward')
        next_state = (data.get('nextRound'), data.get('nextPlayerArmySize'))
        ai_system.update_q_table(state, strategy, reward, next_state)
        return jsonify({'status': 'success'})
    except Exception as e:
        print(f"❌ Error in update_ai: {e}")
        return jsonify({'error': 'Failed to update AI'}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'message': 'AI Chess Game Server is running',
        'endpoints': ['/api/enemy-army', '/api/ai-strategy', '/api/ai-move', '/api/update-ai']
    })

def main():
    try:
        from waitress import serve
        print("🚀 Starting AI Chess Game Server with Waitress...")
        print("📡 Server running at: http://localhost:5000")
        print("📡 Server also available at: http://0.0.0.0:5000")
        print("⏹️  Press Ctrl+C to stop")
        print("=" * 50)
        serve(app, host='0.0.0.0', port=5000, threads=4, max_request_body_size=1073741824,
              cleanup_interval=30, channel_timeout=120, log_untrusted_proxy_headers=True,
              connection_limit=1000, recv_bytes=65536, send_bytes=65536, _quiet=True)
    except ImportError:
        print("❌ Waitress not found! Falling back to Flask...")
        app.run(debug=False, port=5000, host='0.0.0.0', use_reloader=False, threaded=True)
    except KeyboardInterrupt:
        print("\n✅ Server stopped gracefully")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()