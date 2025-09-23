import logging
import os
import random
import sys
from functools import lru_cache

from RPG import RPGAIModel
from opening_detector import load_openings, detect_opening, Opening

import chess
import chess.engine
from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
from RPG import RPGAIModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('chess_ai.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Path to Stockfish executable
STOCKFISH_PATH = os.getenv('STOCKFISH_PATH', r'C:\Users\yosri\Desktop\projects for me\projet Chexy\stockfish-windows-x86-64-avx2\stockfish\stockfish.exe')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

OPENING_FILES = [
    os.path.join(BASE_DIR, 'Openings', 'a.tsv'),
    os.path.join(BASE_DIR, 'Openings', 'b.tsv'),
    os.path.join(BASE_DIR, 'Openings', 'c.tsv'),
    os.path.join(BASE_DIR, 'Openings', 'd.tsv'),
    os.path.join(BASE_DIR, 'Openings', 'e.tsv')
]

# Cache openings loading
@lru_cache(maxsize=1)
def get_openings() -> list[Opening]:
    """Load and cache openings data"""
    return load_openings(OPENING_FILES)

openings = get_openings()

# Ensure Stockfish is accessible
if not os.path.exists(STOCKFISH_PATH):
    logger.error(f"Stockfish not found at {STOCKFISH_PATH}. Please install it or update the path.")
    sys.exit(1)
else:
    logger.info(f"Stockfish found at {STOCKFISH_PATH}")

# Bot configuration cache
BOT_CONFIGS = {
    400: {
        'skill_level': -5,
        'think_time': 0.1,
        'mistake_probability': 0.4,
        'blunder_probability': 0.15,
        'random_move_probability': 0.1,
        'depth_limit': 3
    },
    600: {
        'skill_level': 0,
        'think_time': 0.2,
        'mistake_probability': 0.25,
        'blunder_probability': 0.08,
        'random_move_probability': 0.05,
        'depth_limit': 5
    },
    800: {
        'skill_level': 3,
        'think_time': 0.5,
        'mistake_probability': 0.15,
        'blunder_probability': 0.04,
        'random_move_probability': 0.02,
        'depth_limit': 8
    },
    1200: {
        'skill_level': 8,
        'think_time': 1.0,
        'mistake_probability': 0.08,
        'blunder_probability': 0.02,
        'random_move_probability': 0.01,
        'depth_limit': 12
    },
    1800: {
        'skill_level': 15,
        'think_time': 2.0,
        'mistake_probability': 0.03,
        'blunder_probability': 0.005,
        'random_move_probability': 0.0,
        'depth_limit': 18
    }
}

def get_bot_config(bot_points: int) -> dict:
    """Get bot configuration based on points with optimized lookup"""
    # Find the appropriate config based on bot points
    for threshold in sorted(BOT_CONFIGS.keys()):
        if bot_points <= threshold:
            return BOT_CONFIGS[threshold].copy()
    
    # Default to master level for 2400+ points
    return {
        'skill_level': 20,
        'think_time': 3.0,
        'mistake_probability': 0.0,
        'blunder_probability': 0.0,
        'random_move_probability': 0.0,
        'depth_limit': 20
    }


def evaluate_move_quality(board: chess.Board, move: chess.Move, engine) -> float:
    """Evaluate how good a move is using engine analysis"""
    try:
        # Analyze position before move
        info_before = engine.analyse(board, chess.engine.Limit(depth=10))
        score_before = info_before["score"].relative.score(mate_score=10000) if info_before["score"].relative else 0

        # Make the move temporarily
        board.push(move)

        # Analyze position after move (from opponent's perspective, so negate)
        info_after = engine.analyse(board, chess.engine.Limit(depth=10))
        score_after = -(info_after["score"].relative.score(mate_score=10000) if info_after["score"].relative else 0)

        # Undo the move
        board.pop()

        # Return the evaluation difference (positive = good move)
        return score_after - score_before
    except Exception as e:
        logger.warning(f"Error evaluating move quality: {e}")
        return 0


def get_weak_move(board: chess.Board, engine, config: dict):
    """Intentionally select a weaker move based on bot difficulty"""
    legal_moves = list(board.legal_moves)

    if not legal_moves:
        return None

    # For very weak bots, sometimes just pick a random legal move
    if random.random() < config['random_move_probability']:
        logger.debug("Bot making random move")
        return random.choice(legal_moves)

    try:
        # Get all legal moves with their evaluations
        move_evaluations = []
        for move in legal_moves:
            eval_score = evaluate_move_quality(board, move, engine)
            move_evaluations.append((move, eval_score))

        # Sort by evaluation (best to worst)
        move_evaluations.sort(key=lambda x: x[1], reverse=True)

        # Decide if we should make a mistake or blunder
        if random.random() < config['blunder_probability']:
            # Pick one of the worst 3 moves (blunder)
            logger.debug("Bot making blunder")
            worst_moves = move_evaluations[-3:]
            return random.choice(worst_moves)[0]
        elif random.random() < config['mistake_probability']:
            # Pick from bottom 25% of moves (mistake)
            logger.debug("Bot making mistake")
            bottom_quarter = max(1, len(move_evaluations) // 4)
            weak_moves = move_evaluations[-bottom_quarter:]
            return random.choice(weak_moves)[0]
        else:
            # Pick from top moves but with some randomness
            top_moves = move_evaluations[:min(3, len(move_evaluations))]
            return random.choice(top_moves)[0]

    except Exception as e:
        logger.error(f"Error in get_weak_move: {e}")
        # Fallback to random move
        return random.choice(legal_moves)


@app.route('/api/classic/ai-move', methods=['POST'])
def classic_ai_move():
    try:
        data = request.get_json()
        fen = data.get('fen')
        bot_points = data.get('botPoints', 600)
        game_id = data.get('gameId', '')

        logger.info(f"Received AI move request for game {game_id}: botPoints={bot_points}, fen={fen}")

        if not fen:
            logger.error(f"Missing FEN string for game {game_id}")
            return jsonify({'error': 'FEN string is required'}), 400

        # Initialize chess board
        try:
            board = chess.Board(fen)
            logger.debug(f"Successfully initialized board for game {game_id} with FEN: {fen}")
        except ValueError as e:
            logger.warning(f"Invalid FEN for game {game_id}: {fen}, error: {e}. Using fallback position.")
            turn = fen.split()[1] if len(fen.split()) > 1 and fen.split()[1] in ['w', 'b'] else 'w'
            fen = f"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR {turn} KQkq - 0 1"
            board = chess.Board(fen)
            logger.info(f"Fallback FEN applied for game {game_id}: {fen}")

        if board.is_game_over():
            logger.info(f"Game over for game {game_id}: {board.outcome()}")
            return jsonify({'move': None})

        # Get bot configuration
        config = get_bot_config(bot_points)
        logger.debug(f"Game {game_id}: Bot config for {bot_points} points: {config}")

        # Use Stockfish engine
        with chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH) as engine:
            # Configure engine
            if config['skill_level'] >= 0:
                engine.configure({"Skill Level": config['skill_level']})
            else:
                # For negative skill levels, we'll handle weakness manually
                engine.configure({"Skill Level": 0})

            logger.debug(f"Stockfish configured for game {game_id}")

            # For weak bots, use our custom weak move selection
            if bot_points <= 600:
                move = get_weak_move(board, engine, config)
            else:
                # For stronger bots, use normal engine play with limited time/depth
                limit = chess.engine.Limit(
                    time=config['think_time'],
                    depth=config['depth_limit']
                )
                result = engine.play(board, limit)
                move = result.move

                # Still apply occasional mistakes for intermediate bots
                if random.random() < config['mistake_probability']:
                    move = get_weak_move(board, engine, config)

            if not move:
                logger.warning(f"No move returned for game {game_id}")
                return jsonify({'move': None})

            # Convert move to frontend coordinates
            from_square = move.from_square
            to_square = move.to_square
            from_row = 7 - (from_square // 8)
            from_col = from_square % 8
            to_row = 7 - (to_square // 8)
            to_col = to_square % 8

            logger.info(
                f"Bot move for game {game_id} (points: {bot_points}): {move.uci()} (from: [{from_row},{from_col}], to: [{to_row},{to_col}])")

            return jsonify({
                'move': {
                    'from': {'row': from_row, 'col': from_col},
                    'to': {'row': to_row, 'col': to_col},
                    'gameId': game_id,
                    'botPoints': bot_points,
                    'difficulty': 'weak' if bot_points <= 600 else 'normal'
                }
            })

    except Exception as e:
        logger.error(f"Error in classic_ai_move for game {game_id}: {str(e)}", exc_info=True)
        return jsonify({'error': 'Failed to get AI move'}), 500


@app.route('/api/detect-opening', methods=['POST'])
def detect_opening_endpoint():
    try:
        data = request.json
        moves = data.get('moves')
        game_id = data.get('gameId', 'unknown')

        if not moves:
            return jsonify({'error': 'No moves provided'}), 400

        board = chess.Board()
        san_moves = []

        def is_valid_coord(row: int, col: int) -> bool:
            return isinstance(row, int) and isinstance(col, int) and 0 <= row < 8 and 0 <= col < 8

        for idx, move in enumerate(moves):
            try:
                from_row = move['from']['row']
                from_col = move['from']['col']
                to_row = move['to']['row']
                to_col = move['to']['col']

                if not (is_valid_coord(from_row, from_col) and is_valid_coord(to_row, to_col)):
                    logger.warning(f"[Opening] Invalid coordinates at move {idx}: {move}")
                    break

                from_square = chess.square(from_col, 7 - from_row)
                to_square = chess.square(to_col, 7 - to_row)

                piece = board.piece_at(from_square)
                promotion_piece = None
                if piece and piece.piece_type == chess.PAWN and to_row in (0, 7):
                    promotion_piece = chess.QUEEN

                move_obj = chess.Move(from_square, to_square, promotion=promotion_piece) if promotion_piece else chess.Move(from_square, to_square)

                if not board.is_legal(move_obj):
                    # Try promotion fallback if not already set
                    if piece and piece.piece_type == chess.PAWN and promotion_piece is None and to_row in (0, 7):
                        promo_try = chess.Move(from_square, to_square, promotion=chess.QUEEN)
                        if board.is_legal(promo_try):
                            move_obj = promo_try
                        else:
                            logger.warning(f"[Opening] Illegal move at move {idx}: {move}. Stopping opening parsing.")
                            break
                    else:
                        logger.warning(f"[Opening] Illegal move at move {idx}: {move}. Stopping opening parsing.")
                        break

                san = board.san(move_obj)
                san_moves.append(san)
                board.push(move_obj)
            except Exception as e:
                logger.error(f"[Opening] Error processing move {idx}: {move} - {e}")
                break

        if not san_moves:
            return jsonify({'eco': 'Unknown', 'name': 'Unrecognized Opening'})

        result = detect_opening(san_moves, openings)
        if result:
            return jsonify(result)
        else:
            return jsonify({'eco': 'Unknown', 'name': 'Unrecognized Opening'})
    except Exception as e:
        logger.error(f"Error in detect_opening_endpoint: {str(e)}", exc_info=True)
        return jsonify({'error': 'Failed to detect opening'}), 500


@app.route('/api/debug-opening', methods=['POST'])
def debug_opening_endpoint():
    """Debug endpoint to test opening detection with raw move data"""
    try:
        data = request.json
        if not data:
            return jsonify({'error': 'No data provided'}), 400

        # Test with a known opening
        test_moves = [
            "e4", "e5", "Nf3", "Nc6", "Bb5"  # Ruy Lopez
        ]

        result = detect_opening(test_moves, openings)

        # Also test the moves provided in the request
        user_moves = data.get('moves', [])
        user_result = None
        if user_moves:
            user_result = detect_opening(user_moves, openings)

        return jsonify({
            'test_opening': result,
            'user_opening': user_result,
            'test_moves': test_moves,
            'user_moves': user_moves,
            'total_openings_loaded': len(openings)
        })
    except Exception as e:
        logger.error(f"Error in debug endpoint: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/enemy-army', methods=['POST', 'OPTIONS'])
@cross_origin()
def generate_enemy_army():
    try:
        if request.method == 'OPTIONS':
            # Preflight CORS
            return ('', 204)

        data = request.get_json(silent=True) or {}
        round_num = int(data.get('round', 1))
        board_size = int(data.get('boardSize', 8))

        # Simple stub generation based on round/board size
        def mk_piece(pid: str, ptype: str, color: str, name: str, atk: int, df: int, hp: int, rarity: str = 'COMMON'):
            return {
                'id': pid,
                'type': ptype.upper(),
                'color': color.lower(),
                'name': name,
                'description': f"{name} of round {round_num}",
                'specialAbility': '',
                'hp': hp,
                'maxHp': hp,
                'attack': atk,
                'defense': df,
                'rarity': rarity,
                'isJoker': False,
                'currentHp': hp,
                'level': max(1, min(10, round_num // 2 + 1)),
                'experience': 0,
                'enhancedName': name
            }

        base = max(1, round_num)
        pieces = [
            mk_piece('e1', 'knight', 'black', 'Dark Rider', 2 + base, 1 + base // 2, 5 + base),
            mk_piece('e2', 'bishop', 'black', 'Shadow Seer', 2 + base, 2 + base // 3, 4 + base),
            mk_piece('e3', 'rook', 'black', 'Obsidian Guard', 3 + base, 3 + base // 2, 6 + base),
        ]

        # Scale count with board size modestly
        if board_size >= 10:
            pieces.append(mk_piece('e4', 'pawn', 'black', 'Thrall', 1 + base // 2, 1 + base // 3, 3 + base // 2))
            pieces.append(mk_piece('e5', 'pawn', 'black', 'Thrall', 1 + base // 2, 1 + base // 3, 3 + base // 2))

        return jsonify({ 'pieces': pieces })
    except Exception as e:
        logger.error(f"Error generating enemy army: {str(e)}", exc_info=True)
        return jsonify({'error': 'Failed to generate enemy army'}), 500

@app.route('/health', methods=['GET'])
def health_check():
    logger.info("Health check requested")
    return jsonify({
        'status': 'healthy',
        'message': 'AI Chess Game Server is running',
        'endpoints': ['/api/classic/ai-move', '/api/detect-opening'],
        'openings_loaded': len(openings),
        'stockfish_path': STOCKFISH_PATH
    })


# --- RPG AI endpoints ---

@app.route('/api/ai-strategy', methods=['POST', 'OPTIONS'])
@cross_origin()
def rpg_ai_strategy():
    try:
        if request.method == 'OPTIONS':
            return ('', 204)

        data = request.get_json(silent=True) or {}
        round_num = int(data.get('round', 1))
        player_army_size = int(data.get('playerArmySize', 8))

        if round_num <= 3:
            strategy = 'defensive'
        elif round_num <= 7:
            strategy = 'balanced'
        else:
            strategy = 'aggressive'

        # Simple adjustment by army size
        if player_army_size >= 12 and strategy != 'aggressive':
            strategy = 'balanced'

        return jsonify({'strategy': strategy})
    except Exception as e:
        logger.error(f"Error in rpg_ai_strategy: {str(e)}", exc_info=True)
        return jsonify({'error': 'Failed to get AI strategy'}), 500


@app.route('/api/ai-move', methods=['POST', 'OPTIONS'])
@cross_origin()
def rpg_ai_move():
    try:
        if request.method == 'OPTIONS':
            return ('', 204)

        data = request.get_json(silent=True) or {}

        board_state = data.get('board', [])
        enemy_pieces = data.get('enemyPieces', [])
        player_pieces = data.get('playerPieces', [])
        strategy = data.get('strategy', 'balanced')
        board_size = int(data.get('boardSize', 8))
        round_num = int(data.get('round', 1))
        game_id = data.get('gameId', '')

        # Difficulty scaling: start 300, +50 per round
        difficulty = max(300, 300 + (round_num - 1) * 50)

        ai = RPGAIModel(difficulty=difficulty)
        move = ai.calculate_move(board_state, enemy_pieces, player_pieces, strategy, board_size, round_num)

        if not move:
            return jsonify({'move': None, 'difficulty': difficulty})

        return jsonify({
            'move': {
                'from': {'row': move.from_pos[0], 'col': move.from_pos[1]},
                'to': {'row': move.to_pos[0], 'col': move.to_pos[1]}
            },
            'pieceType': move.piece_type,
            'pieceName': move.piece_name,
            'moveType': move.move_type,
            'confidence': move.confidence,
            'reasoning': move.reasoning,
            'difficulty': difficulty,
            'gameId': game_id
        })
    except Exception as e:
        logger.error(f"Error in rpg_ai_move: {str(e)}", exc_info=True)
        return jsonify({'error': 'Failed to calculate RPG AI move'}), 500



def main():
    try:
        from waitress import serve
        logger.info("Starting AI Chess Game Server with Waitress on http://localhost:5000")
        serve(app, host='0.0.0.0', port=5000, threads=4)
    except ImportError:
        logger.warning("Waitress not found! Falling back to Flask development server")
        app.run(debug=False, port=5000, host='0.0.0.0', threaded=True)
    except Exception as e:
        logger.error(f"Error starting server: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()