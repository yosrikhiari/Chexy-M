import logging
import os
import random
import sys
from opening_detector import load_openings, detect_opening, Opening

import chess
import chess.engine
from flask import Flask, request, jsonify
from flask_cors import CORS

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
openings = load_openings(OPENING_FILES)

# Ensure Stockfish is accessible
if not os.path.exists(STOCKFISH_PATH):
    logger.error(f"Stockfish not found at {STOCKFISH_PATH}. Please install it or update the path.")
    sys.exit(1)
else:
    logger.info(f"Stockfish found at {STOCKFISH_PATH}")


def get_bot_config(bot_points: int) -> dict:
    """Get bot configuration based on points with more aggressive difficulty scaling"""
    if bot_points <= 400:  # Beginner Bot - Make mistakes frequently
        return {
            'skill_level': -5,  # Negative skill level for very weak play
            'think_time': 0.1,
            'mistake_probability': 0.4,  # 40% chance to make a mistake
            'blunder_probability': 0.15,  # 15% chance to blunder
            'random_move_probability': 0.1,  # 10% chance for completely random move
            'depth_limit': 3
        }
    elif bot_points <= 600:  # Easy Bot
        return {
            'skill_level': 0,
            'think_time': 0.2,
            'mistake_probability': 0.25,
            'blunder_probability': 0.08,
            'random_move_probability': 0.05,
            'depth_limit': 5
        }
    elif bot_points <= 800:  # Novice Bot
        return {
            'skill_level': 3,
            'think_time': 0.5,
            'mistake_probability': 0.15,
            'blunder_probability': 0.04,
            'random_move_probability': 0.02,
            'depth_limit': 8
        }
    elif bot_points <= 1200:  # Intermediate Bot
        return {
            'skill_level': 8,
            'think_time': 1.0,
            'mistake_probability': 0.08,
            'blunder_probability': 0.02,
            'random_move_probability': 0.01,
            'depth_limit': 12
        }
    elif bot_points <= 1800:  # Advanced Bot
        return {
            'skill_level': 15,
            'think_time': 2.0,
            'mistake_probability': 0.03,
            'blunder_probability': 0.005,
            'random_move_probability': 0.0,
            'depth_limit': 18
        }
    else:  # Master Bot (2400+ points)
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
    except:
        return 0


def get_weak_move(board: chess.Board, engine, config: dict):
    """Intentionally select a weaker move based on bot difficulty"""
    legal_moves = list(board.legal_moves)

    if not legal_moves:
        return None

    # For very weak bots, sometimes just pick a random legal move
    if random.random() < config['random_move_probability']:
        logger.info("Bot making random move")
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
            logger.info("Bot making blunder")
            worst_moves = move_evaluations[-3:]
            return random.choice(worst_moves)[0]
        elif random.random() < config['mistake_probability']:
            # Pick from bottom 25% of moves (mistake)
            logger.info("Bot making mistake")
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
            logger.info(f"Successfully initialized board for game {game_id} with FEN: {fen}")
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
        logger.info(f"Game {game_id}: Bot config for {bot_points} points: {config}")

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
    data = request.json
    moves = data.get('moves')
    game_id = data.get('gameId', 'unknown')

    board = chess.Board()
    san_moves = []
    for move in moves:
        from_square = chess.square(move['from']['col'], 7 - move['from']['row'])
        to_square = chess.square(move['to']['col'], 7 - move['to']['row'])
        move_obj = chess.Move(from_square, to_square)
        if board.piece_at(from_square).piece_type == chess.PAWN and move['to']['row'] in (0, 7):
            move_obj = chess.Move(from_square, to_square, promotion=chess.QUEEN)
        san = board.san(move_obj)
        san_moves.append(san)
        board.push(move_obj)

    result = detect_opening(san_moves, openings)
    if result:
        return jsonify(result)
    else:
        return jsonify({'eco': 'Unknown', 'name': 'Unrecognized Opening'})




@app.route('/api/debug-opening', methods=['POST'])
def debug_opening_endpoint():
    """Debug endpoint to test opening detection with raw move data"""
    data = request.json
    if not data:
        return jsonify({'error': 'No data provided'}), 400

    try:
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


@app.route('/health', methods=['GET'])
def health_check():
    logger.info("Health check requested")
    return jsonify({
        'status': 'healthy',
        'message': 'AI Chess Game Server is running',
        'endpoints': ['/api/classic/ai-move']
    })


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