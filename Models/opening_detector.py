from collections import namedtuple
import csv
import logging

logger = logging.getLogger(__name__)

Opening = namedtuple('Opening', ['eco', 'name', 'moves'])


def parse_pgn(pgn: str) -> list[str]:
    """Parse a PGN string into a list of SAN moves, ignoring move numbers."""
    try:
        moves = []
        parts = pgn.split()
        for part in parts:
            if '.' in part:  # Skip move numbers like "1." or "2."
                continue
            if part and not part.startswith('{') and not part.endswith('}'):  # Skip comments
                moves.append(part)
        return moves
    except Exception as e:
        logger.error(f"Error parsing PGN '{pgn}': {e}")
        return []


def load_openings(file_paths: list[str]) -> list[Opening]:
    """Load opening data from TSV files into a list of Opening objects."""
    openings = []
    for file_path in file_paths:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f, delimiter='\t', fieldnames=['eco', 'name', 'pgn'])
                next(reader)  # Skip header row if present
                for row_num, row in enumerate(reader, start=2):
                    try:
                        eco = row.get('eco', '').strip()
                        name = row.get('name', '').strip()
                        pgn = row.get('pgn', '').strip()
                        
                        if not all([eco, name, pgn]):
                            logger.warning(f"Skipping incomplete row {row_num} in {file_path}")
                            continue
                            
                        moves = parse_pgn(pgn)
                        if moves:
                            openings.append(Opening(eco, name, moves))
                        else:
                            logger.warning(f"No valid moves found in row {row_num} of {file_path}")
                    except Exception as e:
                        logger.error(f"Error processing row {row_num} in {file_path}: {e}")
                        continue
                        
            logger.info(f"Loaded {len([o for o in openings if o.eco.startswith(eco[:1])])} openings from {file_path}")
        except FileNotFoundError:
            logger.error(f"Opening file not found: {file_path}")
        except Exception as e:
            logger.error(f"Error loading openings from {file_path}: {e}")
    
    logger.info(f"Total openings loaded: {len(openings)}")
    return openings


def detect_opening(game_moves: list[str], openings: list[Opening]) -> dict | None:
    """
    Detect the opening from a list of moves.
    
    Args:
        game_moves: List of SAN moves played in the game
        openings: List of Opening objects to search through
        
    Returns:
        Dictionary with opening info or None if no match found
    """
    if not game_moves or not openings:
        return None
        
    try:
        # Find openings that match the game moves
        matching_openings = []
        for opening in openings:
            if len(opening.moves) <= len(game_moves):
                if opening.moves == game_moves[:len(opening.moves)]:
                    matching_openings.append(opening)
        
        if not matching_openings:
            return None
            
        # Select the longest matching opening
        longest_opening = max(matching_openings, key=lambda x: len(x.moves))
        
        return {
            'eco': longest_opening.eco,
            'name': longest_opening.name,
            'moves_played': len(longest_opening.moves),
            'total_moves': len(game_moves),
            'is_complete': len(longest_opening.moves) == len(game_moves)
        }
    except Exception as e:
        logger.error(f"Error detecting opening: {e}")
        return None


def get_opening_statistics(openings: list[Opening]) -> dict:
    """Get statistics about loaded openings."""
    if not openings:
        return {}
        
    eco_codes = {}
    move_counts = []
    
    for opening in openings:
        # Count ECO codes
        eco_prefix = opening.eco[:1] if opening.eco else 'Unknown'
        eco_codes[eco_prefix] = eco_codes.get(eco_prefix, 0) + 1
        
        # Track move counts
        move_counts.append(len(opening.moves))
    
    return {
        'total_openings': len(openings),
        'eco_distribution': eco_codes,
        'avg_moves': sum(move_counts) / len(move_counts) if move_counts else 0,
        'min_moves': min(move_counts) if move_counts else 0,
        'max_moves': max(move_counts) if move_counts else 0
    }