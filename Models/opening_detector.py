from collections import namedtuple
import csv

Opening = namedtuple('Opening', ['eco', 'name', 'moves'])


def parse_pgn(pgn: str) -> list[str]:
    """Parse a PGN string into a list of SAN moves, ignoring move numbers."""
    moves = []
    parts = pgn.split()
    for part in parts:
        if '.' in part:  # Skip move numbers like "1." or "2."
            continue
        moves.append(part)
    return moves


def load_openings(file_paths: list[str]) -> list[Opening]:
    """Load opening data from TSV files into a list of Opening objects."""
    openings = []
    for file_path in file_paths:
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter='\t', fieldnames=['eco', 'name', 'pgn'])
            next(reader)  # Skip header row if present
            for row in reader:
                eco = row['eco']
                name = row['name']
                pgn = row['pgn']
                moves = parse_pgn(pgn)
                openings.append(Opening(eco, name, moves))
    return openings


def detect_opening(game_moves: list[str], openings: list[Opening]) -> dict | None:
    """Detect the most specific opening that matches the game move sequence."""
    matching_openings = []
    for opening in openings:
        if len(opening.moves) > len(game_moves):
            continue
        if opening.moves == game_moves[:len(opening.moves)]:
            matching_openings.append(opening)

    if not matching_openings:
        return None

    # Select the opening with the longest move sequence
    max_length = max(len(op.moves) for op in matching_openings)
    longest_openings = [op for op in matching_openings if len(op.moves) == max_length]
    selected_opening = longest_openings[0]  # Pick the first if multiple match

    return {'eco': selected_opening.eco, 'name': selected_opening.name}