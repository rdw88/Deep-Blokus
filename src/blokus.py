import json
import numpy as np
import abc

from PIL import Image


class Board:
    SIZE = 20

    def __init__(self):
        self.players = [
            RedPlayer(),
            BluePlayer(),
            GreenPlayer(),
            YellowPlayer()
        ]

        self.board = [[Block((x, y), self.players) for x in range(Board.SIZE)] for y in range(Board.SIZE)]

        self.get_block((0, 0)).set_anchor(self.players[0], True)
        self.get_block((19, 0)).set_anchor(self.players[1], True)
        self.get_block((19, 19)).set_anchor(self.players[2], True)
        self.get_block((0, 19)).set_anchor(self.players[3], True)


    def make_move(self, move):
        if not move.is_valid():
            raise RuntimeError('Invalid move!')

        blocks_affected = move.get_blocks_affected()

        for block in blocks_affected:
            block.set_player(move.player)

            invalid_positions = [(-1, 0), (1, 0), (0, 1), (0, -1)]
            valid_positions = [(-1, -1), (1, -1), (1, 1), (-1, 1)]

            for position in invalid_positions + valid_positions:
                adjacent_block = self.get_block((position[0] + block.x, position[1] + block.y))

                if not adjacent_block or adjacent_block in blocks_affected:
                    continue

                if position in invalid_positions:
                    adjacent_block.disallow_player(move.player)
                else:
                    adjacent_block.set_anchor(move.player, True)

        move.player.play_piece(move._piece)


    def get_block(self, position):
        if position[0] < 0 or position[1] < 0 or position[0] >= Board.SIZE or position[1] >= Board.SIZE:
            return None

        return self.board[position[1]][position[0]]


    def get_image(self):
        image = Image.new(mode='RGB', size=(Board.SIZE * 40, Board.SIZE * 40))

        for y in range(len(self.board)):
            for x in range(len(self.board[y])):
                image_position = (x * 40, y * 40)
                image.paste(self.board[y][x].get_image(), image_position)

        return image


class Move:
    def __init__(self, board, piece, position, orientation=0):
        if not piece.get_anchors(orientation):
            raise ValueError(f'Orientation {orientation} is not valid for the provided piece!')

        self._board = board
        self._piece = piece
        self._position = position
        self._orientation = orientation


    def get_piece_matrix(self):
        return self._piece.orientations[self._orientation]


    def is_valid(self):
        if self._piece.is_played():
            return False

        is_anchored = False

        for block in self.get_blocks_affected():
            if not block or not block.allows_player(self.player):
                return False

            if block.anchors_player(self.player):
                is_anchored = True

        return is_anchored


    def get_blocks_affected(self):
        matrix = self.get_piece_matrix()

        return [
            self._board.get_block((x + self.x, y + self.y))
            for y in range(len(matrix))
            for x in range(len(matrix[y]))
            if matrix[y][x] == 1
        ]


    def get_anchors(self):
        return self._piece.get_anchors(orientation=self._orientation)


    @property
    def x(self):
        return self._position[0]


    @property    
    def y(self):
        return self._position[1]


    @property
    def player(self):
        return self._piece.player



class Piece:
    def __init__(self, raw_piece, player):
        self.player = player
        self.orientations = self._init_orientations([ raw_piece ])
        self._anchors = [Piece._get_anchors_for_piece(orientation) for orientation in self.orientations]

    
    def _init_orientations(self, pieces, rotation=1):
        if rotation == 4:
            # Cover the case where the original piece passed in needs to be flipped
            flipped = np.rot90(pieces[0].transpose(), -1)
            return pieces + ([ flipped ] if not np.array_equal(flipped, pieces[0]) else [])

        rotated = np.rot90(pieces[0], 4 - rotation)
        pieces += [ rotated ] if not any([np.array_equal(rotated, piece) for piece in pieces]) else []
        
        flipped = np.rot90(rotated.transpose(), -1)
        pieces += [ flipped ] if not any([np.array_equal(flipped, piece) for piece in pieces]) else []

        return self._init_orientations(pieces, rotation + 1)


    @staticmethod
    def get_all_pieces(player):
        with open('resources/pieces.json', 'r') as f:
            raw_pieces = json.loads(''.join(f.readlines()))
            f.close()

        return [Piece(np.array(raw_piece), player) for raw_piece in raw_pieces]


    @staticmethod
    def _get_value_of_block(piece, position):
        x = position[0]
        y = position[1]
        width = len(piece[0])
        height = len(piece)

        return piece[y][x] if 0 <= x < width and 0 <= y < height else 0


    @staticmethod
    def _is_anchor(piece, anchor):
        x = anchor[0]
        y = anchor[1]

        if piece[y][x] == 0:
            return False

        right  = Piece._get_value_of_block(piece, (x + 1, y))
        left   = Piece._get_value_of_block(piece, (x - 1, y))
        top    = Piece._get_value_of_block(piece, (x, y - 1))
        bottom = Piece._get_value_of_block(piece, (x, y + 1))

        return not ((right == left == 1) or (top == bottom == 1))


    @staticmethod
    def _get_anchors_for_piece(piece):
        return [
            (x, y) for y in range(len(piece)) for x in range(len(piece[y]))
            if Piece._is_anchor(piece, (x, y))
        ]


    def get_anchors(self, orientation):
        if orientation >= len(self._anchors) or orientation < 0:
            return None

        return self._anchors[orientation]


    def get_orientation(self, index):
        if index >= len(self.orientations) or index < 0:
            return None

        return self.orientations[index]


    def get_orientation_count(self):
        return len(self.orientations)


    def get_size(self):
        return sum([block for block in self.orientations[0].flatten()])


    def is_played(self):
        return self not in self.player.unplayed_pieces


    def __str__(self):
        piece_string = str()
        piece = self.orientations[0]

        for y in range(len(piece)):
            for x in range(len(piece[y])):
                piece_string += self.player.get_code() if piece[y][x] == 1 else '0'

            piece_string += '\n'

        return piece_string



class Block:
    def __init__(self, position, players):
        self.position = position
        self.player = None
        self.default_image = Image.open('resources/block.png')
        
        self._allows_player = { player: True for player in players }
        self._anchors = { player: False for player in players }

        for player in players:
            player.set_block_allowed(self, True)


    def disallow_player(self, player):
        if not self.allows_player(player):
            return

        self.set_anchor(player, False)
        self._allows_player[player] = False
        player.set_block_allowed(self, False)


    def allows_player(self, player):
        return not self.player and self._allows_player[player]


    def set_anchor(self, player, anchor):
        if anchor and not self.allows_player(player):
            return

        self._anchors[player] = anchor
        player.set_anchor(self, anchor)


    def anchors_player(self, player):
        return not self.player and self._anchors[player]


    def set_player(self, player):
        if not self.allows_player(player):
            raise RuntimeError(f'Player not allowed on block {self.position}!')

        self.disallow_player(player)
        self.player = player


    def get_image(self):
        return self.player.get_image() if self.player else self.default_image


    def __str__(self):
        return self.player.get_code() if self.player else '0'


    @property
    def x(self):
        return self.position[0]


    @property
    def y(self):
        return self.position[1]


class Player(abc.ABC):
    def __init__(self):
        self.unplayed_pieces = Piece.get_all_pieces(self)
        self.allowed_blocks = set()
        self.anchors = set()
        self._image = None


    @abc.abstractmethod
    def get_code(self):
        pass


    @abc.abstractmethod
    def _get_image_path(self):
        pass


    def get_valid_moves(self, board, pieces=None):
        if not pieces:
            pieces = self.unplayed_pieces

        valid_moves = list()

        for anchor in self.anchors:
            for piece in pieces:
                for i in range(piece.get_orientation_count()):
                    for piece_anchor in piece.get_anchors(orientation=i):
                        position_offset = (
                            anchor.position[0] - piece_anchor[0],
                            anchor.position[1] - piece_anchor[1]
                        )

                        move = Move(board, piece, position_offset, orientation=i)
                        if move.is_valid():
                            valid_moves.append(move)

        return valid_moves


    def get_image(self):
        if not self._image:
            self._image = Image.open(self._get_image_path())

        return self._image


    def set_block_allowed(self, block, is_allowed):
        if is_allowed:
            self.allowed_blocks.add(block)
        elif block in self.allowed_blocks:
            self.allowed_blocks.remove(block)
            self.set_anchor(block, False)


    def set_anchor(self, block, is_anchor):
        if is_anchor:
            self.anchors.add(block)
        elif block in self.anchors:
            self.anchors.remove(block)


    def play_piece(self, piece):
        assert piece.player == self
        assert piece in self.unplayed_pieces

        self.unplayed_pieces.remove(piece)


    def get_score(self):
        return sum([piece.get_size() for piece in self.unplayed_pieces])


class RedPlayer(Player):
    def get_code(self):
        return 'R'


    def _get_image_path(self):
        return 'resources/red.png'


class BluePlayer(Player):
    def get_code(self):
        return 'B'


    def _get_image_path(self):
        return 'resources/blue.png'


class GreenPlayer(Player):
    def get_code(self):
        return 'G'


    def _get_image_path(self):
        return 'resources/green.png'


class YellowPlayer(Player):
    def get_code(self):
        return 'Y'


    def _get_image_path(self):
        return 'resources/yellow.png'