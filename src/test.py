from unittest import TestCase, main
from unittest.mock import patch, Mock

import blokus
import random
import os

import numpy as np



class BoardTestCase(TestCase):
    def test_initialization(self):
        board = blokus.Board()

        self.assertEqual(len(board.players), 4)
        self.assertEqual(blokus.Board.SIZE, 20)
        self.assertEqual(len(board.board), blokus.Board.SIZE)
        self.assertTrue(all([len(row) == blokus.Board.SIZE for row in board.board]))

        self.assertTrue(board.get_block((0, 0)).anchors_player(board.players[0]))
        self.assertTrue(board.get_block((19, 0)).anchors_player(board.players[1]))
        self.assertTrue(board.get_block((19, 19)).anchors_player(board.players[2]))
        self.assertTrue(board.get_block((0, 19)).anchors_player(board.players[3]))


    def test_make_move(self):
        board = blokus.Board()

        with patch('blokus.Move') as invalid_move:
            invalid_move.is_valid.return_value = False

            with self.assertRaises(RuntimeError):
                board.make_move(invalid_move)

        player = board.players[2]
        piece = blokus.Piece(np.array([
            [1, 1, 1],
            [0, 1, 0],
            [0, 1, 0]
        ]), player)

        player.unplayed_pieces = [ piece ]

        # Player 2's start position is (19, 19)
        # [0, 0, 1]
        # [1, 1, 1]
        # [0, 0, 1]
        move = blokus.Move(board, piece, (17, 17), orientation=1)
        board.make_move(move)

        self.assertNotIn(move._piece, player.unplayed_pieces)

        self._verify_make_move(
            board=board,
            player=player,
            expected_assigned=set([ (19, 17), (19, 18), (19, 19), (17, 18), (18, 18) ]),
            expected_anchors=set([ (18, 16), (16, 17), (16, 19) ]),
            expected_disallowed=set([ (19, 16), (18, 17), (17, 17), (18, 19), (17, 19), (16, 18) ])
        )

        piece_2 = blokus.Piece(np.array([
            [1, 1],
            [1, 0],
            [1, 1]
        ]), player)

        player.unplayed_pieces = [ piece_2 ]

        move = blokus.Move(board, piece_2, (15, 17))
        board.make_move(move)

        self._verify_make_move(
            board=board,
            player=player,
            expected_assigned=set([
                (19, 17), (19, 18), (19, 19), (17, 18), (18, 18),
                (16, 17), (15, 17), (15, 18), (15, 19), (16, 19)
            ]),
            expected_anchors=set([
                (18, 16), (17, 16), (14, 16)
            ]),
            expected_disallowed=set([
                (19, 16), (18, 17), (17, 17), (18, 19), (17, 19), (16, 18),
                (14, 19), (14, 18), (14, 17), (15, 16), (16, 16)
            ])
        )


    def _verify_make_move(self, board, player, expected_assigned, expected_anchors, expected_disallowed):
        assigned_blocks = set([block.position for row in board.board for block in row if block.player == player])

        self.assertEqual(assigned_blocks, expected_assigned)

        anchors = set([block.position for row in board.board for block in row if block.anchors_player(player)])

        self.assertEqual(anchors, expected_anchors)
        self.assertEqual(anchors, set([block.position for block in player.anchors]))

        expected_disallowed = expected_disallowed.union(expected_assigned)
        disallowed = set([block.position for row in board.board for block in row if not block.allows_player(player)])

        self.assertEqual(disallowed, expected_disallowed)

        allowed = set([block for row in board.board for block in row if block.allows_player(player)])
        self.assertEqual(allowed, player.allowed_blocks)


    def test_get_block(self):
        board = blokus.Board()

        self.assertIsNone(board.get_block((-1, 0)))
        self.assertIsNone(board.get_block((0, -1)))
        self.assertIsNone(board.get_block((blokus.Board.SIZE, 0)))
        self.assertIsNone(board.get_block((0, blokus.Board.SIZE)))

        self.assertEqual(board.get_block((5, 9)), board.board[9][5])



class MoveTestCase(TestCase):
    @patch('blokus.Board')
    @patch('blokus.Player')
    def test_initialization(self, player, board):
        piece = blokus.Piece(np.array([[ 1 ]]), player)

        with self.assertRaises(ValueError):
            # The above piece only has 1 valid orientation
            move = blokus.Move(board, piece, (0, 0), orientation=1)

        move = blokus.Move(board, piece, (5, 3), orientation=0)

        self.assertEqual(move._board, board)
        self.assertEqual(move._piece, piece)
        self.assertEqual(move._position, (5, 3))
        self.assertEqual(move._orientation, 0)
        self.assertEqual(move.x, 5)
        self.assertEqual(move.y, 3)
        self.assertEqual(move.player, player)


    @patch('blokus.Board')
    @patch('blokus.Player')
    def test_get_piece_matrix(self, player, board):
        piece = blokus.Piece(np.array([
            [1, 1, 1],
            [1, 0, 0],
            [1, 0, 0]
        ]), player)

        move = blokus.Move(board, piece, (10, 15), orientation=3)
        matrix = move.get_piece_matrix()

        expected_matrix = np.array([
            [1, 0, 0],
            [1, 0, 0],
            [1, 1, 1],
        ])

        self.assertTrue(np.array_equal(matrix, expected_matrix))


    def test_is_valid(self):
        '''
        A valid move is a move where:
            1. At least 1 block for the move is an anchor for the player
            2. All blocks allow the player
            3. The piece isn't falling out of bounds
        '''
        board = blokus.Board()
        player = board.players[0]

        piece = blokus.Piece(np.array([
            [1, 1],
            [1, 0]
        ]), player)

        player.unplayed_pieces.append(piece)

        move = blokus.Move(board, piece, (0, 0))
        self.assertTrue(move.is_valid())

        # Invalid orientation, does not line up with anchor
        # [0, 1]
        # [1, 1]
        move = blokus.Move(board, piece, (0, 0), orientation=2)
        self.assertFalse(move.is_valid())

        # Piece is falling off the board
        move = blokus.Move(board, piece, (0, -1))
        self.assertFalse(move.is_valid())
        move = blokus.Move(board, piece, (-1, 0))
        self.assertFalse(move.is_valid())

        board.get_block((1, 0)).disallow_player(player)

        # Piece covers a block where the player is not allowed
        move = blokus.Move(board, piece, (0, 0))
        self.assertFalse(move.is_valid())

        # Piece has already been played
        with patch('blokus.Piece') as played_piece:
            played_piece.is_played.return_value = True
            move = blokus.Move(board, played_piece, (0, 0))
            self.assertFalse(move.is_valid())



    def test_get_blocks_affected(self):
        board = blokus.Board()
        player = board.players[0]

        piece = blokus.Piece(np.array([
            [1, 1, 1],
            [0, 1, 0],
            [0, 1, 0]
        ]), player)

        move = blokus.Move(board, piece, (13, 6))
        expected = [
            board.get_block((13, 6)),
            board.get_block((14, 6)),
            board.get_block((15, 6)),
            board.get_block((14, 7)),
            board.get_block((14, 8))
        ]

        self.assertEqual(set(move.get_blocks_affected()), set(expected))

        # [1, 0, 0]
        # [1, 1, 1]
        # [1, 0, 0]
        move = blokus.Move(board, piece, (5, 9), orientation=2)
        expected = [
            board.get_block((5, 9)),
            board.get_block((5, 10)),
            board.get_block((6, 10)),
            board.get_block((7, 10)),
            board.get_block((5, 11))
        ]

        self.assertEqual(set(move.get_blocks_affected()), set(expected))

        move = blokus.Move(board, piece, (-1, 0))
        expected = [
            None,
            board.get_block((0, 0)),
            board.get_block((1, 0)),
            board.get_block((0, 1)),
            board.get_block((0, 2))
        ]

        self.assertEqual(set(move.get_blocks_affected()), set(expected))



class PieceTestCase(TestCase):
    @patch('blokus.Player')
    def test_initialization(self, player):
        raw_piece = np.array([
            [1, 0],
            [1, 1],
            [1, 0],
            [1, 0]
        ])

        piece = blokus.Piece(raw_piece, player)

        self.assertEqual(piece.get_orientation_count(), 8)
        self.assertEqual(piece.player, player)
        self.assertEqual(len(piece._anchors), 8)

        rotated_once = np.array([
            [1, 1, 1, 1],
            [0, 0, 1, 0]
        ])

        self.assertTrue(np.array_equal(piece.orientations[1], rotated_once))

        expected_anchors = [ (0, 0), (3, 0), (2, 1) ]
        self._validate_anchors(rotated_once, expected_anchors)

        rotated_thrice_flipped = np.array([
            [1, 1, 1, 1],
            [0, 1, 0, 0]
        ])

        self.assertTrue(np.array_equal(piece.orientations[2], rotated_thrice_flipped))

        expected_anchors = [ (0, 0), (3, 0), (1, 1) ]
        self._validate_anchors(rotated_thrice_flipped, expected_anchors)

        raw_piece = np.array([
            [1, 1],
            [1, 1]
        ])

        piece = blokus.Piece(raw_piece, player)

        self.assertEqual(piece.get_orientation_count(), 1)
        self.assertEqual(len(piece._anchors), 1)
        self.assertEqual(len(piece.get_anchors(orientation=0)), 4)


    @patch('blokus.Player')
    def test_get_all_pieces(self, player):
        all_pieces = blokus.Piece.get_all_pieces(player)

        self.assertEqual(len(all_pieces), 21)

        piece_size_counts = dict()
        for piece in all_pieces:
            size = piece.get_size()
            current_count = piece_size_counts.get(size, 0)
            piece_size_counts[size] = current_count + 1

        self.assertEqual(piece_size_counts[1], 1)
        self.assertEqual(piece_size_counts[2], 1)
        self.assertEqual(piece_size_counts[3], 2)
        self.assertEqual(piece_size_counts[4], 5)
        self.assertEqual(piece_size_counts[5], 12)


    def test_get_anchors_for_piece(self):
        '''
        Valid anchors are defined by unique locations on the piece
        where another piece for the same player can be played at least
        on of its diagonal.
        '''
        piece = np.array([
            [0, 1, 1],
            [1, 1, 0],
            [1, 0, 0]
        ])

        expected_anchors = [ (1, 0), (2, 0), (1, 1), (0, 2), (0, 1) ]
        self._validate_anchors(piece, expected_anchors)

        piece = np.array([
            [1, 1],
            [1, 0],
            [1, 0],
            [1, 0]
        ])

        expected_anchors = [ (0, 0), (1, 0), (0, 3) ]
        self._validate_anchors(piece, expected_anchors)

        piece = np.array([
            [ 1, 1 ],
            [ 1, 0 ],
            [ 1, 1 ]
        ])

        expected_anchors = [ (0, 0), (1, 0), (0, 2), (1, 2) ]
        self._validate_anchors(piece, expected_anchors)

        piece = np.array([
            [ 1 ],
            [ 1 ],
            [ 1 ],
            [ 1 ],
            [ 1 ]
        ])

        expected_anchors = [ (0, 0), (0, 4) ]
        self._validate_anchors(piece, expected_anchors)


    def _validate_anchors(self, piece, expected_anchors):
        anchors = blokus.Piece._get_anchors_for_piece(piece)

        self.assertEqual(len(anchors), len(expected_anchors))
        for anchor in anchors:
            self.assertIn(anchor, expected_anchors)


    def test_is_anchor(self):
        piece = np.array([
            [1, 1, 1],
            [1, 0, 0],
            [1, 0, 0]
        ])

        self.assertFalse(blokus.Piece._is_anchor(piece, (1, 1)))
        self.assertFalse(blokus.Piece._is_anchor(piece, (2, 2)))
        self.assertFalse(blokus.Piece._is_anchor(piece, (0, 1)))
        self.assertFalse(blokus.Piece._is_anchor(piece, (1, 0)))
        self.assertTrue(blokus.Piece._is_anchor(piece, (0, 0)))
        self.assertTrue(blokus.Piece._is_anchor(piece, (2, 0)))
        self.assertTrue(blokus.Piece._is_anchor(piece, (0, 2)))


    @patch('blokus.Player')
    def test_get_anchors(self, player):
        raw_piece = np.array([
            [0, 1, 1],
            [1, 1, 0],
            [0, 1, 0]
        ])

        piece = blokus.Piece(raw_piece, player)

        expected = [ (0, 1), (1, 0), (2, 0), (1, 2) ]
        anchors = piece.get_anchors(orientation=0)

        self.assertEqual(set(anchors), set(expected))

        expected = [ (0, 1), (1, 0), (1, 2), (2, 2) ]
        anchors = piece.get_anchors(orientation=4)

        self.assertEqual(set(anchors), set(expected))

        self.assertIsNone(piece.get_anchors(orientation=-1))
        self.assertIsNone(piece.get_anchors(orientation=8))
        self.assertIsNotNone(piece.get_anchors(orientation=7))


    def test_get_value_of_block(self):
        piece = np.array([
            [1, 1],
            [1, 0]
        ])

        self.assertEqual(blokus.Piece._get_value_of_block(piece, (0, 0)), 1)
        self.assertEqual(blokus.Piece._get_value_of_block(piece, (1, 1)), 0)
        self.assertEqual(blokus.Piece._get_value_of_block(piece, (-1, 0)), 0)
        self.assertEqual(blokus.Piece._get_value_of_block(piece, (0, -1)), 0)
        self.assertEqual(blokus.Piece._get_value_of_block(piece, (0, 2)), 0)
        self.assertEqual(blokus.Piece._get_value_of_block(piece, (2, 0)), 0)


    @patch('blokus.Player')
    def test_is_played(self, player):
        piece = blokus.Piece(np.array([
            [1, 1, 1]
        ]), player)

        player.unplayed_pieces = [ piece ]

        self.assertFalse(piece.is_played())

        player.unplayed_pieces.remove(piece)

        self.assertTrue(piece.is_played())


    @patch('blokus.Player')
    def test_get_size(self, player):
        raw_piece = np.array([
            [1, 1],
            [1, 0]
        ])

        piece = blokus.Piece(raw_piece, player)
        self.assertEqual(piece.get_size(), 3)

        raw_piece = np.array([
            [1, 0],
            [1, 1],
            [1, 0],
            [1, 0]
        ])

        piece = blokus.Piece(raw_piece, player)
        self.assertEqual(piece.get_size(), 5)


@patch('blokus.Player')
@patch('blokus.Player')
class BlockTestCase(TestCase):
    def test_initialization(self, player_1, player_2):
        players = [ player_1, player_2 ]
        block = blokus.Block((5, 3), players)

        self.assertEqual(block.position, (5, 3))
        self.assertIsNone(block.player)
        self.assertEqual(len(block._allows_player.keys()), 2)
        self.assertEqual(block.x, 5)
        self.assertEqual(block.y, 3)

        self.assertTrue(block._allows_player[player_1])
        self.assertTrue(block._allows_player[player_2])
        self.assertFalse(block._anchors[player_1])
        self.assertFalse(block._anchors[player_2])

        for player in players:
            player.set_block_allowed.assert_called_once_with(block, True)


    @patch('blokus.Player')
    def test_allows_player(self, player_1, player_2, player_3):
        players = [ player_1, player_2, player_3 ]
        block = blokus.Block((4, 9), players)

        self.assertTrue(all([block.allows_player(player) for player in players]))

        block.disallow_player(player_1)

        self.assertFalse(block.allows_player(player_1))
        self.assertTrue(block.allows_player(player_2))
        self.assertTrue(block.allows_player(player_3))

        block.set_player(player_2)

        player_2.set_block_allowed.assert_called()

        self.assertTrue(all([not block.allows_player(player) for player in players]))


    @patch('blokus.Player')
    def test_anchors_player(self, player_1, player_2, player_3):
        players = [ player_1, player_2, player_3 ]
        block = blokus.Block((1, 1), players)

        self.assertTrue(all([not block.anchors_player(player) for player in players]))

        block.set_anchor(player_1, True)

        player_1.set_anchor.assert_called_once_with(block, True)

        self.assertTrue(block.anchors_player(player_1))
        self.assertFalse(block.anchors_player(player_2))
        self.assertFalse(block.anchors_player(player_3))

        block.disallow_player(player_2)

        self.assertFalse(block.anchors_player(player_2))

        player_2.reset_mock()

        block.set_anchor(player_2, True)
        player_2.set_anchor.assert_not_called()


    def test_set_player(self, player_1, player_2):
        player_2.get_code.return_value = 'B'
        player_2.get_image.return_value = 'IMAGE'

        players = [ player_1, player_2 ]
        block = blokus.Block((0, 0), players)

        self.assertIsNone(block.player)
        self.assertEqual(str(block), '0')
        
        block.set_player(player_2)

        self.assertEqual(block.player, player_2)
        self.assertEqual(str(block), 'B')
        self.assertEqual(block.get_image(), 'IMAGE')
        self.assertFalse(block.allows_player(player_2))
        self.assertFalse(block.allows_player(player_1))


class PlayerTestCase(TestCase):
    def test_initialization(self):
        player = blokus.RedPlayer()

        self.assertEqual(len(player.unplayed_pieces), 21)
        self.assertTrue(all([piece.player == player for piece in player.unplayed_pieces]))


    def test_get_valid_moves(self):
        board = blokus.Board()
        player = board.players[0]
        unplayed_piece = blokus.Piece(np.array([
            [1, 0],
            [1, 1],
            [1, 1]
        ]), player)

        player.unplayed_pieces = [ unplayed_piece ]

        valid_moves = player.get_valid_moves(board)
        self.assertEqual(len(valid_moves), 6)

        invalid_move = np.array([
            [0, 1],
            [1, 1],
            [1, 1]
        ])

        self.assertTrue(all([not np.array_equal(invalid_move, move) for move in valid_moves]))

        invalid_move = np.array([
            [0, 1, 1],
            [1, 1, 1]
        ])

        self.assertTrue(all([not np.array_equal(invalid_move, move) for move in valid_moves]))

        move_positions = map(lambda x: x._position, valid_moves)
        self.assertTrue(all([position == (0, 0) for position in move_positions]))


    def test_get_image(self):
        player = blokus.RedPlayer()
        image = player.get_image()

        self.assertIsNotNone(image)

        image = player.get_image()

        self.assertIsNotNone(image)


    def test_get_image_path(self):
        red = blokus.RedPlayer()
        green = blokus.GreenPlayer()
        blue = blokus.BluePlayer()
        yellow = blokus.YellowPlayer()

        self.assertEqual(red._get_image_path(), 'resources/red.png')
        self.assertEqual(green._get_image_path(), 'resources/green.png')
        self.assertEqual(blue._get_image_path(), 'resources/blue.png')
        self.assertEqual(yellow._get_image_path(), 'resources/yellow.png')


    def test_get_code(self):
        red = blokus.RedPlayer()
        green = blokus.GreenPlayer()
        blue = blokus.BluePlayer()
        yellow = blokus.YellowPlayer()

        self.assertEqual(red.get_code(), 'R')
        self.assertEqual(green.get_code(), 'G')
        self.assertEqual(blue.get_code(), 'B')
        self.assertEqual(yellow.get_code(), 'Y')



if __name__ == '__main__':
    main(warnings='ignore')