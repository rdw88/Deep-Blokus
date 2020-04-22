from unittest import TestCase, main
from unittest.mock import patch, MagicMock
from multiprocessing import Event, Queue

import blokus
import ai
import numpy as np


class TestAi(ai.Ai):
    TEST_MOVE = patch('blokus.Move')

    def __init__(self):
        super().__init__()
        self.return_move = True


    def next_move(self, player):
        return TestAi.TEST_MOVE if self.return_move else None



class AiBaseClassTestCase(TestCase):
    @patch('blokus.Board')
    def setUp(self, board):
        self.test_board = board
        self.test_board.players = [ 
            blokus.RedPlayer(),
            blokus.BluePlayer(),
            blokus.GreenPlayer(),
            blokus.YellowPlayer()
        ]


    def test_initialize(self):
        test_ai = TestAi()
        
        self.assertIsNotNone(test_ai.board)
        self.assertEqual(test_ai.turn_count, 0)
        self.assertSetEqual(test_ai.finished_players, set([]))
        self.assertFalse(test_ai.is_complete)


    def test_reinitialize(self):
        test_ai = TestAi()

        current_board = test_ai.board
        test_ai.turn_count = 12
        test_ai.finished_players.add(current_board.players[0])
        test_ai.is_complete = True

        test_ai.reinitialize()

        self.assertNotEqual(current_board, test_ai.board)
        self.assertEqual(test_ai.turn_count, 0)
        self.assertSetEqual(test_ai.finished_players, set([]))
        self.assertFalse(test_ai.is_complete)


    def test_next_turn(self):
        test_ai = TestAi()
        test_ai.board = self.test_board

        test_ai.next_turn()

        self.assertEqual(test_ai.turn_count, 1)
        
        self.test_board.make_move.assert_called_once_with(TestAi.TEST_MOVE)
        self.test_board.make_move.reset_mock()

        test_ai.return_move = False

        test_ai.next_turn()

        self.assertEqual(test_ai.turn_count, 2)

        self.test_board.make_move.assert_not_called()

        self.assertSetEqual(test_ai.finished_players, set([self.test_board.players[1]]))

        test_ai.finished_players = set(test_ai.board.players)
        test_ai.next_turn()

        self.assertTrue(test_ai.is_complete)


    def test_end_game(self):
        test_ai = TestAi()
        test_ai.end_game()
        self.assertTrue(test_ai.is_complete)


    def test_get_turn(self):
        test_ai = TestAi()
        test_ai.board = self.test_board

        self.assertEqual(test_ai.get_turn(), 0)
        test_ai.next_turn()

        self.assertEqual(test_ai.get_turn(), 1)
        test_ai.next_turn()

        self.assertEqual(test_ai.get_turn(), 2)
        test_ai.next_turn()

        self.assertEqual(test_ai.get_turn(), 3)
        test_ai.next_turn()

        self.assertEqual(test_ai.get_turn(), 0)
        test_ai.next_turn()



class LSTMAiTestCase(TestCase):
    @patch('blokus.Board')
    def setUp(self, board):
        self.test_board = board
        self.test_board.players = [
            blokus.RedPlayer(),
            blokus.BluePlayer(),
            blokus.GreenPlayer(),
            blokus.YellowPlayer()
        ]


    def test_initialize(self):
        lstm_ai = ai.LSTMAi()

        self.assertIsNotNone(lstm_ai.model)


    def test_reinitialize(self):
        lstm_ai = ai.LSTMAi()
        
        model = lstm_ai.model
        lstm_ai.encoded_boards.append([1, 2, 3])

        lstm_ai.reinitialize()

        self.assertEqual(lstm_ai.model, model)
        self.assertListEqual(lstm_ai.encoded_boards, [])


    def test_next_move(self):
        lstm_ai = ai.LSTMAi()

        expected_valid_moves = lstm_ai.board.players[0].get_valid_moves(lstm_ai.board)
        expected_move_values = lstm_ai.get_predictions_for_turn(
            lstm_ai.get_predictions(expected_valid_moves)
        ).tolist()

        max_value = max(expected_move_values)
        expected_result = expected_valid_moves[expected_move_values.index(max_value)]

        move = lstm_ai.next_move(lstm_ai.board.players[0])
        self.assertEqual(move, expected_result)

        lstm_ai.next_turn()

        self.assertListEqual(lstm_ai.encoded_boards, [])

        current_board = lstm_ai.board.get_encoded_board().tolist()

        lstm_ai.board.players[1].unplayed_pieces = []
        move = lstm_ai.next_move(lstm_ai.board.players[1])

        self.assertIsNone(move)
        self.assertEqual(len(lstm_ai.encoded_boards), 1)
        self.assertListEqual(lstm_ai.encoded_boards[0].tolist(), current_board)

        lstm_ai.next_turn()

        self.assertSetEqual(lstm_ai.finished_players, set([lstm_ai.board.players[1]]))


    def test_get_predictions(self):
        lstm_ai = ai.LSTMAi()

        valid_moves = lstm_ai.board.players[0].get_valid_moves(lstm_ai.board)

        predictions = lstm_ai.get_predictions(valid_moves)

        self.assertIsNotNone(predictions)
        self.assertTupleEqual(predictions.shape, (len(valid_moves), 4))


    def test_get_predictions_for_turn(self):
        lstm_ai = ai.LSTMAi()        

        for i in range(6):
            self._verify_prediction_for_turn(lstm_ai)
            lstm_ai.next_turn()


    def _verify_prediction_for_turn(self, lstm_ai):
        valid_moves = lstm_ai.board.players[lstm_ai.get_turn()].get_valid_moves(lstm_ai.board)
        predictions = lstm_ai.get_predictions(valid_moves)

        turn_predictions = lstm_ai.get_predictions_for_turn(predictions)

        self.assertEqual(len(turn_predictions), len(valid_moves))

        for i, prediction in enumerate(turn_predictions):
            self.assertEqual(prediction, predictions[i][lstm_ai.get_turn()])


    def test_get_encoded_valid_moves(self):
        lstm_ai = ai.LSTMAi()

        piece = blokus.Piece(np.array([[ 1 ]]), lstm_ai.board.players[0])

        valid_moves = [ blokus.Move(lstm_ai.board, piece, (0, 0)) ]

        encoded_moves = lstm_ai.get_encoded_valid_moves(valid_moves)

        self.assertIsNotNone(encoded_moves)
        self.assertEqual(len(encoded_moves), 1)

        expected_encoded = [0, 1, 0, 0, 0] + ([1, 0, 0, 0, 0] * (blokus.Board.SIZE ** 2 - 1))

        self.assertListEqual(expected_encoded, encoded_moves[0].tolist())


    def test_is_training(self):
        lstm_ai = ai.LSTMAi()
        
        self.assertFalse(lstm_ai.is_training())

        event = Event()
        event.set()

        lstm_ai.train(Queue(), event)

        self.assertFalse(lstm_ai.is_training())

        event.clear()

        self.assertTrue(lstm_ai.is_training())



if __name__ == '__main__':
    main(warnings='ignore')