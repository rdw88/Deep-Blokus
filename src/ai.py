import os

# Disable tensorflow warnings regarding AVX support on CPU. These warnings
# don't apply since we are using the GPU.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np

import blokus
import abc
import random
import statistics
import argparse
import time
import socket
import json
import struct
import sys
import traceback
import models

from threading import Thread
from timer import timer, print_results

from multiprocessing import Process, Pool


class Ai(abc.ABC):
    def __init__(self):
        self.reinitialize()


    def reinitialize(self):
        self.board = blokus.Board()
        self.turn_count = 0
        self.finished_players = set()
        self.is_complete = False


    @abc.abstractmethod
    def next_move(self, player):
        pass


    def play_game(self):
        while not self.is_complete:
            self.next_turn()


    def play_games(self, num_games=0):
        games_played = 0
        play_indefinitely = num_games == 0

        while play_indefinitely or games_played < num_games:
            self.play_game()
            self.reinitialize()
            games_played += 1


    def next_turn(self):
        if len(self.finished_players) == 4:
            self.end_game()
            return

        player = self.board.players[self.get_turn()]
        move = self.next_move(player)

        self.turn_count += 1

        if move:
            self.board.make_move(move)
        else:
            self.finished_players.add(player)


    def end_game(self):
        self.is_complete = True
        blokus.Move._move_cache.clear()


    def get_turn(self):
        return self.turn_count % len(self.board.players)



'''
Each player makes a random move on each turn.
'''
class RandomAi(Ai):
    def next_move(self, player):
        valid_moves = player.get_valid_moves(self.board)

        if len(valid_moves) > 0:
            return valid_moves[random.randint(0, len(valid_moves) - 1)]


class LSTMAi(Ai):
    def __init__(self):
        super().__init__()
        self.model = models.blokus_model()


    def reinitialize(self):
        super().reinitialize()
        self.encoded_boards = list()


    def next_move(self, player):
        if self.turn_count > 0:
            self.encoded_boards.append(self.board.get_encoded_board().tolist())

        valid_moves = player.get_valid_moves(self.board)
        if len(valid_moves) == 0:
            return None

        return valid_moves[random.randint(0, len(valid_moves) - 1)]

        # TODO: Playing live with updated model
        '''
        valid_move_values = self.get_predictions_for_turn(
            self.get_predictions(valid_moves)
        )

        return valid_moves[np.argmax(valid_move_values)]
        '''


    def get_predictions(self, valid_moves):
        return self.model(
            self.get_encoded_valid_moves(valid_moves)
        ).numpy()


    def get_predictions_for_turn(self, predictions):
        return np.array([
            prediction[self.get_turn()] for prediction in predictions
        ])


    def get_encoded_valid_moves(self, valid_moves):
        encoded_valid_moves = list()

        for move in valid_moves:
            self.board.fake_move(move)
            encoded_valid_moves.append(self.board.get_encoded_board(fake=True))

        return np.array(encoded_valid_moves)


    def train(self, trainer_client):
        self._training = True
        self._trainer_client = trainer_client

        self.play_games()


    def is_training(self):
        return hasattr(self, '_training') and self._training


    def end_game(self):
        super().end_game()

        if not self.is_training():
            return

        scores = [player.get_score() for player in self.board.players]

        # We will not train on games that resulted in ties
        if len(set(scores)) == 4:
            is_model_available = self._trainer_client.send_training_example(self.encoded_boards, scores)

            if is_model_available:
                del self.model

                self.model = tf.keras.models.load_model(self._trainer_client.remote_model_path)



def start_worker(trainer_address, trainer_port):
    sys.stdout = open(f'logs/worker_log_{os.getpid()}.out', 'a')
    sys.stderr = open(f'logs/worker_error_{os.getpid()}.out', 'a')

    from trainer import TrainerClient

    client = TrainerClient(trainer_address, trainer_port)

    try:
        LSTMAi().train(client)
    except:
        print(traceback.format_exc(), flush=True, file=sys.stderr)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Run worker processes that play games of Blokus for a trainer.')
    parser.add_argument('--address', action='store', required=True, help='The IP address of the trainer.')
    parser.add_argument('--port', action='store', type=int, required=True, help='The port number the trainer is running on.')
    parser.add_argument('--num-workers', action='store', type=int, help='The number of worker processes. Defaults to os.cpu_count()')

    args = parser.parse_args()

    num_workers = args.num_workers if args.num_workers else os.cpu_count()

    existing_logs = [file_name for file_name in os.listdir('logs/') if os.path.splitext(file_name)[1] == '.out']
    for existing_log in existing_logs:
        os.remove(f'logs/{existing_log}')

    print(f'Initializing pool of {num_workers} workers... ', end='')

    pool = Pool(processes=num_workers)

    for i in range(num_workers):
        pool.apply_async(start_worker, args=(args.address, args.port))

    print('done!')

    try:
        while True:
            time.sleep(100)
    except KeyboardInterrupt:
        pool.terminate()
        pool.join()