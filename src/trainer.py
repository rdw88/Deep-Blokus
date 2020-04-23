import tensorflow as tf
import numpy as np
import traceback
import sys
import os
import argparse
import json
import struct
import socket
import models
import datetime

from socketserver import ThreadingTCPServer, StreamRequestHandler

from multiprocessing import Process, Queue, Event
from threading import Thread
from pathlib import Path

from ai import LSTMAi



class BlokusGameRequestHandler(StreamRequestHandler):
    '''
    Handle a single connection from a worker thread playing Blokus games.
    The connection will remain open indefinitely until the trainer is done training,
    continuously receiving results of games played by the worker.

    Occasionally, we will send each worker an updated version of the model to use
    for future games.
    '''
    def handle(self):
        print('Connection to new worker established')

        games_received = 0
        trainer = self.server.trainer

        while True:
            try:
                raw_data = self.rfile.readline()

                if trainer.training_complete:
                    break

                data = json.loads(raw_data)
            except json.JSONDecodeError:
                if raw_data == b'':
                    break

                print(f'Invalid JSON in request: {raw_data}')
                continue

            if 'boards' not in data or 'scores' not in data:
                print('Invalid request, could not find boards or scores in request!')
                continue

            trainer.enqueue_game(data['boards'], data['scores'])

            games_received += 1

            if games_received % BlokusTrainer.MODEL_UPDATE_FREQUENCY == 0:
                model_path = f'models/training_model_{int(games_received / BlokusTrainer.MODEL_UPDATE_FREQUENCY)}.h5'

                trainer.model.save(model_path)

                with open(model_path, 'rb') as f:
                    data = struct.pack('i', os.path.getsize(model_path)) + f.read()
                    self.wfile.write(data)
                    f.close()
            else:
                self.wfile.write(bytes(struct.pack('i', 0)))


class TrainerClient:
    def __init__(self, server_address, server_port):
        self.connection = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.connection.connect((server_address, server_port))
        self.remote_model_path = 'models/remote_model.h5'


    '''
    Sends a training example to the trainer.
    
    Returns true if a new model is available to use from the trainer, false otherwise.
    '''
    def send_training_example(self, boards, scores):
        serialized_game = json.dumps({
            'boards': boards,
            'scores': scores
        }) + '\n'

        self.connection.sendall(bytes(serialized_game, 'utf-8'))

        response = self._get_response()
        if not response:
            return False

        with open(self.remote_model_path, 'wb') as f:
            f.write(response)

        print('Received updated model')
        return True


    def _get_response(self):
        response = bytes()
        content_length = struct.unpack('i', self.connection.recv(4))[0]

        if content_length == 0:
            return None

        while len(response) < content_length:
            response += self.connection.recv(65536)

        return response



class BlokusTrainer:
    # Number of games 
    MODEL_UPDATE_FREQUENCY = 100

    def __init__(self, output_file, batch_size=5, num_steps=32, num_epochs=1, validate_path=None, metrics_path_override=None):
        if not output_file:
            print('WARNING: No output file for model!')

        _, file_extension = os.path.splitext(output_file)
        if file_extension != '.h5':
            raise ValueError(f'Output file with extension .h5 expected, got "{file_extension}".')

        self.output_file = output_file
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.num_epochs = num_epochs
        self.validation_data = self._init_validation_data(validate_path) if validate_path else None
        self.metrics_path_override = metrics_path_override

        self.model = models.blokus_model()

        self.game_queue = Queue()
        self.server = None
        self.training_complete = False

        # Initiate the server on a new thread. Server will wait for incoming connections
        # from workers playing games of Blokus.
        Thread(target=self._start_server).start()


    def _start_server(self):
        host, port = 'localhost', 8888

        with ThreadingTCPServer((host, port), BlokusGameRequestHandler) as server:
            self.server = server
            server.trainer = self

            print(f'Server ready on port {port}')

            server.serve_forever()


    def _init_validation_data(self, validate_path):
        print(f'Loading validation data from {validate_path}...', end='', flush=True)

        if not os.path.exists(validate_path):
            raise ValueError(f'Path {validate_path} could not be found!')

        with open(validate_path, 'r') as f:
            raw_data = ''.join(f.readlines())

        data = json.loads(raw_data)

        if not isinstance(data, list):
            raise ValueError(f'Expected a list of training data in {validate_path} but got: {data.__class__}')

        inputs = list()
        outputs = list()

        for example in data:
            if 'boards' not in example or 'scores' not in example:
                raise ValueError(f'Invalid game data in file {validate_path}')

            moves = example['boards']
            scores = self.get_target(example['scores'])

            for move in moves:
                inputs.append(np.array(move))
                outputs.append(np.array(scores))

        print('done')

        return (np.array(inputs), np.array(outputs))


    def train(self):
        training_metrics_path = f'metrics/{datetime.datetime.now().strftime("%Y%m%d%H%M%S")}'.replace('/', os.sep)
        if self.metrics_path_override:
            training_metrics_path = self.metrics_path_override.replace('/', os.sep).replace('\\', os.sep)

        Path(training_metrics_path).mkdir(parents=True, exist_ok=True)
        
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=training_metrics_path)

        try:
            self.model.fit(
                x=self.get_batches(),
                steps_per_epoch=self.num_steps,
                epochs=self.num_epochs,
                validation_data=self.validation_data,
                callbacks=[tensorboard_callback]
            )
        except Exception as e:
            print(traceback.format_exc())
        finally:
            print('Saving model... ', end='')

            self.model.save(self.output_file)

            print('done!')

            if self.server:
                self.server.shutdown()
                self.training_complete = True
                print('Server shutdown')


    def enqueue_game(self, boards, scores):
        self.game_queue.put({
            'boards': boards,
            'scores': scores
        })


    '''
    Read the latest games from the queue until we have reached
    batch size. Each game can have an arbitrary number of moves,
    batch size is based on number of moves.
    '''
    def get_batches(self):
        inputs = list()
        outputs = list()

        while True:
            game = self.game_queue.get()

            target = self.get_target(game['scores'])
            boards = game['boards']

            examples_needed = self.batch_size - len(inputs)

            new_inputs = boards[:examples_needed]
            inputs += new_inputs
            outputs += ([ target ] * len(new_inputs))

            while len(inputs) == self.batch_size:
                yield (np.array(inputs), np.array(outputs))

                remaining_inputs = boards[examples_needed:]

                inputs = remaining_inputs[:self.batch_size]
                outputs = [ target ] * len(inputs)

                if len(remaining_inputs) >= self.batch_size:
                    examples_needed += self.batch_size


    def get_target(self, scores):
        min_score = min(scores)
        return [int(score == min_score) for score in scores]



if __name__ == '__main__':
    parser = argparse.ArgumentParser('Train a model for Blokus')
    parser.add_argument('-o', action='store', required=True, help='The output file of the resulting model.', dest='output_path')
    parser.add_argument('--batch-size', action='store', type=int, required=True, help='The batch size of the model.')
    parser.add_argument('--steps', action='store', type=int, required=True, help='The number of training examples per epoch.')
    parser.add_argument('--epochs', action='store', type=int, required=True, help='The number of epochs of training.')
    parser.add_argument('--validate', action='store', help='A path to a JSON file containing data to validate the performance of the model on.')
    parser.add_argument('--metrics', action='store', help='A directory name to store training metrics.')

    args = parser.parse_args()

    if not os.path.exists(os.path.dirname(args.output_path)):
        print('Output directory does not exist!')
        sys.exit(1)

    if os.path.exists(args.output_path):
        answer = input(f'The file {args.output_path} already exists and will be replaced. Do you want to continue? [y/n]')
        if answer.lower().strip() != 'y':
            sys.exit(1)
        
        os.remove(args.output_path)

    trainer = BlokusTrainer(
        output_file=args.output_path,
        batch_size=args.batch_size,
        num_steps=args.steps,
        num_epochs=args.epochs,
        validate_path=args.validate,
        metrics_path_override=args.metrics
    )

    trainer.train()