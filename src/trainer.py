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
import random
import hashlib
import statistics

from socketserver import ThreadingTCPServer, StreamRequestHandler

from threading import Thread, Event, Condition, Lock
from pathlib import Path



def model_digest(model):
    digest = hashlib.md5()

    for tensor in model.get_weights():
        for element in tensor.flatten():
            digest.update(element)

    return digest.hexdigest()



class BlokusGameRequestHandler(StreamRequestHandler):
    WORKER_CONNECTIONS = list()

    '''
    Handle a single connection from a worker thread playing Blokus games.
    The connection will remain open indefinitely until the trainer is done training,
    continuously receiving results of games played by the worker.

    Occasionally, we will send each worker an updated version of the model to use
    for future games.
    '''
    def handle(self):
        BlokusGameRequestHandler.WORKER_CONNECTIONS.append(self)
        self.epoch_end_event = Event()

        print('Connection to new worker established')
        trainer = self.server.trainer

        self.send_model_to_client()

        while True:
            try:
                raw_data = self.rfile.readline()
                data = json.loads(raw_data)
            except json.JSONDecodeError:
                if raw_data == b'':
                    break

                print(f'Invalid JSON in request: {raw_data}')
                continue

            if trainer.training_complete:
                self.wfile.write(bytes(struct.pack('i', -1)))
                break

            if 'boards' not in data or 'targets' not in data:
                print('Invalid request, could not find boards or targets in request!')
                continue

            buffer_full = trainer.enqueue_game(data['boards'], data['targets'])
            if buffer_full:
                self.epoch_end_event.wait()

            if self.epoch_end_event.is_set():
                self.epoch_end_event.clear()
                self.send_model_to_client()
            else:
                self.wfile.write(bytes(struct.pack('i', 0)))


    def send_model_to_client(self):
        with open(BlokusTrainer.MODEL_SYNC_OUTPUT_PATH, 'rb') as f:
            self.wfile.write(
                struct.pack('i', os.path.getsize(BlokusTrainer.MODEL_SYNC_OUTPUT_PATH)) + f.read()
            )

            f.close()


    def on_epoch_end(self):
        self.epoch_end_event.set()



class TrainerClient:
    def __init__(self, server_address, server_port):
        self.address = server_address
        self.port = server_port

        self.connection = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.remote_model_path = f'models/remote_model_{os.getpid()}.h5'


    def connect(self):
        self.connection.connect((self.address, self.port))
        self._download_model(self._get_response())


    def _download_model(self, raw_response):
        if os.path.exists(self.remote_model_path):
            os.remove(self.remote_model_path)

        with open(self.remote_model_path, 'wb') as f:
            f.write(raw_response)

        print('Received updated model')


    '''
    Sends a training example to the trainer.
    
    Returns true if a new model is available to use from the trainer, false otherwise.
    '''
    def send_training_example(self, boards, targets):
        serialized_game = json.dumps({
            'boards': boards,
            'targets': targets
        }) + '\n'

        self.connection.sendall(bytes(serialized_game, 'utf-8'))

        response = self._get_response()
        if not response:
            return False

        self._download_model(response)
        return True


    def _get_response(self):
        response = bytes()
        
        raw_content_length = self.connection.recv(4)
        if not raw_content_length:
            return None

        content_length = struct.unpack('i', raw_content_length)[0]

        if content_length == -1:
            print('Training complete, shutting down worker.')
            self.connection.close()
            sys.exit(0)

        if content_length == 0:
            return None

        while len(response) < content_length:
            response += self.connection.recv(65536)

        return response



class ExperienceReplayBuffer:
    def __init__(self, *, batch_size, num_entries, num_steps):
        self.num_entries = num_entries
        self.num_steps = num_steps
        self.batch_size = batch_size

        self.buffer = list()
        self.steps_returned = 0

        self.buffer_full = Condition()
        self.buffer_add_lock = Lock()


    def add(self, board, target):
        with self.buffer_add_lock:
            with self.buffer_full:
                if self.is_full():
                    return True

                self.buffer.append((board, target))

                print(f'\rExperience received: {len(self.buffer)}/{self.num_entries}', end='')
                
                if self.is_full():
                    self.buffer_full.notify_all()
                    return True

                return False


    def get(self):
        with self.buffer_full:
            while not self.is_full():
                self.buffer_full.wait()
                print('Average:', statistics.mean([choice[1][0] for choice in self.buffer]))

        choices = random.choices(self.buffer, k=self.batch_size)

        inputs = list()
        outputs = list()

        for choice in choices:
            inputs.append(choice[0])
            outputs.append(choice[1])

        self.steps_returned += 1

        if self.steps_returned == self.num_steps:
            self.buffer.clear()
            self.steps_returned = 0

        return (np.array(inputs), np.array(outputs))


    def is_full(self):
        return len(self.buffer) == self.num_entries



class BlokusTrainer:
    MODEL_SYNC_OUTPUT_PATH = 'models/training_model.h5'

    def __init__(self, output_file, batch_size=5, num_steps=32, num_epochs=1, experience_size=16, validate_path=None, metrics_path_override=None):
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
        self.model.save(BlokusTrainer.MODEL_SYNC_OUTPUT_PATH)
        print('Model md5 digest:', model_digest(self.model))

        self.experience_replay = ExperienceReplayBuffer(batch_size=batch_size, num_entries=experience_size, num_steps=num_steps)

        self.server = None
        self.training_complete = False

        # Initiate the server on a new thread. Server will wait for incoming connections
        # from workers playing games of Blokus.
        Thread(target=self._start_server).start()


    def _start_server(self):
        host, port = '0.0.0.0', 8888

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
            if not isinstance(example, list) or len(example) != 2:
                raise ValueError(f'Invalid game data in file {validate_path}')

            inputs.append(example[0])
            outputs.append([example[1]])

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
                callbacks=[tensorboard_callback, BlokusTrainerCallback()]
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


    def enqueue_game(self, boards, targets):
        for board, target in zip(boards, targets):
            buffer_full = self.experience_replay.add(board, np.array([target]))
            if buffer_full:
                return True

        return False


    '''
    Read the latest games from the queue until we have reached
    batch size. Each game can have an arbitrary number of moves,
    batch size is based on number of moves.
    '''
    def get_batches(self):
        while True:
            yield self.experience_replay.get()



class BlokusTrainerCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        self.model.save(BlokusTrainer.MODEL_SYNC_OUTPUT_PATH)

        for client in BlokusGameRequestHandler.WORKER_CONNECTIONS:
            client.on_epoch_end()



if __name__ == '__main__':
    parser = argparse.ArgumentParser('Train a model for Blokus')
    parser.add_argument('-o', action='store', required=True, help='The output file of the resulting model.', dest='output_path')
    parser.add_argument('--batch-size', action='store', type=int, required=True, help='The batch size of the model.')
    parser.add_argument('--steps', action='store', type=int, required=True, help='The number of training examples per epoch.')
    parser.add_argument('--epochs', action='store', type=int, required=True, help='The number of epochs of training.')
    parser.add_argument('--experience', action='store', type=int, required=True, help='The size of the experience buffer.')
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
        experience_size=args.experience,
        validate_path=args.validate,
        metrics_path_override=args.metrics
    )

    trainer.train()