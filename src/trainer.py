from multiprocessing.connection import Listener
from threading import Thread

from queue import Queue


batch_queue = Queue()


class BlokusTrainer:
    def __init__(self):
        self.models = self._init_models()


    def _init_models(self):
        '''
        Models for each player, ordered as follows:
        Red
        Blue
        Green
        Yellow
        '''

        for i in range(4):
            pass


    def train(self):
        for batch in self.next_batch():
            print(batch)


    '''
    Read the latest games from the queue until we have reached
    batch size.
    '''
    def next_batch(self):
        while True:
            yield batch_queue.get(timeout=30)



'''
Receive a message from the Blokus client that contains:
- All board states for each move in the game played
- Final scores of each player
'''
def on_message_received(trainer, message):
    batch_queue.put((message['boards'], message['scores']))


def server_loop():
    listener = Listener(('localhost', 8080))
    connection = listener.accept()

    print('Connection established with Blokus player')

    while True:
        try:
            message = connection.recv()
            if not message:
                break
        except EOFError:
            break

        on_message_received(trainer, message)

    listener.close()


if __name__ == '__main__':
    thread = Thread(target=server_loop)
    thread.start()

    trainer = BlokusTrainer()
    trainer.train()