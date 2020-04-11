import blokus
import abc
import random

from timer import timer, print_results


class Ai(abc.ABC):
    def __init__(self):
        self.board = blokus.Board()
        self.turn_count = 0
        self.players_remaining = len(self.board.players)
        self.is_complete = False


    @abc.abstractmethod
    def next_move(self, player):
        pass


    @timer
    def play_game(self):
        while not self.is_complete:
            self.next_turn()

        self.board.save('boards/game.png')


    def next_turn(self):
        if self.players_remaining == 0:
            self.end_game()
            return

        player = self.board.players[self.get_turn()]
        self.turn_count += 1

        move = self.next_move(player)
        if move:
            self.board.make_move(move)
        else:
            self.players_remaining -= 1


    def end_game(self):
        print('Game over!')
        print('Final score:')

        red_player = self.board.players[0]
        blue_player = self.board.players[1]
        green_player = self.board.players[2]
        yellow_player = self.board.players[3]

        print('Red:', red_player.get_score())
        print('Blue:', blue_player.get_score())
        print('Green:', green_player.get_score())
        print('Yellow:', yellow_player.get_score())

        self.is_complete = True


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


if __name__ == '__main__':
    for i in range(10):
        ai = RandomAi()
        ai.play_game()
        blokus.Move._move_cache.clear()

    print_results()