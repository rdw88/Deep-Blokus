import blokus
import abc
import random
import os


class Ai(abc.ABC):
    pass



'''
Each player makes a random move on each turn but prioritizes
playing pieces with a greater point value.
'''
class RandomAi(Ai):
    def __init__(self):
        self.board = blokus.Board()
        self.turn_count = 0
        self.finished_count = 0
        self.completed = False


    def play_game(self):
        while not self.completed:
            self.next_turn()

        image = self.board.get_image()

        if os.path.exists('board/game.png'):
            os.remove('boards/game.png')

        image.save('boards/game.png')


    def next_turn(self):
        if self.finished_count == 4:
            self.end_game()
            return

        player = self.board.players[self.get_turn()]
        self.turn_count += 1

        move = self.next_move(player)
        if move:
            self.board.make_move(move)
            #image = self.board.get_image()
            #image.save('boards/game_%s.png' % self.turn_count)
        else:
            self.finished_count += 1


    def next_move(self, player):
        for i in reversed(range(1, 6)):
            pieces = list(filter(lambda piece: piece.get_size() == i, player.unplayed_pieces))

            valid_moves = player.get_valid_moves(self.board, pieces=pieces)

            if len(valid_moves) > 0:
                return valid_moves[random.randint(0, len(valid_moves) - 1)]


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

        self.completed = True


    def get_turn(self):
        return self.turn_count % len(self.board.players)


if __name__ == '__main__':
    ai = RandomAi()
    ai.play_game()