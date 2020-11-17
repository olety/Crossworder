from string import ascii_lowercase
import numpy as np
from copy import copy
import sys

sys.setrecursionlimit(1000)


class BBoard:

    alphabet = list(ascii_lowercase)

    def __init__(self, rows=5, cols=5):
        self.rows = rows
        self.cols = cols
        self.init_board()

    def init_board(self):
        self.board = np.empty(shape=(self.rows, self.cols), dtype="str")
        self.frw_options = np.empty(shape=(self.rows, self.cols), dtype="object")
        for i in range(self.rows):
            for j in range(self.cols):
                self.frw_options[i, j] = copy(self.alphabet)

    def check(self, letter, row, col):
        # print('Checking letter {} @ r{}c{}'.format(letter, row, col))
        fail = False
        # print(letter)
        # print(self.board[row-1][col])
        if row < self.rows - 1 and self.board[row + 1][col] == letter:
            fail = True
        if col < self.cols - 1 and self.board[row][col + 1] == letter:
            fail = True
        if row > 0 and self.board[row - 1][col] == letter:
            fail = True
        if col > 0 and self.board[row][col - 1] == letter:
            fail = True
        # print('Check : {}'.format(not fail))
        return not fail

    def check_board(self):
        for index, letter in enumerate(self.board):
            if not self.check(letter, index[0], index[1]):
                return False
        return True

    def next_pos(self, row, col):
        if row < self.rows - 1:
            return row + 1, col
        elif col < self.cols - 1:
            return 0, col + 1

    def backtracking_rec(self, row, col):
        for letter in self.alphabet:
            if self.check(letter, row, col):
                print("Recursion - r{} c{} l{}".format(row, col, letter))
                self.board[row, col] = letter
                if not self.next_pos(row, col) or self.backtracking_rec(
                    *self.next_pos(row, col)
                ):
                    return True
                else:
                    self.board[row, col] = ""
        return False

    def backtracking(self):
        for letter in self.alphabet:
            if self.check(letter, 0, 0):
                print("Processing a route starting with {}".format(letter))
                self.board[0, 0] = letter
                if not self.next_pos(0, 0) or self.backtracking_rec(
                    *self.next_pos(0, 0)
                ):
                    return True, self.board
                else:
                    self.board[0, 0] = ""
        return False

    def frw_rec(self, row, col):
        for letter in self.frw_options[row][col]:
            print("Recursion - r{} c{} l{}".format(row, col, letter))
            self.board[row, col] = letter
            self.remove_options(row, col, letter)
            if not self.next_pos(row, col) or self.frw_rec(*self.next_pos(row, col)):
                return True
            else:
                self.board[row, col] = ""
        return False

    def remove_options(self, row, col, letter):
        try:
            if row < self.rows - 1:
                self.frw_options[row + 1][col].remove(letter)
            if col < self.cols - 1:
                self.frw_options[row][col + 1].remove(letter)
            if row > 0:
                self.frw_options[row - 1][col].remove(letter)
            if col > 0:
                self.frw_options[row][col - 1].remove(letter)
        except Exception:
            pass

    def forward_checking(self):
        for letter in self.frw_options[0][0]:
            print("Processing a route starting with {}".format(letter))
            self.board[0, 0] = letter
            self.remove_options(0, 0, letter)
            if not self.next_pos(0, 0) or self.frw_rec(*self.next_pos(0, 0)):
                return True, self.board
            else:
                self.board[0, 0] = ""
        return False


b = BBoard(26, 5)
res = b.forward_checking()
from pprint import pprint

print(np.array_str(res[1]))
