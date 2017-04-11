import numpy as np
import copy
import logging
import sys


class Word:
    def __init__(self, word, row, col, horizontal):
        # Starting cell = top left corner of the word
        self.word = word
        self.length = len(self.word)
        self.row = row
        self.col = col
        self.horizontal = horizontal

    def __len__(self):
        return len(self.word)

    def __str__(self):
        return self.word

    def __repr__(self):
        return self.word


class Crossword:
    current_words = []

    def __init__(self, n=5, m=5, file_name='lemma.num.txt', maximize_len=False):
        logging.info('Crossword __init__: Initializing crossword...')
        logging.debug('Crossword __init__: n={}, m={}, fname={}'.format(n, m, file_name))
        # Initializing crossword bounds
        self.rows = n
        self.cols = m
        logging.debug('Crossword __init__: initing the board')
        self._init_board()
        # Loading words
        logging.debug('Crossword __init__: Started loading words from {}'.format(file_name))
        arr = np.genfromtxt(file_name, dtype='str', delimiter=' ')
        self.words = arr[np.in1d(arr[:, 3], ['v', 'n', 'adv', 'a'])][:, 2].tolist()
        # Number of words loaded
        logging.debug('Crossword __init__: Number of words loaded: {}'.format(len(self.words)))
        self.maximize = maximize_len
        self.words = sorted([x for x in self.words if len(x) <= n and len(x) <= m], key=len, reverse=maximize_len)
        # After filter logging
        logging.debug('Crossword __init__: Number of words after filter: {}, maxlen = {}'.format(len(self.words), len(
            max(self.words, key=len))))

    def _init_board(self):
        logging.debug('Crossword init_board: initing a list, rows={}, cols={}'.format(self.rows, self.cols))
        self.board = [[None for _ in range(self.cols)] for _ in range(self.rows)]

    def generate(self, num_words, algo='back'):
        # Algo choices: 'back', 'frwd'
        # Setting the number of words
        if num_words < 0 or num_words > self.rows * self.cols:
            raise Exception('Bad number of words {}'.format(num_words))
        self.num_words = num_words
        # Calling an algorythm function
        if algo == 'back':
            return self._back()
        elif algo == 'forward':
            return self._forward()
        else:
            raise Exception('Bad algo chosen: {}'.format(algo))

    def _fill(self, word):
        self.current_words.append(word)
        if word.horizontal:
            for i in range(word.col, word.col + word.length):
                self.board[word.row][i] = word.word[i - word.col]
        else:
            for i in range(word.row, word.row + word.length):
                self.board[i][word.col] = word.word[i - word.row]

    def _erase(self, word):
        self.current_words.remove(word)
        if word.horizontal:
            for i in range(word.col, word.col + word.length):
                self.board[word.row][i] = None
        else:
            for i in range(word.row, word.row + word.length):
                self.board[i][word.col] = None

    def _erase_all_words(self):
        for word in self.current_words:
            self._erase(word)

    def _check_word(self, word, start_row, start_col, horizontal):
        print(word, start_row, start_col, horizontal)
        for i in range(len(word)):
            cell = self.board[start_row + (0 if horizontal else i)][start_col + (i if horizontal else 0)]
            if cell and cell != word[i]:
                return False
        return True

    def _check_cross(self, word_board, word_string, letter_pos):
        horizontal = not word_board.horizontal
        intersect_pos = [index for index, letter in enumerate(word_board.word) if letter == word_string[letter_pos]]
        for ind in intersect_pos:
            if self._check_word(word_string, word_board.row + ind if horizontal else 0,
                                word_board.col + 0 if horizontal else ind, horizontal):
                return Word(word_string, word_board.row + (ind if horizontal else 0),
                            word_board.col + (0 if horizontal else ind), horizontal)
        return None

    def _check_empty(self, word_string):
        return None

    def _next_pos(self, word):
        if not self.current_words:
            return Word(word, 0, 0, True)
        # Check the existing words
        for cur_w in self.current_words:
            for index in range(len(word)):
                if word[index] in cur_w.word:
                    gen_w = self._check_cross(cur_w, word, index)
                    if gen_w:
                        return gen_w
        return None
        # Not checking the empty space
        #return self._check_empty(word)

    @property
    def _count_words(self):
        return len(self.current_words)

    def _delete_unusable(self):
        cur_word = self.current_words[-1]
        for word in self.usable_words_frw:
            pass

    @property
    def _usable_words(self):
        usable_words = copy.copy(self.words)
        for word in self.current_words:
            usable_words.remove(word.word)
        return usable_words

    def _forward_rec(self):
        for word in self._usable_words_frw:
            cur_word = self._next_pos(word)
            if cur_word:
                self._fill(cur_word)
                if self._count_words >= self.num_words:
                    return True
                else:
                    return self._forward_rec()
        return False

    def _forward(self):
        # DELETE WORDS IN FORWARD
        # ONLY CHECK INTERSECTS
        self.usable_words_frw = copy.copy(self.words)
        for word in self._usable_words_frw:
            cur_word = self._next_pos(word)
            if cur_word:
                self._fill(cur_word)
                self._delete_unusable()
                if self._count_words >= self.num_words or self._forward_rec():
                    return True, self.board
                else:
                    self._erase(cur_word)
        return [False]

    def _back_rec(self):
        for word in self._usable_words:
            cur_word = self._next_pos(word)
            if cur_word:
                self._fill(cur_word)
                if self._count_words >= self.num_words:
                    return True
                else:
                    return self._back_rec()
        return False

    def _back(self):
        for word in self._usable_words:
            cur_word = self._next_pos(word)
            if cur_word:
                self._fill(cur_word)
                if self._count_words >= self.num_words or self._back_rec():
                    return True, self.board
                else:
                    self._erase(cur_word)
        return [False]

    def show_crossword(self):
        from pprint import pprint
        pprint(self.board)

    def __str__(self):
        return str(self.current_words)


logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
c = Crossword(maximize_len=False)
res = c.generate(5, algo='back')
if len(res) > 1:
    from pprint import pprint
    pprint(res[1])
else:
    print(res[0])
