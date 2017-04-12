import numpy as np
import copy
import logging
import sys
import signal
import time
from processify import processify
from contextlib import contextmanager

sys.setrecursionlimit(1000)


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
    HORIZONTAL = 1
    VERTICAL = 2
    hor_arr = (HORIZONTAL, HORIZONTAL + VERTICAL)
    ver_arr = (VERTICAL, HORIZONTAL + VERTICAL)
    current_words = []

    def __init__(self, **kwargs):
        logging.info('Crossword __init__: Initializing crossword...')
        logging.debug('kwargs:', kwargs)
        # Reading kwargs
        self.setup = kwargs
        self.rows = int(kwargs.get('n', 5))
        self.cols = int(kwargs.get('m', 5))
        self.words_file = str(kwargs.get('word_file', 'lemma.num.txt'))
        self.sort = bool(kwargs.get('sort', False))
        self.maximize_len = bool(kwargs.get('maximize_len', False))
        self.repeat_words = bool(kwargs.get('repeat_words', False))
        logging.debug('Crossword __init__: n={}, m={}, fname={}'.format(self.rows, self.cols, self.words_file))
        # Loading words
        logging.debug('Crossword __init__: Started loading words from {}'.format(self.words_file))
        arr = np.genfromtxt(self.words_file, dtype='str', delimiter=' ')
        self.words = arr[np.in1d(arr[:, 3], ['v', 'n', 'adv', 'a'])][:, 2].tolist()
        # Number of words loaded
        logging.debug('Crossword __init__: Number of words loaded: {}'.format(len(self.words)))
        self.words = list(set(x for x in self.words if len(x) <= self.rows and len(x) <= self.cols))
        if self.sort:
            self.words = sorted(self.words, key=len, reverse=self.maximize_len)
        # After filter logging
        logging.debug('Crossword __init__: Number of words after filter: {}, maxlen = {}'.format(len(self.words), len(
            max(self.words, key=len))))

    def _init_board(self):
        logging.debug('Crossword init_board: initing a list, rows={}, cols={}'.format(self.rows, self.cols))
        self.board = [[[None, 0] for _ in range(self.cols)] for _ in range(self.rows)]

    # @processify
    def generate(self, num_words, algo='back', **kwargs):
        # Algo choices: 'back', 'frwd'
        # Setting the number of words
        logging.debug('Crossword __init__: initing the board')
        self._init_board()
        self._erase_all_words()
        self._init_board()
        if num_words < 0 or num_words > self.rows * self.cols:
            raise Exception('Bad number of words {}'.format(num_words))
        self.num_words = num_words
        # Calling an algorythm function
        time_start = time.perf_counter()
        if kwargs.get('max_time', None):
            max_time = int(kwargs['max_time'])
        if algo == 'back':
            if max_time:
                with time_limit(max_time):
                    try:
                        res = self._back()
                    except Exception:
                        return [False, None]
            else:
                res = self._back()
        elif algo == 'forward':
            if max_time:
                with time_limit(max_time):
                    try:
                        res = self._forward()
                    except Exception:
                        return [False, None]
            else:
                res = self._forward()
        else:
            raise Exception('Bad algo chosen: {}'.format(algo))
        time_end = time.perf_counter()
        if kwargs.get('save_file', None):
            self._save(kwargs['save_file'], time_end - time_start, max_time, algo, res, num_words)
        return res

    def _fill(self, word):
        self.current_words.append(word)
        flag = False
        if word.horizontal:
            for i in range(word.col, word.col + word.length):
                self.board[word.row][i][0] = word.word[i - word.col]
                self.board[word.row][i][1] += self.HORIZONTAL
                if self.board[word.row][i][1] > 3:
                    flag = True
        else:
            for i in range(word.row, word.row + word.length):
                self.board[i][word.col][0] = word.word[i - word.row]
                self.board[i][word.col][1] += self.VERTICAL
                if self.board[i][word.col][1] > 3:
                    flag = True
        if flag:
            self._erase(word)
            raise Exception('Bad word placement')

    def _erase(self, word):
        self.current_words.remove(word)
        if word.horizontal:
            for i in range(word.col, word.col + word.length):
                if not ((self._check_inside(word.row - 1, i) and self.board[word.row - 1][i][0]) or (
                            self._check_inside(word.row + 1, i) and self.board[word.row + 1][i][0])):
                    self.board[word.row][i][0] = None
                self.board[word.row][i][1] -= self.HORIZONTAL
        else:
            for i in range(word.row, word.row + word.length):
                if not ((self._check_inside(i, word.col - 1) and self.board[i][word.col - 1][0]) or (
                            self._check_inside(i, word.col + 1) and self.board[i][word.col + 1][0])):
                    self.board[i][word.col][0] = None
                self.board[i][word.col][1] -= self.VERTICAL

    def _erase_all_words(self):
        # for word in self.current_words[:]:
        #     self._erase(word)
        self.current_words = []

    def _check_inside(self, row, col):
        return (row < self.rows) and (col < self.cols) and row >= 0 and col >= 0

    def _check_exists(self, row, col, hor):
        for word in self.current_words:
            if word.row == row and word.col == col and word.horizontal == hor:
                return True
        return False

    def _check_word(self, word, start_row, start_col, horizontal, end_row, end_col):
        # print(word, start_row, start_col, horizontal, end_row, end_col)
        vert_one = 0 if horizontal else 1
        hor_one = 1 if horizontal else 0

        # Initial checks
        if not self._check_inside(start_row, start_col) or \
                not self._check_inside(end_row, end_col) or \
                self._check_exists(start_row, start_col, horizontal):
            return False

        # Checking if there's another word near the end
        if self._check_inside(end_row + vert_one, end_col + hor_one) and \
                        self.board[end_row + vert_one][end_col + hor_one][0] is not None:
            return False

        # Checking if there's another word near the start
        if self._check_inside(start_row - vert_one, start_col - hor_one) and \
                        self.board[start_row - vert_one][start_col - hor_one][0] is not None:
            return False
        # check_neg = ''
        # check_pos = ''
        for i in range(len(word)):
            cur_pos = (start_row + vert_one * i, start_col + hor_one * i)
            try:
                cell = self.board[cur_pos[0]][cur_pos[1]]
            except Exception:
                return False

            if cell[0] and cell[0] != word[i]:
                return False

            if ((self._check_inside(cur_pos[0] + hor_one, cur_pos[1] + vert_one) and
                         self.board[cur_pos[0] + hor_one][cur_pos[1] + vert_one][1] not in [
                         self.VERTICAL if horizontal else self.HORIZONTAL, 0]) or
                    (self._check_inside(cur_pos[0] - hor_one, cur_pos[1] - vert_one) and
                             self.board[cur_pos[0] - hor_one][cur_pos[1] - vert_one][1] not in [
                             self.VERTICAL if horizontal else self.HORIZONTAL, 0])):
                return False
        # if self._check_inside(cur_pos[0] + hor_one, cur_pos[1] + vert_one):
        #         check_pos += str(self.board[cur_pos[0] + hor_one][cur_pos[1] + vert_one][1]) + ' vs ' + str(
        #             self.hor_arr if horizontal else self.ver_arr) + ', res = ' + str(
        #             self.board[cur_pos[0] + hor_one][cur_pos[1] + vert_one][1] in (
        #             self.hor_arr if horizontal else self.ver_arr)) + ' | '
        #     if self._check_inside(cur_pos[0] - hor_one, cur_pos[1] - vert_one):
        #         check_neg += str(self.board[cur_pos[0] - hor_one][cur_pos[1] - vert_one][1]) + ' vs ' + str(
        #             self.hor_arr if horizontal else self.ver_arr) + ', res = ' + str(
        #             self.board[cur_pos[0] - hor_one][cur_pos[1] - vert_one][1] in (
        #             self.hor_arr if horizontal else self.ver_arr)) + ' | '
        # print('Accepted {}. start=({},{}), end=({},{}), hor={}'.format(word,start_row,start_col,end_row,end_col,horizontal))
        # print('Checkstring - negative: {}'.format(check_neg))
        # print('Checkstring - positive: {}'.format(check_pos))
        return True

    def _check_cross(self, word_board, word_string, letter_pos):
        horizontal = not word_board.horizontal
        intersect_pos = [index for index, letter in enumerate(word_board.word) if letter == word_string[letter_pos]]
        for ind in intersect_pos:
            start_pos = (word_board.row + (ind if horizontal else -1 * letter_pos),
                         word_board.col + (0 if horizontal else ind))
            end_pos = (start_pos[0] + (0 if horizontal else (len(word_string) - 1)),
                       start_pos[1] + ((len(word_string) - 1) if horizontal else 0))
            if self._check_inside(start_pos[0], start_pos[1]) \
                    and self._check_word(word_string,
                                         start_pos[0],
                                         start_pos[1],
                                         horizontal,
                                         end_pos[0],
                                         end_pos[1]
                                         ):
                return Word(word_string, word_board.row + (ind if horizontal else -1 * letter_pos),
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
        # return self._check_empty(word)

    def _next_pos_fill(self, word):
        if not self.current_words:
            return Word(word, 0, 0, True)
        # Check the existing words
        for cur_w in self.current_words:
            for index in range(len(word)):
                if word[index] in cur_w.word:
                    gen_w = self._check_cross(cur_w, word, index)
                    if gen_w:
                        try:
                            self._fill(gen_w)
                        except Exception:
                            continue
                        return gen_w
        return None

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
        if not self.repeat_words:
            for word in self.current_words:
                usable_words.remove(word.word)
        return usable_words

    @property
    def _usable_words_frw(self):
        usable_words = copy.copy(self.words)
        if not self.repeat_words:
            for word in self.current_words:
                usable_words.remove(word.word)
        for uword in usable_words[:]:
            for letter in uword:
                if not any(letter in word and len(word) > 1 for word in uword.split()):
                    usable_words.remove(uword)
        return usable_words

    def _forward_rec(self):
        for word in self._usable_words_frw:
            cur_word = self._next_pos_fill(word)
            if cur_word:
                if self._count_words >= self.num_words or self._forward_rec():
                    return True
                else:
                    self._erase(cur_word)
        return False

    def _forward(self):
        # DELETE WORDS IN FORWARD
        # ONLY CHECK INTERSECTS
        for word in self._usable_words_frw:
            cur_word = self._next_pos(word)
            if cur_word:
                self._fill(cur_word)
                if self._count_words >= self.num_words or self._forward_rec():
                    return True, self.board
                else:
                    self._erase(cur_word)
        return [False, None]

    def _save(self, file_path, time, max_time, algo, res, num_words):
        np.save(file_path, np.array([self.current_words, res, self.setup, time, max_time, algo, num_words]))

    def _back_rec(self):
        for word in self._usable_words:
            cur_word = self._next_pos_fill(word)
            if cur_word:
                if self._count_words >= self.num_words or self._back_rec():
                    return True
                else:
                    self._erase(cur_word)

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

    def show_results(self, res):
        if len(res) > 1:
            from pprint import pprint
            for word in self.current_words:
                print(word, word.row, word.col)
            print('-' * (self.cols * 2 - 1))
            for row in res[1]:
                for word in row:
                    print('-' if not word[0] else word[0], end=' ')
                print()
            print('-' * (self.cols * 2 - 1))
            for row in res[1]:
                for word in row:
                    print('-' if not word[0] else word[1], end=' ')
                print()
            print('-' * (self.cols * 2 - 1))
        else:
            print(res[0])


class TimeoutException(Exception): pass


@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException()

    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)


def results(algo='forward'):
    # 5x5 board - 3,5,7 words - rand, max and min len
    # 15x15 board - 13,15,17 words - rand, max and min len
    # 15x30 board - 30 words - rand, max and min len
    import yaml, time
    # logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    with open('settings.yaml', 'r') as f:
        settings = yaml.load(f)
    results = []
    for board in settings[0]['boards'][0]:
        c = Crossword(n=board['n'], m=board['m'], sort=board['sort'],
                      maximize_len=board.get('max', False), repeat_words=board.get('dupes', False))
        for num_words in board['word_len']:
            print('Processing board {} num_words {}'.format(board, num_words))
            times = []
            ress = []
            for i in range(settings[0].get('num_repeats', 10)):
                print('i = {} ...'.format(i), end=' ')
                time_start = time.perf_counter()
                ress.append(c.generate(num_words, algo=algo, max_time=15))
                time_end = time.perf_counter()
                times.append(time_end - time_start)
                print('Execution took {:.5f} seconds'.format(time_end - time_start))
            print('Avg time: {:.5f}'.format(np.average(times)))
            print('Stddev: {:.5f}'.format(np.std(times)))
            results.append({
                'board': board,
                'num_words': num_words,
                'results': ress,
                'times': times})
            # Crossword.show_results(c, res)

    np.save(algo, np.array(results))


def process_results(file1='backtracking.npy', file2='forward.npy'):
    from pprint import pprint
    a2 = np.load(file2)
    pprint(a2[0])

if __name__ == '__main__':
    results()
    process_results()
    pass
