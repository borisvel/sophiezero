class Cell:
    def __init__(self, letter, state, color):
        self.letter = letter
        self.state = state
        self.color = color

    def __eq__(self, other):
        return self.letter == other.letter and self.state == other.state and self.color == other.color

    def __str__(self):
        return str((self.state, self.letter, self.color))

    def is_visible(self):
        return self.state == 1

    def get_color(self):
        return self.color

    def get_letter(self):
        return self.letter


class Board(Cell):
    def __init__(self):
        self.board = []
        for i in range(6):
            self.board.append([])
            for j in range(5):
                c = Cell("a", 0, "B")
                self.board[-1].append(c)
        self.information = []
        for i in range(26):
            self.information.append([[0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1]])

    def __str__(self):
        res = ""
        for i in range(6):
            for j in range(5):
                res += self.board[i][j].__str__() + "   "
            res += '\n'
        return res

    def to_binary(self, x):
        binary = []
        while x > 0:
            binary.append(x % 2)
            x //= 2
        binary.reverse()
        return binary

    def encode1(self):
        enc = []
        for i in range(6):
            for j in range(5):
                cell = self.board[i][j]
                enc.append(int(cell.is_visible()))
                letter = ord(cell.get_letter()) - 97
                binary = self.to_binary(letter)
                enc += [0]*(5 - len(binary)) + binary
                color = cell.get_color()
                if color == 'B':
                    enc += [0, 0]
                if color == 'Y':
                    enc += [0, 1]
                if color == 'G':
                    enc += [1, 0]
        return enc

    def encode2(self):
        enc = []
        for i in range(6):
            for j in range(5):
                cell = self.board[i][j]
                if cell.is_visible():
                    enc += [0, 1]
                else:
                    enc += [1, 0]
                letter = ord(cell.get_letter()) - 97
                one_hot = [0]*26
                one_hot[letter] = 1
                enc += one_hot
                color = cell.get_color()
                if color == 'B':
                    enc += [0, 0, 1]
                if color == 'Y':
                    enc += [0, 1, 0]
                if color == 'G':
                    enc += [1, 0, 0]
        return enc

    def hashcode(self):
        enc = self.encode1()
        h = 0
        for digit in enc:
            h *= 10
            h += digit
        return h

    def __eq__(self, other):
        for i in range(6):
            for j in range(5):
                if self.get_cell(i, j) != other.get_cell(i, j):
                    return False
        return True

    def get_cell(self, i, j):
        return self.board[i][j]

    def current_pointer(self):
        for i in range(6):
            if not self.board[i][0].is_visible(): return i
        return 6

    def update(self, update, final_word):
        pt = self.current_pointer()
        yellow_idx = set()
        for i in range(5):
            pos = ord(update[i:i+1]) - 97
            if update[i:i+1] not in final_word:
                c = Cell(update[i:i + 1], 1, 'B')
                self.board[pt][i] = c
                self.information[pos] = [[0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0]]
            elif update[i:i+1] == final_word[i:i+1]:
                c = Cell(update[i:i + 1], 1, 'G')
                self.board[pt][i] = c
                self.information[pos][i] = [1, 0, 0]
            else:
                yellow_idx.add(i)
        almost = {}
        for i in yellow_idx:
            letter = update[i:i+1]
            pos = ord(letter) - 97
            if letter not in almost:
                almost[letter] = 0
            total_known = 0
            total_final_word = 0
            for j in range(5):
                if update[j:j+1] == letter and update[j:j+1] == final_word[j:j+1]:
                    total_known += 1
                if final_word[j:j+1] == letter:
                    total_final_word += 1
            if total_final_word - total_known - almost[letter] > 0:
                almost[letter] += 1
                c = Cell(update[i:i+1], 1, 'Y')
                self.board[pt][i] = c
                self.information[pos][i] = [0, 1, 0]
            else:
                c = Cell(update[i:i+1], 1, 'B')
                for j in range(5):
                    if self.information[pos][j] == [0, 0, 1]:
                        self.information[pos][j] = [0, 1, 0]
                self.board[pt][i] = c

    def correct_letters_last_try(self):
        pt = self.current_pointer()
        if pt == 0: return 0
        res = 0
        for i in range(5):
            if self.board[pt - 1][i].get_color() == 'G':
                res += 1
        return res

    def cost(self):
        pt = self.current_pointer()
        if self.correct_letters_last_try() == 5:
            return pt
        elif pt == 6:
            return 7
        return 0


class Game(Board):
    def __init__(self, final_word, board):
        self.board = board
        self.final_word = final_word
        self.guesses = []

    def __eq__(self, other):
        pt = self.board.current_pointer()
        if pt != other.board.current_pointer():
            return False
        for i in range(pt):
            for j in range(5):
                if self.board.get_cell(i, j) != other.board.get_cell(i, j):
                    return False
        return True

    def encode1(self):
        return self.board.encode1()

    def encode2(self):
        return self.board.encode2()

    def hashcode(self):
        return self.board.hashcode()

    def is_done(self):
        return self.board.current_pointer() == 6 or self.board.correct_letters_last_try() == 5

    def get_guesses(self):
        return self.guesses

    def step(self, update):
        self.guesses.append(update)
        if self.is_done(): return 0.0
        self.board.update(update, self.final_word)
        return self.board.cost()

    def get_features(self, a):
        pt = self.board.current_pointer()
        guesses_left = 6 - pt
        correct = {}
        correct_idx = {}
        almost = {}
        for i in range(pt):
            for j in range(5):
                if self.board.get_cell(i, j).get_color() == 'G':
                    correct_idx[j] = self.board.get_cell(i, j).get_letter()

        correct_letters_known = len(correct_idx)

        for idx in correct_idx:
            letter = correct_idx[idx]
            if letter not in correct:
                correct[letter] = 0
            correct[letter] += 1

        for i in range(pt):
            curr = []
            for j in range(5):
                if self.board.get_cell(i, j).get_color() == 'Y' or self.board.get_cell(i, j).get_color() == 'G':
                    curr.append(self.board.get_cell(i, j).get_letter())
            for letter in curr:
                cnt = curr.count(letter)
                if letter not in almost:
                    almost[letter] = 0
                almost[letter] = max(almost[letter], cnt)

        wrong_place_letters_known = 0

        for letter in almost:
            if letter in correct:
                wrong_place_letters_known += almost[letter] - correct[letter]
            else:
                wrong_place_letters_known += almost[letter]

        correct_letters_in_action = 0

        for i in range(len(a)):
            if i in correct_idx and correct_idx[i] == a[i:i+1]:
                correct_letters_in_action += 1

        letters_not_yet_known_place = 0

        for letter in a:
            if letter not in correct and letter in almost:
                letters_not_yet_known_place += 1

        return 1, guesses_left, wrong_place_letters_known, correct_letters_known, correct_letters_in_action, letters_not_yet_known_place


class GameState:
    def __init__(self, final_word):
        self.information = []
        self.pt = 0
        self.guesses = []
        self.final_word = final_word
        self.alfabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r',
                        's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

        for i in range(26):
            self.information.append([1, 1, 1, 1, 1])

    def __str__(self):
        res = ""
        for i in range(26):
            res += self.alfabet[i] + ': '
            for j in range(5):
                res += str(self.information[i][j]) + '  '
            res += '\n'
        return res

    def __eq__(self, other):
        return self.information == other.information and self.pt == other.pt

    def get_guesses(self):
        return self.guesses

    def hashcode(self):
        hash = 0
        for i in range(26):
            for j in range(5):
                hash *= 10
                hash += self.information[i][j]
        return hash

    def is_done(self):
        return self.pt == 6 or self.final_word in self.guesses

    def cost(self):
        if self.final_word in self.guesses:
            return self.pt
        if self.pt == 6:
            return 7
        return 0

    def step(self, a):
        self.pt += 1
        self.guesses.append(a)
        for i in range(len(a)):
            letter = a[i:i+1]
            pos = (ord(letter) - 97)
            if letter == self.final_word[i]:
                self.information[pos] = [0, 0, 0, 0, 0]
                self.information[pos][i] = 1
            elif letter not in self.final_word:
                self.information[pos] = [0, 0, 0, 0, 0]
            else:
                self.information[pos][i] = 0
        return self.cost()

