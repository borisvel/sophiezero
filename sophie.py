import numpy as np
from environment import Board, Game, GameState
import copy as cp
import torch.nn as nn
import torch
import torch.optim as optim


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(261, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        x = self.fc(x)
        return x


class Sophie:
    def __init__(self, nn_model=False):
        self.lr = 1e-6
        self.allowed_guesses = set()
        self.allowed_answers = set()
        self.epsilon = 0.5
        with open('wordle-allowed-guesses.txt') as f:
            lines = f.readlines()
            for line in lines:
                self.allowed_guesses.add(line[0:5])
        with open('wordle-answers-alphabetical.txt') as f:
            lines = f.readlines()
            for line in lines:
                self.allowed_answers.add(line[0:5])
        self.allowed_words = cp.deepcopy(self.allowed_answers).union(cp.deepcopy(self.allowed_guesses))
        self.allowed_words_list = list(self.allowed_words)
        self.allowed_guesses_list = list(self.allowed_guesses)
        self.allowed_answers_list = list(self.allowed_answers)
        self.nn_model = nn_model
        self.possible = []
        if self.nn_model:
            self.network = Net()
            self.weights = self.network.parameters()
            self.optimizer = optim.SGD(self.network.parameters(), lr=self.lr, momentum=0, weight_decay=1e-6)
            torch.autograd.set_detect_anomaly(True)
            #self.network(torch.randn(1, 1, 1030)).backward()
        else:
            self.weights = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    def to_binary(self, x):
        binary = []
        while x > 0:
            binary.append(x % 2)
            x //= 2
        binary.reverse()
        return binary

    def get_possible(self, s, previously_possible):
        guesses = s.get_guesses()
        possible = []
        for possible_final_word in previously_possible:
            board_sim = Board()
            G_sim = Game(possible_final_word, board_sim)
            eq = True
            for i in range(len(guesses)):
                guess = guesses[i]
                G_sim.step(guess)
                for j in range(5):
                    if G_sim.board.get_cell(i, j) != s.board.get_cell(i, j):
                        eq = False
                        break
                if not eq:
                    break
            if eq:
                possible.append(possible_final_word)

        return possible

    def Q_tilde(self, s, a):
        if self.nn_model:
            """
            ##Use for full representations
            encode = s.encode2()
            for letter in a:
                #binary = self.to_binary(ord(letter) - 97)
                #encode += [0]*(5 - len(binary)) + binary
                one_hot = [0]*26
                one_hot[ord(letter) - 97] = 1
                encode += one_hot
            encode = torch.tensor(np.array(encode)).float().view(1, 1060)
            #encode = encode.expand(1, 1, 265)
            Q_s_a = self.network(encode)
            """
            ##Use for information representations
            encode = np.array(s.information).flatten()
            action_encode = []
            for letter in a:
                enc = [0]*26
                enc[ord(letter) - 97] = 1
                action_encode += enc
            action_encode = np.array(action_encode).flatten()
            encode = np.append(encode, action_encode)
            encode = np.append(encode, s.pt)
            encode = torch.tensor(np.array(encode)).float().view(1, 261)
            Q_s_a = self.network(encode)
            return Q_s_a, 1

        features = np.array(s.get_features(a))
        res = np.dot(self.weights, features)
        return res, features

    def train(self, episodes):
        best_actions = {}
        for episode in range(episodes):
            final_word = self.allowed_answers_list[np.random.randint(0, len(self.allowed_answers_list))]
            board = Board()
            G = GameState(final_word)
            G_prime = Game(final_word, board)
            d = []
            self.possible = self.allowed_words
            while not G.is_done():
                r = np.random.uniform(0, 1)
                hash = G.hashcode()
                s = cp.deepcopy(G)
                #possible = self.allowed_words_list
                if hash not in best_actions:
                    possible = self.get_possible(G_prime, self.possible)
                    self.possible = possible
                    guess = possible[np.random.randint(0, len(possible))]
                    Q_s_a = self.Q_tilde(G, guess)
                    best_actions[hash] = (guess, Q_s_a[0])
                    c = G.step(guess)
                    c_prime = G_prime.step(guess)
                    if c != c_prime: print("P")
                elif r < self.epsilon/(np.sqrt((24*episode/episodes) + 1)):
                    possible = self.get_possible(G_prime, self.possible)
                    self.possible = possible
                    guess = possible[np.random.randint(0, len(possible))]
                    Q_s_a = self.Q_tilde(G, guess)
                    if Q_s_a[0] < best_actions[hash][1]:
                        best_actions[hash] = (guess, Q_s_a[0])
                    c = G.step(guess)
                    c_prime = G_prime.step(guess)
                    if c != c_prime: print("P")
                else:
                    guess = best_actions[hash][0]
                    Q_s_a = self.Q_tilde(G, guess)
                    best_actions[hash] = (guess, Q_s_a[0])
                    c = G.step(guess)
                    c_prime = G_prime.step(guess)
                    if c != c_prime: print("P")

                d.append((Q_s_a[0], Q_s_a[1], s, guess, c))

                if len(d) > 1:
                    if not self.nn_model:
                        for i in range(len(self.weights)):
                            self.weights[i] += self.lr/(episode+1)*d[-2][1][i]*(d[-2][4] + d[-1][0] - d[-2][0])

            if not self.nn_model:
                features = np.array(G.get_features("aaaaa"))
                Q_terminal = np.dot(self.weights, features)
                grad = features
                d.append((Q_terminal, grad, 0))
                for i in range(len(self.weights)):
                    self.weights[i] += self.lr/(episode+1) * d[-2][1][i] * (d[-2][4] + d[-1][0] - d[-2][0])
                if (episode + 1) % 1000 == 0 or episode == 0:
                    print(episode + 1)
                    print(self.weights)
                    best = {}
                    print(self.score(0.025, best))
            else:
                c = d[-1][4]
                for i in range(len(d)):
                    datapoint = d[i]
                    Q = self.Q_tilde(datapoint[2], datapoint[3])[0]
                    self.optimizer.zero_grad()
                    loss = (Q - c)**2
                    loss.backward()
                    self.optimizer.step()
                if (episode+1)%1000 == 0 or episode == 0:
                    print(episode + 1)
                    best = {}
                    print(self.score(0.025, best))

        return best_actions

    def reccommend_only_params(self, s, s_prime):
        Q = np.inf
        rec = 'x'
        guesses = s.get_guesses()
        self.possible = self.allowed_words
        possible = self.get_possible(s_prime, self.possible)
        for a in possible:
            if a in guesses:
                continue
            Q_prime = self.Q_tilde(s, a)
            if Q_prime[0] <= Q:
                rec = a
                Q = Q_prime[0]
        return rec

    def reccommend_after_train(self, best_actions, s):
        hash = s.hashcode()
        if hash in best_actions:
            return best_actions[hash][0]
        else:
            return self.reccommend_only_params(s)

    def play(self, G, G_prime):
        r = 0
        while not G.is_done():
            a = self.reccommend_only_params(G, G_prime)
            #a = self.dummy(G, G_prime)
            r = G.step(a)
            r_prime = G_prime.step(a)
        return r

    def score(self, r, best_actions):
        total = 0
        correct = 0
        avg_time = 0
        for i in range(int(len(self.allowed_answers)*r)):
            #print(i/(int(len(self.allowed_answers)*r)))
            ans = self.allowed_answers_list[np.random.randint(0, len(self.allowed_answers_list))]
            total += 1
            board = Board()
            board_prime = Board()
            G = GameState(ans)
            G_prime = Game(ans, board_prime)
            c = self.play(G, G_prime)
            if 0 < c < 7:
                correct += 1
                avg_time += c
            if c >= 7:
                avg_time += 0

        return correct/total, avg_time/correct

    def dummy(self, s, s_prime):
        self.possible = self.allowed_words
        possible = self.get_possible(s_prime, self.possible)
        return possible[np.random.randint(0, len(possible))]





