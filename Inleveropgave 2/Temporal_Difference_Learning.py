from dataclasses import dataclass
import random


class Maze:
    def __init__(self):
        # stelt een grid op met states aan de hand van de gegeven afbeelding
        self.grid = {}
        for y in range(4):
            for x in range(4):
                if [x, y] == [3, 0]:
                    self.grid[x, y] = State(x=x, y=y, reward=40, value=0, finished=True)
                elif [x, y] == [2, 1] or [x, y] == [3, 1]:
                    self.grid[x, y] = State(x=x, y=y, reward=-10, value=0, finished=False)
                elif [x, y] == [0, 3]:
                    self.grid[x, y] = State(x=x, y=y, reward=10, value=0, finished=True)
                elif [x, y] == [1, 3]:
                    self.grid[x, y] = State(x=x, y=y, reward=-2, value=0, finished=False)
                else:
                    self.grid[x, y] = State(x=x, y=y, reward=-1, value=0, finished=False)
        self.actions = {'↑': (0, -1), '→': (1, 0), '↓': (0, 1), '←': (-1, 0)}

    def step(self, state, action):
        # step neemt de gegeven action en state, en returnt de state dat dit opleverd
        move = self.actions[action]
        location = (state.x, state.y)
        newlocation = (location[0]+move[0], location[1]+move[1])
        if newlocation in self.grid.keys():
            return self.grid[newlocation]
        return state


@dataclass
class State:
    # de dataclass bevat de data op die een state
    # nodig heeft om t algoritme uit te voeren
    x: int
    y: int
    reward: int
    value: float
    finished: bool


class Agent:
    def __init__(self, start: tuple):
        # initialiseerd de maze en de policy, en runt het algoritme
        self.maze = Maze()
        self.location = self.maze.grid[start]
        self.discount = 0.9
        self.policy = Policy()
        self.policy.state_grid = self.maze.grid
        self.temporal_difference_learning()

    def step(self):
        # de step functie verplaats de agent in de maze
        # aan de hand van de huidige policy
        move = self.choice(self.location)
        newlocation = self.maze.step(self.location, move)
        self.location = newlocation

    def choice(self, state):
        # choice vraagt de action voor de huidige state op uit de policy
        return self.policy.select_action(state)

    def temporal_difference_learning(self):
        # de functie bevat het algoritme zoals hij in de pseudocode is beschreven
        learning_rate = 0.1
        for episode_num in range(100000):
            print(episode_num)
            self.location = random.choice(list(self.maze.grid.values()))
            while not self.location.finished:
                location = self.location
                self.step()
                new_location = self.location
                location.value = location.value + learning_rate * (new_location.reward + (self.discount * new_location.value) - location.value)
                self.policy.create_policy()
        # code om de values te visualiseren, en de policy te visualiseren
        printgrid = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
        for key in self.maze.grid.keys():
            printgrid[key[1]][key[0]] = self.maze.grid[key].value
        [print(i) for i in printgrid]
        print()
        self.policy.print_policy()


class Policy:
    def __init__(self):
        # policy wordt geinitialiseerd met een random policy op k = 0
        self.actions = {(0, -1): '↑', (1, 0): '→', (0, 1): '↓', (-1, 0): '←'}
        self.policy_grid = {(x, y): ['↑', '→', '↓', '←'] for x in range(4) for y in range(4)}
        self.state_grid = {}

    def create_policy(self):
        # bepaalt de policy voor de grid aan de hand van de results in de omgeving
        for location in self.state_grid.keys():
            if not self.state_grid[location].finished:
                neighbors = [(location[0] - 1, location[1]), (location[0], location[1] + 1),
                             (location[0] + 1, location[1]),
                             (location[0], location[1] - 1)]
                results = {neighbor: self.state_grid[neighbor].reward + self.state_grid[neighbor].value
                           if neighbor in self.state_grid.keys()
                           else float('-inf')
                           for neighbor in neighbors}

                max_value = max(results.values())
                max_key = [key for key, value in results.items() if value == max_value]
                move = [self.actions[(key[0]-location[0], key[1]-location[1])] for key in max_key]
                self.policy_grid[location] = move
            else:
                self.policy_grid[location] = '○'

    def select_action(self, state):
        # haalt de action uit de policy op voor de gegeven state, en returnt deze
        key = (state.x, state.y)
        return random.choice(self.policy_grid[key])

    def print_policy(self):
        # visualiseert de policy in een grid
        printgrid = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
        print('Policy:')
        for key in self.policy_grid.keys():
            printgrid[key[1]][key[0]] = self.policy_grid[key]
        [print(i) for i in printgrid]


if __name__ == "__main__":
    agent = Agent((3, 2))
    # agent.step()
