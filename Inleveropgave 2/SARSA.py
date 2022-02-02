from dataclasses import dataclass
from matplotlib import pyplot as plt
import random
import heatmap
from tqdm import tqdm


class Maze:
    def __init__(self):
        # stelt een grid op met states aan de hand van de gegeven afbeelding
        self.grid = {}
        for y in range(4):
            for x in range(4):
                if [x, y] == [3, 0]:
                    self.grid[x, y] = State(x=x, y=y, reward=40, Q={'↑': 0, '→': 0, '↓': 0, '←': 0}, finished=True)
                elif [x, y] == [2, 1] or [x, y] == [3, 1]:
                    self.grid[x, y] = State(x=x, y=y, reward=-10, Q={'↑': 0, '→': 0, '↓': 0, '←': 0}, finished=False)
                elif [x, y] == [0, 3]:
                    self.grid[x, y] = State(x=x, y=y, reward=10, Q={'↑': 0, '→': 0, '↓': 0, '←': 0}, finished=True)
                elif [x, y] == [1, 3]:
                    self.grid[x, y] = State(x=x, y=y, reward=-2, Q={'↑': 0, '→': 0, '↓': 0, '←': 0}, finished=False)
                else:
                    self.grid[x, y] = State(x=x, y=y, reward=-1, Q={'↑': 0, '→': 0, '↓': 0, '←': 0}, finished=False)
        self.actions = {'↑': (0, -1), '→': (1, 0), '↓': (0, 1), '←': (-1, 0)}

    def step(self, state, action):
        # step neemt de gegeven action en state, en returnt de state dat dit opleverd
        move = self.actions[action]
        location = (state.x, state.y)
        newlocation = (location[0] + move[0], location[1] + move[1])
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
    Q: dict
    finished: bool


class Agent:
    def __init__(self, start: tuple):
        # initialiseerd de maze en de policy, en runt het algoritme
        self.maze = Maze()
        self.location = self.maze.grid[start]
        self.discount = 0.9
        self.epsilon = 0.1
        self.policy = Policy(self.epsilon)
        self.policy.state_grid = self.maze.grid
        self.SARSA()

    def step(self, move=None):
        # de step functie verplaats de agent in de maze
        # aan de hand van de huidige policy
        if not move:
            move = self.choice(self.location)
        newlocation = self.maze.step(self.location, move)
        self.location = newlocation

    def choice(self, state):
        # choice vraagt de action voor de huidige state op uit de policy
        return self.policy.select_action(state)

    def SARSA(self):
        # de functie bevat het algoritme zoals hij in de pseudocode is beschreven
        learning_rate = 0.1
        for episode_num in tqdm(range(100000)):
            # print(episode_num)
            self.location = random.choice(list(self.maze.grid.values()))
            location = self.location
            while not self.location.finished:
                action = self.choice(self.location)
                self.step(action)
                newlocation = self.location
                newaction = self.choice(self.location)
                reward = self.location.reward
                location.Q[action] += learning_rate * (reward +
                                                       (self.discount * newlocation.Q[newaction]) -
                                                       location.Q[action])
                self.policy.create_state_policy(location, self.epsilon)
                location = newlocation

        # code om de values te visualiseren, en de policy te visualiseren
        printgrid = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
        for key in self.maze.grid.keys():
            printgrid[key[1]][key[0]] = self.maze.grid[key].Q
        [print(i) for i in printgrid]
        print()
        self.policy.print_policy()


class Policy:
    def __init__(self, epsilon):
        # policy wordt geinitialiseerd met een e-soft policy op k = 0
        self.actions = {(0, -1): '↑', (1, 0): '→', (0, 1): '↓', (-1, 0): '←'}
        self.policy_grid = {(x, y): {'↑': epsilon / 4, '→': epsilon / 4, '↓': epsilon / 4, '←': epsilon / 4}
                            for x in range(4) for y in range(4)}
        self.state_grid = {}

    def create_state_policy(self, state, e):
        # bepaalt de policy voor de gegeven state aan de hand van de Q-table
        max_value = max(state.Q.values())
        max_key = [key for key, value in state.Q.items() if value == max_value][-1]
        for key in self.policy_grid[state.x, state.y].keys():
            if key == max_key:
                self.policy_grid[state.x, state.y][key] = 1 - e + e / 4
            else:
                self.policy_grid[state.x, state.y][key] = e / 4

    def select_action(self, state):
        # haalt de action uit de policy op voor de gegeven state, en returnt deze
        key = (state.x, state.y)
        return random.choices(population=(list(self.policy_grid[key].keys())),
                              weights=(list(self.policy_grid[key].values())))[0]

    def print_policy(self):
        # visualiseert de q-table policy met behulp van heatmap.py.
        # Source: https://stackoverflow.com/questions/66048529/how-to-create-a-heatmap-where-each-cell-is-divided-into-4-triangles
        M, N = 4, 4  # e.g. 5 columns, 4 rows
        values = heatmap.create_demo_data(M, N, self.state_grid)
        triangul = heatmap.triangulation_for_triheatmap(M, N)
        norms = [plt.Normalize(-0.5, 1) for _ in range(4)]
        fig, ax = plt.subplots()
        imgs = [ax.tripcolor(t, val.ravel(), cmap='RdYlGn', vmin=0, vmax=1, ec='white')
                for t, val in zip(triangul, values)]
        for val, dir in zip(values, [(-1, 0), (0, 1), (1, 0), (0, -1)]):
            for i in range(M):
                for j in range(N):
                    v = val[j, i]
                    ax.text(i + 0.3 * dir[1], j + 0.3 * dir[0], f'{v:.2f}', color='k' if 0.2 < v < 0.8 else 'w',
                            ha='center', va='center')
        cbar = fig.colorbar(imgs[0], ax=ax)

        ax.set_xticks(range(M))
        ax.set_yticks(range(N))
        ax.invert_yaxis()
        ax.margins(x=0, y=0)
        ax.set_aspect('equal', 'box')  # square cells
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    agent = Agent((3, 2))
    # agent.step()
