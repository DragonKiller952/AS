from dataclasses import dataclass


class Maze():
    def __init__(self):
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
        move = self.actions[action]
        location = (state.x, state.y)
        newlocation = (location[0]+move[0], location[1]+move[1])
        if newlocation in self.grid.keys():
            return self.grid[newlocation]
        return state


@dataclass
class State:
    x: int
    y: int
    reward: int
    value: float
    finished: bool


class Agent:
    def __init__(self, start: tuple):
        self.maze = Maze()
        self.location = self.maze.grid[start]
        self.discount = 1
        self.policy = Policy()
        self.policy.state_grid = self.maze.grid
        self.value_iteration()

    def step(self):
        print('Current Location:')
        print((self.location.x, self.location.y))
        move = self.choice(self.location)
        newlocation = self.maze.step(self.location, move)
        print('New Location:')
        print((newlocation.x, newlocation.y))
        self.location = newlocation

    def choice(self, state):
        return self.policy.select_action(state)

    def value_iteration(self):
        delta = float('inf')
        k = 0
        while delta > 0.1:
            k += 1
            newvalues = {(x, y): 0 for x in range(4) for y in range(4)}
            curdelta = float('-inf')
            for location in self.maze.grid.keys():
                if not self.maze.grid[location].finished:
                    neighbors = [(location[0] - 1, location[1]), (location[0], location[1] + 1),
                                 (location[0] + 1, location[1]),
                                 (location[0], location[1] - 1)]
                    results = [self.maze.grid[neighbor].reward + (self.discount * self.maze.grid[neighbor].value)
                               if neighbor in self.maze.grid.keys()
                               else self.maze.grid[location].reward + (self.discount * self.maze.grid[location].value)
                               for neighbor in neighbors]
                    newvalue = max(results)
                    newvalues[location] = newvalue
                    newdelta = abs(self.maze.grid[location].value - newvalue)
                    if newdelta > curdelta:
                        curdelta = newdelta
                else:
                    newvalues[location] = self.maze.grid[location].value
            for key in newvalues.keys():
                self.maze.grid[key].value = newvalues[key]
            delta = curdelta
            print(f'k={k}')
            print('delta', delta)
            printgrid = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
            for key in self.maze.grid.keys():
                printgrid[key[1]][key[0]] = self.maze.grid[key].value
            [print(i) for i in printgrid]
        self.policy.create_policy()


class Policy:
    def __init__(self):
        self.actions = {(0, -1): '↑', (1, 0): '→', (0, 1): '↓', (-1, 0): '←'}
        self.policy_grid = {(x, y): 'X' for x in range(4) for y in range(4)}
        self.state_grid = {}

    def create_policy(self):
        for location in self.state_grid.keys():
            if not self.state_grid[location].finished:
                neighbors = [(location[0] - 1, location[1]), (location[0], location[1] + 1),
                             (location[0] + 1, location[1]),
                             (location[0], location[1] - 1)]
                results = {neighbor: self.state_grid[neighbor].reward + self.state_grid[neighbor].value
                           if neighbor in self.state_grid.keys()
                           else float('-inf')
                           for neighbor in neighbors}
                # max_key = max(results, key=results.get)
                max_value = max(results.values())
                max_key = [key for key, value in results.items() if value == max_value][-1]
                move = self.actions[(max_key[0]-location[0], max_key[1]-location[1])]
                self.policy_grid[location] = move
            else:
                self.policy_grid[location] = '○'
        printgrid = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
        print('Policy:')
        for key in self.policy_grid.keys():
            printgrid[key[1]][key[0]] = self.policy_grid[key]
        [print(i) for i in printgrid]

    def select_action(self, state):
        key = (state.x, state.y)
        return self.policy_grid[key]


if __name__ == "__main__":
    agent = Agent((3, 2))
    agent.step()
