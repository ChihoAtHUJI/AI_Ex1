from cmath import sqrt
from turtle import distance
from board import Board
from search import SearchProblem, ucs, astar
import util
import math
import numpy as np


class BlokusFillProblem(SearchProblem):
    """
    A one-player Blokus game as a search problem.
    This problem is implemented for you. You should NOT change it!
    """

    def __init__(self, board_w, board_h, piece_list, starting_point=(0, 0)):
        self.board = Board(board_w, board_h, 1, piece_list, starting_point)
        self.expanded = 0

    def get_start_state(self):
        """
        Returns the start state for the search problem
        """
        return self.board

    def is_goal_state(self, state):
        """
        state: Search state
        Returns True if and only if the state is a valid goal state
        """
        return not any(state.pieces[0])

    def get_successors(self, state):
        """
        state: Search state

        For a given state, this should return a list of triples,
        (successor, action, stepCost), where 'successor' is a
        successor to the current state, 'action' is the action
        required to get there, and 'stepCost' is the incremental
        cost of expanding to that successor
        """
        # Note that for the search problem, there is only one player - #0
        self.expanded = self.expanded + 1
        return [(state.do_move(0, move), move, 1) for move in state.get_legal_moves(0)]

    def get_cost_of_actions(self, actions):
        """
        actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.  The sequence must
        be composed of legal moves
        """
        return len(actions)



#####################################################
# This portion is incomplete.  Time to write code!  #
#####################################################
class BlokusCornersProblem(SearchProblem):
    def __init__(self, board_w, board_h, piece_list, starting_point=(0, 0), targets=[(0,0)]):
        self.expanded = 0
        "*** YOUR CODE HERE ***"
        self.board = Board(board_w, board_h, 1, piece_list, starting_point)

    def get_start_state(self):
        """
        Returns the start state for the search problem
        """
        return self.board

    def is_goal_state(self, state):
        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()
        return -1 not in [state.get_position(state.board_h - 1, state.board_w - 1),
                          state.get_position(state.board_h - 1, 0), state.get_position(0, state.board_w - 1),
                          state.get_position(0, 0)]

    def get_successors(self, state):
        """
        state: Search state

        For a given state, this should return a list of triples,
        (successor, action, stepCost), where 'successor' is a
        successor to the current state, 'action' is the action
        required to get there, and 'stepCost' is the incremental
        cost of expanding to that successor
        """
        # Note that for the search problem, there is only one player - #0
        self.expanded = self.expanded + 1
        return [(state.do_move(0, move), move, move.piece.get_num_tiles()) for move in state.get_legal_moves(0)]

    def get_cost_of_actions(self, actions):
        """
        actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.  The sequence must
        be composed of legal moves
        """
        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()
        return sum([move.piece.get_num_tiles() for move in actions])


def get_pieces_on_board(state):
    pieces = []
    for i in range(state.board_w):
        for j in range(state.board_h):
            if state.check_tile_attached(0, i, j):
                pieces += [[i, j]]
    return pieces


def blokus_corners_heuristic(state, problem):
    """
    Your heuristic for the BlokusCornersProblem goes here.

    This heuristic must be consistent to ensure correctness.  First, try to come up
    with an admissible heuristic; almost all admissible heuristics will be consistent
    as well.

    If using A* ever finds a solution that is worse uniform cost search finds,
    your heuristic is *not* consistent, and probably not admissible!  On the other hand,
    inadmissible or inconsistent heuristics may find optimal solutions, so be careful.
    """
    "*** YOUR CODE HERE ***"
    # the first heuristic function

    # pieces = get_pieces_on_board(state)
    # curr = [100, 100, 100]
    # for tile in pieces:
    #     a_1 = min(util.manhattanDistance([state.board_h -1, state.board_w -1], tile), curr[0])
    #     a_3 = min(util.manhattanDistance([0, state.board_w-1], tile), curr[1])
    #     a_4 = min(util.manhattanDistance([state.board_w-1, 0], tile), curr[2])
    #     curr = [a_1, a_3, a_4]
    # return sum(curr)
    problem.targets = [(0,0), (0, state.board_w-1), (state.board_w-1, 0), (state.board_h -1, state.board_w -1)]
    return blokus_cover_heuristic(state, problem)


class BlokusCoverProblem(SearchProblem):
    def __init__(self, board_w, board_h, piece_list, starting_point=(0, 0), targets=[(0, 0)]):
        self.targets = targets.copy()
        self.expanded = 0
        self.board = Board(board_w, board_h, 1, piece_list, starting_point)

    def get_start_state(self):
        """
        Returns the start state for the search problem
        """
        return self.board

    def is_goal_state(self, state):
        "*** YOUR CODE HERE ***"
        return -1 not in [state.get_position(target[1], target[0]) for target in self.targets]

    def get_successors(self, state):
        """
        state: Search state

        For a given state, this should return a list of triples,
        (successor, action, stepCost), where 'successor' is a
        successor to the current state, 'action' is the action
        required to get there, and 'stepCost' is the incremental
        cost of expanding to that successor
        """
        # Note that for the search problem, there is only one player - #0
        self.expanded = self.expanded + 1
        return [(state.do_move(0, move), move, move.piece.get_num_tiles()) for move in state.get_legal_moves(0)]

    def get_cost_of_actions(self, actions):
        """
        actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.  The sequence must
        be composed of legal moves
        """
        "*** YOUR CODE HERE ***"
        return sum([move.piece.get_num_tiles() for move in actions])


def blokus_cover_heuristic(state, problem):
    "*** YOUR CODE HERE ***"
    tiles = np.matrix(np.where(state.state == 0)).T
    total = 0
    for target in problem.targets:
        distance = abs(tiles - target)
        if distance.size == 0:  # if there aren't any distances
            return math.inf
        manhattan_dist = distance[:, 0] + distance[:, 1]
        condition = np.matrix(np.where(np.squeeze(np.array(manhattan_dist)) == np.min(manhattan_dist))).T # returns only the distances that are equal to the min distance
        min_values = distance[condition].tolist()
        value = np.min(manhattan_dist)

        if value == 1:
            return math.inf
        if any(t in min_values for t in [[value, 0], [0, value]]):
            total += 1
        else:
            total -= 1
        total += value

    return total



class ClosestLocationSearch:
    """
    In this problem you have to cover all given positions on the board,
    but the objective is speed, not optimality.
    """

    def __init__(self, board_w, board_h, piece_list, starting_point=(0, 0), targets=(0, 0)):
        self.expanded = 0
        self.targets = targets
        self.starting_point = starting_point
        "*** YOUR CODE HERE ***"
        self.board = Board(board_w, board_h, 1, piece_list, starting_point)

    def get_start_state(self):
        """
        Returns the start state for the search problem
        """
        return self.board

    def get_successors(self, state):
        self.expanded += 1
        return [(state.do_move(0, move), move, move.piece.get_num_tiles()) for move in state.get_legal_moves(0)]

    def is_goal_state(self, state):
        return -1 not in [state.get_position(target[1], target[0]) for target in self.targets]

    def closest_target(self, state, start_point):
        remaining_targets = [(x, y) for x, y in self.targets if state.get_position(y, x) == -1]
        if len(remaining_targets) == 0:
            return -1, -1
        distances = [0 for i in self.targets]
        for i, target in [len(remaining_targets), remaining_targets]:
            distances[i] = uclid_dist(target, start_point)

        return remaining_targets[distances.index(min(distances))]

    def solve(self):
        """
        This method should return a sequence of actions that covers all target locations on the board.
        This time we trade optimality for speed.
        Therefore, your agent should try and cover one target location at a time. Each time, aiming for the closest uncovered location.
        You may define helpful functions as you wish.

        Probably a good way to start, would be something like this --

        current_state = self.board.__copy__()
        backtrace = []

        while ....

            actions = set of actions that covers the closets uncovered target location
            add actions to backtrace

        return backtrace
        """
        "*** YOUR CODE HERE ***"
        result = []
        targets = self.targets
        point = self.starting_point
        problem = BlokusCoverProblem(self.board.board_w, self.board.board_h, self.board.piece_list, self.starting_point, self.targets)
        while targets:
            close = []
            for nearPoint in targets:
                if not close:
                    close.append(nearPoint)
                if util.manhattanDistance(point, close[-1]) > util.manhattanDistance(point, nearPoint):
                    close.append(nearPoint)
            target = close[-1]
            problem.targets = [target]
            actions = astar(problem, blokus_cover_heuristic)
            for action in actions:
                problem.board.add_move(0, action)
            result += actions
            point = target
            targets.remove(target)
            print(problem.expanded)

        # print(problem.board)
        # print(self.board)
        # self.board = problem.board.__copy__()
        # print(result)
        return result

class MiniContestSearch:
    """
    Implement your contest entry here
    """

    def __init__(self, board_w, board_h, piece_list, starting_point=(0, 0), targets=(0, 0)):
        self.targets = targets.copy()
        "*** YOUR CODE HERE ***"

    def get_start_state(self):
        """
        Returns the start state for the search problem
        """
        return self.board

    def solve(self):
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

def uclid_dist(xy1,xy2):
    return sqrt(pow(xy1[0] -xy2[0], 2) + pow(xy1[1] -xy2[1], 2))