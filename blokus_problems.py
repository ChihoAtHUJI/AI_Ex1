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
        self.starting_point = starting_point
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
    problem.targets = [(0, state.board_w-1), (state.board_h-1, 0), (state.board_h -1, state.board_w -1)]
    tiles = np.matrix(np.where(state.state == 0)).T
    startingPt = problem.starting_point

    bound = state.board_h + state.board_w

    num_target_remain = len(problem.targets)
    total = 0
    # result = [0]
    for target in problem.targets:
        if not state.check_tile_legal(0, target[0], target[1]):
            num_target_remain -= 1
            # print(num_target_remain, "ho!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        distance = abs(tiles - target)
        if distance.size == 0:  # if there aren't any distances
            value = max(abs(target[0] - startingPt[0]), abs(target[1] - startingPt[1]))
        else:
            king_distance = distance.max(axis=1)
            value = np.min(king_distance)
        total += value
    if bound < total:
        return bound
    return total
    # return blokus_cover_heuristic(state, problem)


class BlokusCoverProblem(SearchProblem):
    def __init__(self, board_w, board_h, piece_list, starting_point=(0, 0), targets=[(0, 0)]):
        self.targets = targets
        self.expanded = 0
        self.starting_point = starting_point
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
        # piece_list = sorted(state.get_legal_moves(0), key= lambda x:x.piece.get_num_tiles())
        return [(state.do_move(0, move), move, move.piece.get_num_tiles()) for move in state.get_legal_moves(0)]

    def get_cost_of_actions(self, actions):
        """
        actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.  The sequence must
        be composed of legal moves
        """
        "*** YOUR CODE HERE ***"
        return sum([move.piece.get_num_tiles() for move in actions])

def king_dist(pt1, pt2):
    return max(abs(pt1[0]-pt2[0]), abs(pt1[1]-pt2[1]))




def blokus_cover_heuristic(state, problem):
    "*** YOUR CODE HERE ***"
    tiles = np.matrix(np.where(state.state == 0)).T
    startingpt = problem.starting_point
    lst = problem.targets.copy()
    lst.append(startingpt)
    lst.sort(key=lambda s: s[0])
    bound = lst[-1][0]-lst[0][0]
    lst.sort(key=lambda s: s[1])
    if bound < lst[-1][1] - lst[0][1]:
        bound = lst[-1][1] - lst[0][1]

    num_target_remain = len(problem.targets)
    total = 0
    result = [0]
    for target in problem.targets:
        if not state.check_tile_legal(0, target[0], target[1]):
            num_target_remain -= 1
            # print(num_target_remain , "ho!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        distance = abs(tiles - target)
        if distance.size == 0:  # if there aren't any distances
            value = max(abs(target[0]-startingpt[0]), abs(target[1]-startingpt[1]))
        else:
            king_distance = distance.max(axis=1)
            value = np.min(king_distance)
        total += value
    if bound < total:
        return bound
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

    def closest_target(self, state, targets):
        min_dist = state.board_w * state.board_h
        closest_target = targets[0]
        for target in targets:
            for x in range(state.board_w):
                for y in range(state.board_h):
                    if state.check_tile_legal(0, x, y) and state.connected[0][y][x]:  # if legal
                        dist = math.sqrt((target[0] - y) ** 2 + (target[1] - x) ** 2) / 2 # dist for new target
                        if dist < min_dist:
                            min_dist = dist
                            closest_target = target
        return closest_target

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
        # result = []
        # targets = self.targets
        # point = self.starting_point
        # problem = BlokusCoverProblem(self.board.board_w, self.board.board_h, self.board.piece_list, self.starting_point, self.targets)
        # while targets:
        #     close = []
        #     for nearPoint in targets:
        #         if not close:
        #             close.append(nearPoint)
        #         if util.manhattanDistance(point, close[-1]) > util.manhattanDistance(point, nearPoint):
        #             close.append(nearPoint)
        #     target = close[-1]
        #     problem.targets = [target]
        #     actions = ucs(problem)
        #     for action in actions:
        #         print(problem.board.add_move(0, action))
        #     result += actions
        #     point = target
        #     targets.remove(target)
        #     print(problem.expanded)
        #     self.expanded += problem.expanded
        # return result

        state = self.board.__copy__()
        backtrace = []
        while self.targets:
            target = self.closest_target(state, self.targets)
            self.targets.remove(target)
            problem = BlokusCoverProblem(state.board_w, state.board_h, state.piece_list, self.starting_point, [target])
            problem.board = state
            moves = ucs(problem)
            for move in moves:
                state = state.do_move(0, move)
                backtrace.append(move)
            self.expanded += problem.expanded
        return backtrace

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