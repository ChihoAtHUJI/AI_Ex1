"""
In search.py, you will implement generic search algorithms
"""

import util


class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def get_start_state(self):
        """
        Returns the start state for the search problem
        """
        util.raiseNotDefined()

    def is_goal_state(self, state):
        """
        state: Search state

        Returns True if and only if the state is a valid goal state
        """
        util.raiseNotDefined()

    def get_successors(self, state):
        """
        state: Search state

        For a given state, this should return a list of triples,
        (successor, action, stepCost), where 'successor' is a
        successor to the current state, 'action' is the action
        required to get there, and 'stepCost' is the incremental
        cost of expanding to that successor
        """
        util.raiseNotDefined()

    def get_cost_of_actions(self, actions):
        """
        actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.  The sequence must
        be composed of legal moves
        """
        util.raiseNotDefined()


class Node:
    def __init__(self, state, action=None, parent=None, cost_to_here=0, params={}):
        self.state = state
        self.action = action
        self.parent = parent
        self.cost_to_here = cost_to_here
        self.params = params

    def get_path(self):
        path = []
        node = self
        while node.parent is not None:
            path = [node.action] + path
            node = node.parent

        return path


def depth_first_search(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches
    the goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

	print("Start:", problem.get_start_state().state)
    print("Is the start a goal?", problem.is_goal_state(problem.get_start_state()))
    print("Start's successors:", problem.get_successors(problem.get_start_state()))
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()
    fringe = util.Stack()
    start = problem.get_start_state()
    fringe.push(Node(start))
    visited = set()

    while not fringe.isEmpty():
        node = fringe.pop()

        if problem.is_goal_state(node.state):
            print("Found solution!\n")
            return node.get_path()
        elif node.state not in visited:
            successors = problem.get_successors(node.state)

            for successor, action, cost in successors:
                cost_to_here = node.cost_to_here + cost
                fringe.push(Node(successor, action, node, cost_to_here))
                visited.add(node.state)
    return []


def breadth_first_search(problem):
    """
    Search the shallowest nodes in the search tree first.
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()
    fringe = util.Queue()
    start = problem.get_start_state()
    fringe.push(Node(start))
    visited = set()

    while not fringe.isEmpty():
        node = fringe.pop()

        if problem.is_goal_state(node.state):
            print("Found solution!\n")
            return node.get_path()
        elif node.state not in visited:
            successors = problem.get_successors(node.state)

            for successor, action, cost in successors:
                cost_to_here = node.cost_to_here + cost
                fringe.push(Node(successor, action, node, cost_to_here))
                visited.add(node.state)
    return []


def uniform_cost_search(problem):
    """
    Search the node of least total cost first.
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()
    # ucs is as A* search but with the null heuristic
    return a_star_search(problem)


def null_heuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def a_star_search(problem, heuristic=null_heuristic):
    """
    Search the node that has the lowest combined cost and heuristic first.
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()
    fringe = util.PriorityQueue()
    start = problem.get_start_state()
    fringe.push(Node(start), 0)
    visited = set()

    while not fringe.isEmpty():
        node = fringe.pop()

        if problem.is_goal_state(node.state):
            print("Found solution!\n")
            return node.get_path()
        elif node.state not in visited:
            successors = problem.get_successors(node.state)

            for successor, action, cost in successors:
                cost_to_here = node.cost_to_here + cost
                fringe.push(Node(successor, action, node, cost_to_here), cost_to_here + heuristic(node.state, problem))
                visited.add(node.state)
    return []



# Abbreviations
bfs = breadth_first_search
dfs = depth_first_search
astar = a_star_search
ucs = uniform_cost_search
