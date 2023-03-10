# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

from typing import Set
import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def expand(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (child,
        action, stepCost), where 'child' is a child to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that child.
        """
        util.raiseNotDefined()

    def getActions(self, state):
        """
          state: Search state

        For a given state, this should return a list of possible actions.
        """
        util.raiseNotDefined()

    def getActionCost(self, state, action, next_state):
        """
          state: Search state
          action: action taken at state.
          next_state: next Search state after taking action.

        For a given state, this should return the cost of the (s, a, s') transition.
        """
        util.raiseNotDefined()

    def getNextState(self, state, action):
        """
          state: Search state
          action: action taken at state

        For a given state, this should return the next state after taking action from state.
        """
        util.raiseNotDefined()

    def getCostOfActionSequence(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]


def depthFirstSearch(problem):

    frontier = util.Stack()      #A stack of states
    expanded = set()             #A set of states
    pathUntilCurrState = dict()  #A dict with keys=states and values=list of actions

    frontier.push(problem.getStartState())
    pathUntilCurrState = {problem.getStartState() : []}

    while(not frontier.isEmpty()):
        state = frontier.pop()

        if(problem.isGoalState(state)):
            return pathUntilCurrState.get(state)

        if(state not in expanded):
            expanded.add(state)
            childsOfState = problem.expand(state)

            for childState,childAction,stepCost in childsOfState: #???? stepCost ?????? ???????????????????? ???????? ??????????????????
                frontier.push(childState) #???????????????????????? ???????? ???? state ?????? stack
                pathUntilCurrState[childState] = pathUntilCurrState.get(state) + [childAction]

    return None

def breadthFirstSearch(problem):
    frontier = util.Queue()      #A queue of states
    expanded = set()             #A set of states
    pathUntilCurrState = dict()  #A dict with keys=states and values=list of actions

    frontier.push(problem.getStartState())
    pathUntilCurrState = {problem.getStartState() : []}

    while(not frontier.isEmpty()):
        state = frontier.pop()

        if(problem.isGoalState(state)):
            return pathUntilCurrState.get(state)

        if(state not in expanded):
            expanded.add(state)
            childsOfState = problem.expand(state)

            for childState,childAction,stepCost in childsOfState: #???? stepCost ?????? ???????????????????? ???????? ??????????????????
                frontier.push(childState) #???????????????????????? ???????? ???? state ?????? queue
                if childState not in pathUntilCurrState :
                    pathUntilCurrState[childState] = pathUntilCurrState.get(state) + [childAction]
    return None

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    frontier = util.PriorityQueue()      #A Pqueue of states
    expanded = set()                     #A set of states
    pathUntilCurrState = dict()          #A dict with keys=states and values=list of actions

    frontier.push(problem.getStartState(),0)                   #priority = 0
    pathUntilCurrState = {problem.getStartState() : ([],0)}    #cost = 0

    while(not frontier.isEmpty()):
        state = frontier.pop()

        if(problem.isGoalState(state)):
            return pathUntilCurrState.get(state)[0]

        if(state not in expanded):
            expanded.add(state)
            childsOfState = problem.expand(state)

            for childState,childAction,stepCost in childsOfState:
                newCost = stepCost + pathUntilCurrState.get(state)[1]
                if(childState not in pathUntilCurrState or newCost < pathUntilCurrState.get(childState)[1]):
                    priority = newCost + heuristic(childState,problem)
                    frontier.push(childState, priority)
                    pathUntilCurrState[childState] = (pathUntilCurrState.get(state)[0] + [childAction], newCost)

    return None

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
