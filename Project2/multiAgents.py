# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and child states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed child
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        childGameState = currentGameState.getPacmanNextState(action)
        newPos = childGameState.getPacmanPosition()
        newFood = childGameState.getFood()
        newGhostStates = childGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        score = 0
        foodList = newFood.asList()
        # If in the next position a ghost exists near us which is not scared then dont go
        for ghost in newGhostStates:
            ghostDistance = util.manhattanDistance(newPos, ghost.getPosition())
            if int(ghostDistance) in range(1,5) and ghost.scaredTimer == 0:
                score -= float(1/ghostDistance)

        #we find all the remaining food
        foodDistance = []
        for food in foodList:
            foodDistance.append(util.manhattanDistance(newPos, food))

        #if there is food left -> select the one which is the closest
        if foodDistance:
            closestFood = min(foodDistance)
            #We reciprocal the distance of the closest food
            score += float(1/closestFood)
        
        return childGameState.getScore() + score

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.getNextState(agentIndex, action):
        Returns the child game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        def maxValue(gameState ,depth):
            #we check to see if we are in a terminal state and if so we return the utility(state)
            if gameState.isWin() or gameState.isLose() or self.depth == depth:
                return (self.evaluationFunction(gameState),)
            
            actions = gameState.getLegalActions(0)
            maxEval = (float('-inf'),)

            #for each action of the max player (pacman)
            #we try and find the maxinum evaluation of the min players(ghosts)
            #We store the value and the action
            for action in actions:
                nextState = gameState.getNextState(0,action)
                maxEval = max((minValue(nextState,depth, 1),action),maxEval)
            return maxEval

        def minValue(gameState, depth, agent):
            #we check to see if we are in a terminal state and if so we return the utility(state)
            if gameState.isWin() or gameState.isLose() or self.depth == depth:
                return self.evaluationFunction(gameState)

            actions = gameState.getLegalActions(agent)
            minEval = float('inf')
            numGhosts = gameState.getNumAgents() - 1

            #for each action of the min players (ghosts)
            #we try and find the mininum evaluation of the max player(pacman)
            #We store the only the value
            for action in actions:
                nextState = gameState.getNextState(agent,action)
                #if there is no other min players we continue to the next depth with the max player
                if agent + 1 > numGhosts:
                    minEval = min(maxValue(nextState,depth+1)[0],minEval)
                #else we evaluate the other min players
                else:
                    minEval = min(minValue(nextState,depth,agent+1),minEval)
            return minEval
        
        return maxValue(gameState, 0)[1] #we return only the action

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        def maxValue(gameState, depth, alpha, beta):
            if gameState.isWin() or gameState.isLose() or self.depth == depth:
                return (self.evaluationFunction(gameState),)
            
            actions = gameState.getLegalActions(0)
            maxEval = (float('-inf'),)

            for action in actions:
                nextState = gameState.getNextState(0,action)
                maxEval = max((minValue(nextState,depth, 1, alpha, beta),action),maxEval)
                #we update alpha
                alpha = max(maxEval[0], alpha)
                #pruning occurs
                if beta < alpha:
                    break
            return maxEval

        def minValue(gameState, depth, agent, alpha, beta):
            if gameState.isWin() or gameState.isLose() or self.depth == depth:
                return self.evaluationFunction(gameState)

            actions = gameState.getLegalActions(agent)
            minEval = float('inf')
            numGhosts = gameState.getNumAgents() - 1

            for action in actions:
                nextState = gameState.getNextState(agent,action)
                if agent + 1 > numGhosts:
                    minEval = min(maxValue(nextState,depth+1, alpha, beta)[0],minEval)
                else:
                    minEval = min(minValue(nextState,depth,agent+1, alpha, beta),minEval)
                #we update beta
                beta = min(beta, minEval)
                #pruning occurs
                if beta < alpha:
                    break
            return minEval
        alpha = float('-inf')
        beta = float('inf')
        return maxValue(gameState, 0, alpha, beta)[1]

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        def maxValue(gameState ,depth):
            if gameState.isWin() or gameState.isLose() or self.depth == depth:
                return (self.evaluationFunction(gameState),)
            
            actions = gameState.getLegalActions(0)
            maxEval = (float('-inf'),)

            for action in actions:
                nextState = gameState.getNextState(0,action)
                maxEval = max((chanceValue(nextState,depth, 1),action),maxEval)
            return maxEval

        def chanceValue(gameState, depth, agent):
            if gameState.isWin() or gameState.isLose() or self.depth == depth:
                return self.evaluationFunction(gameState)

            actions = gameState.getLegalActions(agent)
            chanceEval = 0
            numGhosts = gameState.getNumAgents() - 1

            #Here we dont find the min value but we sum and get the average
            for action in actions:
                nextState = gameState.getNextState(agent,action)
                if agent + 1 > numGhosts:
                    chanceEval += maxValue(nextState,depth+1)[0]
                else:
                    chanceEval += chanceValue(nextState,depth,agent+1)
            return chanceEval/len(actions)
        
        return maxValue(gameState, 0)[1]

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    pos = currentGameState.getPacmanPosition()
    foodList = currentGameState.getFood().asList()
    ghostStates = currentGameState.getGhostStates()
    capsules = currentGameState.getCapsules()

    score = 0

    #check for each ghost
    for ghost in ghostStates:
        ghostPos = ghost.getPosition()
        ghostDistance = util.manhattanDistance(pos,ghostPos)

        #if it is close (1 to 5 moves) and scared (for atleast 5 more moves) give high score
        if int(ghostDistance) in range(1,6) and ghost.scaredTimer  >= 5:
            score += 10*(1/float(ghostDistance))
        #if it is next to us and not scared give low score    
        elif ghostDistance == 1  and ghost.scaredTimer  <= 0:
            score -= 10*(1/float(ghostDistance))
        #if it is far from us we dont care about its state and we give average score
        elif ghostDistance != 0:
            score += (1/float(ghostDistance))

    foodDistance = []
    capsulesDistance = []

    for food in foodList:
        foodDistance.append(util.manhattanDistance(pos, food))

    for capsule in capsules:
        capsulesDistance.append(util.manhattanDistance(pos, capsule))
    
    #if there is food left -> select the one which is the closest
    if foodDistance:
        closestFood = min(foodDistance)
        score += float(1/closestFood)
    
    #if there is a capsule near we give high score cause it may help with eating the ghost
    if capsulesDistance:
        closestCapsule = min(capsulesDistance)
        score += 5*float(1/closestCapsule)
    
    return currentGameState.getScore() + score

# Abbreviation
better = betterEvaluationFunction
