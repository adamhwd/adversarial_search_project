# adversarialAgents.py
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
#
# Modified for use at University of Bath.


from util import manhattanDistance
from game import Directions
import random, util

from math import sqrt
from game import Agent


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.
    """


    def getAction(self, gameState):
        """
        getAction chooses among the best options according to the evaluation
        function.

        getAction takes a GameState and returns some Directions.X
        for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action)
                  for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores))
                       if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are
        better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        return successorGameState.getScore()


def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()


class AdversarialSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    adversarial searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent and AlphaBetaPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(AdversarialSearchAgent):
    """
    Your minimax agent (question 1)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing
        minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """

        "*** YOUR CODE HERE ***"

        bestScore = -1000000
#        print(f"Depth of {self.depth}")
#        print(f"There are {gameState.getNumAgents()} agents")

        for action in gameState.getLegalActions(0):
            currentScore = self.minValue(gameState.generateSuccessor(0, action), 1)
            if currentScore >= bestScore:
                bestAction = action
                bestScore = currentScore
#        print(f"Minimax value: {bestScore}")
        return bestAction


    def maxValue(self, gameState, recursionLayer):
        playerIndex = recursionLayer % gameState.getNumAgents()
#        print(f"MaxValue Recursion layer:{recursionLayer} Player index: {playerIndex}")
        if gameState.isLose() or gameState.isWin() or recursionLayer == self.depth * gameState.getNumAgents():
            return self.evaluationFunction(gameState)
        if playerIndex != 0:
            v = self.minValue(gameState, recursionLayer)
            return v
        else:
            v = -1000000
            for action in gameState.getLegalActions(playerIndex):
                v = max(v, self.minValue(gameState.generateSuccessor(playerIndex, action), recursionLayer + 1))
#        print(f"{recursionLayer}: Max Search returned {v}, playerIndex={playerIndex}")
        return v

    def minValue(self, gameState, recursionLayer):
        playerIndex = recursionLayer % gameState.getNumAgents()
#        print(f"MinValue Recursion layer:{recursionLayer} Player index: {playerIndex}")
        if gameState.isLose() or gameState.isWin() or recursionLayer == self.depth * gameState.getNumAgents():
            return self.evaluationFunction(gameState)
        v = 1000000
        for action in gameState.getLegalActions(playerIndex):
            v = min(v, self.maxValue(gameState.generateSuccessor(playerIndex, action), recursionLayer + 1))
#        print(f"{recursionLayer}: Min Search returned {v}, playerIndex={playerIndex}")
        return v


class AlphaBetaAgent(AdversarialSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 2)
    """
    alphaBetaList = []

    def getAction(self, gameState):
        """
        Returns the minimax with alpha-beta pruning action using self.depth and
        self.evaluationFunction
        """

        "*** YOUR CODE HERE ***"
        alphaBetaList = [-1000000] + [1000000 for x in range(1, gameState.getNumAgents())]

        bestAction = self.maxValue(gameState, 0, alphaBetaList, True)
        return bestAction

    def maxValue(self, gameState, recursionLayer, alphaBetaList, returnAction=False):
        playerIndex = recursionLayer % gameState.getNumAgents()
        if gameState.isLose() or gameState.isWin() or recursionLayer == self.depth * gameState.getNumAgents():
            return self.evaluationFunction(gameState)
        if playerIndex != 0:
            v = self.minValue(gameState, recursionLayer, alphaBetaList)
            return v
        else:
            bestAction = ""
            v = -1000000
            for action in gameState.getLegalActions(playerIndex):
                vCandidate = self.minValue(gameState.generateSuccessor(playerIndex, action), recursionLayer + 1, alphaBetaList.copy())
                if vCandidate > v:
                    v = vCandidate
                    bestAction = action
                if v > alphaBetaList[1]:
                    return v
                alphaBetaList[0] = max(alphaBetaList[0], v)
        if returnAction:
            return bestAction
        else:
            return v

    def minValue(self, gameState, recursionLayer, alphaBetaList):
        playerIndex = recursionLayer % gameState.getNumAgents()
        if gameState.isLose() or gameState.isWin() or recursionLayer == self.depth * gameState.getNumAgents():
            return self.evaluationFunction(gameState)
        v = 1000000
        for action in gameState.getLegalActions(playerIndex):
            v = min(v, self.maxValue(gameState.generateSuccessor(playerIndex, action), recursionLayer + 1, alphaBetaList.copy()))
            if v < alphaBetaList[0]:
                return v
            alphaBetaList[1] = min(alphaBetaList[1], v)
        return v



def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 3).

    DESCRIPTION: <write something here so we know what you did>
    """

    "*** YOUR CODE HERE ***"
    pacmanPos = currentGameState.getPacmanPosition()
    foodPositions = currentGameState.getFood()
    ghostStates = currentGameState.getGhostStates()
    ghostPositions = currentGameState.getGhostPositions()
    scaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]

    FOOD_DISTANCE_WEIGHT = 1.5
    SCARED_GHOST_WEIGHT = 1
    GHOST_DISTANCE_WEIGHT = 1
    score = currentGameState.getScore()

    ghostIsScared = False
    for item in scaredTimes:
        if item > 0:
            ghostIsScared = True
    if ghostIsScared:
        score += 100

    for i in range(len(ghostPositions)):
        ghostMD = manhattanDistance(pacmanPos, ghostPositions[i])
        if ghostMD < 4 and 0 < scaredTimes[i] < 10:
            score += 1000
            score -= ghostMD * SCARED_GHOST_WEIGHT
        elif ghostMD < 2:
            score += ghostMD * GHOST_DISTANCE_WEIGHT

    for item in foodPositions.asList():
        foodPos = list(item)
        foodMD = manhattanDistance(pacmanPos, foodPos)
        score -= sqrt(foodMD) * FOOD_DISTANCE_WEIGHT


    return score

def manhattanDistance(pos1, pos2):
    return abs(pos1[0]-pos2[0]) + abs(pos1[1]-pos2[1])


