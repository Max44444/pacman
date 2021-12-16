import util

from game import *
from learningAgents import ReinforcementAgent


class PacmanQAgent(ReinforcementAgent):
    def __init__(self, epsilon=0.65, gamma=0.8, alpha=0.2, numTraining=0):
        ReinforcementAgent.__init__(self, epsilon=epsilon, gamma=gamma, alpha=alpha, numTraining=numTraining)

        self.q_values = util.Counter()
        self.discarded = []
        self.num_discarded = 0

    def getQValue(self, state, action):
        return self.q_values[(state, action)] if (state, action) in self.q_values else -1.0

    def computeValueFromQValues(self, state):
        action_list = state.getLegalActions()

        if not action_list:
            return 0.0

        return max([self.getQValue(state, action) for action in action_list])

    def computeActionFromQValues(self, state):
        max_q = self.computeValueFromQValues(state)
        action_list = state.getLegalActions()

        if not action_list:
            return None

        max_action_list = [action for action in action_list if self.getQValue(state, action) == max_q]

        return random.choice(max_action_list)

    def getAction(self, state):
        legalActions = state.getLegalActions()

        if not legalActions:
            action = None
        elif util.flipCoin(self.epsilon):
            action = random.choice(legalActions)
        else:
            action = self.computeActionFromQValues(state)

        self.lastState = state
        self.lastAction = action
        return action

    def update(self, state, action, nextState, reward):
        max_q = self.computeValueFromQValues(nextState)
        td_target = reward + self.discount * max_q
        td_delta = td_target - self.getQValue(state, action)

        self.q_values[(state, action)] = self.getQValue(state, action) + self.alpha * td_delta
