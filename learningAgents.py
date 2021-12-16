import util
from game import Agent


class ReinforcementAgent(Agent):
    def update(self, state, action, nextState, reward):
        util.raiseNotDefined()

    def observeTransition(self, state, action, nextState, deltaReward):
        self.episodeRewards += deltaReward
        self.update(state, action, nextState, deltaReward)

    def stopEpisode(self):
        self.accumTrainRewards += self.episodeRewards
        self.episodesSoFar += 1


    def __init__(self, numTraining=500, epsilon=0.5, alpha=0.5, gamma=1):
        self.episodesSoFar = 0
        self.accumTrainRewards = 0.0
        self.numTraining = int(numTraining)
        self.alpha = float(alpha)
        self.epsilon = float(epsilon)
        self.discount = float(gamma)
        self.total_lose = 0

    def observationFunction(self, state):
        if self.lastState is not None:
            reward = state.getScore() - self.lastState.getScore()
            self.observeTransition(self.lastState, self.lastAction, state, reward)
        return state

    def registerInitialState(self, _):
        self.lastState = None
        self.lastAction = None
        self.episodeRewards = 0.0

    def final(self, state):
        deltaReward = state.getScore() - self.lastState.getScore()
        self.observeTransition(self.lastState, self.lastAction, state, deltaReward)
        self.stopEpisode()

        if 'lastWindowAccumRewards' not in self.__dict__:
            self.lastWindowAccumRewards = 0.0
        if 'lastloss' not in self.__dict__:
            self.lastloss = 0

        self.lastWindowAccumRewards += state.getScore()
        if state.isLose():
            self.total_lose += 1

        if self.episodesSoFar % 10 == 0:
            windowAvg = self.lastWindowAccumRewards / float(10)
            if self.episodesSoFar <= self.numTraining:
                trainAvg = self.accumTrainRewards / float(self.episodesSoFar)
                print(self.total_lose, self.total_lose - self.lastloss, trainAvg, windowAvg)
            self.lastWindowAccumRewards = 0.0
            self.lastloss = self.total_lose
