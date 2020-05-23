
import random


class MemoryInstance:
    """ remember a specific state -> action -> reward, next_state training example """
    def __init__(self, state, action, reward, next_state):
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state

    def asTuple(self):
        """ Returns memory instance as a length 4 tuple """
        return (self.state, self.action, self.reward, self.next_state)



class Memory:
    """ Memory of recent memory_instances (training examples) that the agent has encountered """
    def __init__(self, max_memory):
        self.max_memory = max_memory
        self.samples = []
        

    def add_sample(self, memory_instance):
        """ Adds a memory_instance sample in queue fashion (deletes old ones) """
        self.samples.append(memory_instance.asTuple())
        if len(self.samples) > self.max_memory:
            del self.samples[0]

    def sample(self, no_samples):
        """ Randomly samples no_samples from recent memory, or all of the samples if there aren't enough"""
        if no_samples > len(self.samples):
            return random.sample(self.samples, len(self.samples))
        else:
            return random.sample(self.samples, no_samples)