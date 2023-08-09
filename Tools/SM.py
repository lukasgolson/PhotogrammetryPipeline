import pickle

from statemachine import StateMachine


class SerializableStateMachine(StateMachine):
    def __init__(self, filename=None):
        super().__init__()
        self.current_state = self.initial_state
        self.filename = filename

        if self.filename:
            try:
                self.deserialize(self.filename)
            except FileNotFoundError:
                pass

    def serialize(self, filename):
        with open(filename, 'wb') as file:
            pickle.dump(self.current_state, file)

    def deserialize(self, filename):
        with open(filename, 'rb') as file:
            self.current_state = pickle.load(file)

    def transition_to(self, next_state):
        super().transition_to(next_state)
        if self.filename:
            self.serialize(self.filename)
