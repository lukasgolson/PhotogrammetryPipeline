import pickle

from statemachine import StateMachine


class SerializableStateMachine(StateMachine):
    def __init__(self, filename=None):
        super().__init__()
        self.current_state = self.initial_state
        self.filename = filename

        if self.filename:
            try:
                self.deserialize_statemachine()
            except FileNotFoundError:
                pass

    def serialize_statemachine(self):
        if self.filename:

            pickle_dict = {
                "version": 1,
                "sm_state": self.current_state.id,
                "external_state": self.get_supplementary_state()
            }

            with open(self.filename, 'wb') as file:
                pickle.dump(pickle_dict, file)
        else:
            print("No filename. Can't serialize.")

    def deserialize_statemachine(self):
        if self.filename:
            supplementary_state = None
            supp_succeeded = None
            state = None

            with open(self.filename, 'rb') as file:
                try:

                    pickle_dict = pickle.load(file)

                    for t in self.states:
                        if t.id == pickle_dict["sm_state"]:
                            state = t
                            break

                    supplementary_state = pickle_dict["external_state"]

                except Exception as e:
                    print("Failed to load serialized checkpoint:", e)

            supp_succeeded = self.set_supplementary_state(supplementary_state)

            if supp_succeeded is None:
                raise Exception("StateMachine set_supplementary_state must return a bool to indicate loading status.")

            try:
                if state is not None and supp_succeeded is True:
                    print("Found saved state... Loading from step: ", t.name)
                    self.current_state = state
            except Exception as e:
                print("Failed to load current state from checkpoint ", e)

        else:
            print("No filename. Can't deserialize.")

    def get_supplementary_state(self) -> dict:
        dict = {"placeholder": 1}
        return dict

    def set_supplementary_state(self, dictionary: dict) -> bool:
        return True
