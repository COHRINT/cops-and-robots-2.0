"""
Temporary queue for CnR 1.5 to store Observations

Should not be used in CnR 2.0
    -Belief updates should occur with each new observation
        not, as in CnR 1.5, with belief updates occuring
        only with new goal_pose in the policy_translator_server
"""


class Obs_Queue:
    def __init__(self):
        self.Obs = []

    # return all of the observations
    def flush(self):
        f = list(self.Obs) # copy the list
        self._delete()
        return f

    # adds a single item to the queue
    def add(self, obs_id=0, obs_pos_neg=False):
        obs = [obs_id, obs_pos_neg]
        self.Obs.append(obs)

    def _delete(self):
        del self.Obs[:]

    def print_queue(self):
        for i in self.Obs:
            print(i)


if __name__ == "__main__":
    o = Obs_Queue()

    ints = [1,2,3,4,5,6,7]
    neg = False
    for i in ints:
        o.add(i, neg)

    print(o.flush())
    print(o.Obs)
