"""
Temporary queue for CnR 1.5 to store Observations

Should not be used in CnR 2.0
    -Belief updates should occur with each new observation
        not, as in CnR 1.5, with belief updates occuring
        only with new goal_pose in the policy_translator_server

Data structure is a "list (self.Obs) of 2 item lists[int, bool]" inside of an instance of Obs_Queue
"""

__author__ = ["Luke Barbier", "Ian Loefgren"]
__copyright__ = "Copyright 2017, Cohrint"
__credits__ = ["Luke Barbier", "Ian Loefgren"]
__license__ = "GPL"
__version__ = "1.1"
__maintainer__ = "Luke Barbier"
__email__ = "luba6098@colorado.edu"
__status__ = "Development"


class Obs_Queue:
    def __init__(self):
        self.Obs = []


    def flush(self):
        """
        return all of the observations
        Clears the queue
        Returns
        ----------
        list of 2 item lists ,
        ie: [[int,bool], [int, bool]]
        """
        f = list(self.Obs) # copy the list
        self._delete()
        return f


    def add(self, text, obs_room_num, obs_model, obs_id=0, obs_pos_neg=False):
        """
        -Adds a single item to the queue
        Parameters
        ----------
        obs_id : integer , corresponds to the index of the numpy likelihood
        obs_pos_neg : Boolean, corresponds to the likelihood being true or untrue
        """
        obs = [obs_room_num, obs_model, obs_id, obs_pos_neg, text]
        self.Obs.append(obs)

    def _delete(self):
        del self.Obs[:]

    def print_queue(self):
        """ Prints the contents of the queue to stdin"""
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
