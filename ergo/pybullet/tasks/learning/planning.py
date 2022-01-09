import random
import numpy as np
from time import perf_counter

class Planner:

    def __init__(self, domain, wm, ltm):
        self.domain = domain
        self.wm = wm
        self.ltm = ltm

    # pure recall
    def pure_recall(self, initial_state, goal_state, search_depth=0, max_rollouts=0, max_actions=0):
        state = initial_state
        plan = []

        start_time = perf_counter()
        while len(plan) <= max_actions:
            if state not in self.ltm: break
            action, _ = self.ltm[state]
            state = self.domain.perform(action, state)
            plan.append(action)

        total_time = perf_counter() - start_time
        stats = 0, 0, total_time

        return plan, stats

    # random search to next state
    def landmark_search(self, initial_state, goal_state, search_depth=0, max_rollouts=0, max_actions=0):

        recalled_states = tuple(self.ltm.keys())
        recalled_indexs = tuple(self.ltm[state][1] for state in recalled_states)
        landmarks = tuple(recalled_states[i] for i in np.argsort(recalled_indexs))
        landmarks += (goal_state,)

        state = initial_state
        plan = []
        total_rollouts = 0
        total_nodes = 0

        start_time = perf_counter()
        for landmark in landmarks:

            best_subplan, best_substate = None, None
            for r in range(max_rollouts):
                subplan, substate = [], state
                for i in range(min(search_depth, max_actions - len(plan))):
                    if substate == landmark: break
                    if best_subplan is not None and len(subplan) > len(best_subplan): break
                    action = random.choice(self.domain.valid_actions(substate))
                    substate = self.domain.perform(action, substate)
                    subplan.append(action)
                    total_nodes += 1

                if best_subplan is None:
                    best_subplan, best_substate = subplan, substate
                if substate == landmark and len(subplan) < len(best_subplan):
                    best_subplan, best_substate = subplan, substate

                total_rollouts += 1
                if len(best_subplan) == 1: break

            plan += best_subplan
            state = best_substate

            if state != landmark: break
            if state == goal_state: break

            plan += (self.ltm[state][0],)
            state = self.domain.perform(self.ltm[state][0], state)
            if len(plan) > max_actions: break

        total_time = perf_counter() - start_time
        stats = total_rollouts, total_nodes, total_time

        return plan, stats
        

    # # random search
    # def solve(self, initial_state, goal_state, search_depth=0, max_rollouts=0, max_actions=0):

    #     self.wm.clear()

    #     def rollout(state, depth=0):
    #         # if state in self.ltm or state == goal_state or depth == search_depth:
    #         if (state not in self.wm and state in self.ltm) or depth == search_depth:
    #             return (), ()
    #         action = random.choice(self.domain.valid_actions(state))
    #         state = self.domain.perform(action, state)
    #         subplan, substates = rollout(state, depth+1)
    #         return (action,) + subplan, (state,) + substates

    #     plan = []
    #     total_rollouts = 0
    #     total_nodes = 0

    #     start_time = perf_counter()
    #     state = initial_state
    #     while len(plan) < max_actions:
    #         if state == goal_state: break

    #         if state in self.ltm:
    #             action = self.ltm[state]
    #             plan.append(action)
    #             state = self.domain.perform(action, state)
    #             self.wm.store(state)

    #         else:
    #             # stop at first found
    #             for r in range(max_rollouts):
    #                 subplan, substates = rollout(state)
    #                 total_rollouts += 1
    #                 total_nodes += len(subplan)
    #                 # if substates[-1] in self.ltm or substates[-1] == goal_state: break
    #                 if substates[-1] in self.ltm: break

    #             # # find min plan
    #             # subplan, last_state = None, None
    #             # for r in range(max_rollouts):
    #             #     subplan_r, last_state_r = rollout(state)
    #             #     total_rollouts += 1
    #             #     total_nodes += len(subplan_r)
    #             #     # if last_state in self.ltm or last_state == goal_state: break
    #             #     if last_state in self.ltm:
    #             #         if subplan is None or len(subplan_r) < len(subplan):
    #             #             subplan, last_state = subplan_r, last_state_r

    #             # if subplan is None or subplan == (): break

    #             if subplan == (): break
    #             plan += subplan
    #             state = substates[-1]
    #             for substate in substates: self.wm.store(substate)

        total_time = perf_counter() - start_time

        stats = total_rollouts, total_nodes, total_time
        return plan, stats

