from demonstrations import demos
from memory import WorkingMemory, LongTermMemory
from planning import Planner
from levenshtein import lvd
import domain

def run_experiment(demo, capacity, consolidation_rate, max_rollouts, num_trials):

    # get intermediate demo states
    s = domain.initial_state(num_slots=4, num_disks=8)
    states = [s]
    for a in demo:
        print(a)
        assert a in domain.valid_actions(s)
        s = domain.perform(a, s)
        states.append(s)

    # initialize memory
    wm = WorkingMemory(capacity)
    ltm = LongTermMemory(consolidation_rate)

    # run trials
    trial_results = []
    for t in range(num_trials):
        print(f"trial {t}...")

        # observe demo and store in working memory
        wm.clear()
        for i,(s,a) in enumerate(zip(states[:-1], demo)):
            if s not in ltm: wm.store((s,a,i))

        # consolidate into long term memory
        for (s,a,i) in wm.contents: ltm[s] = (a, i)

        # try planning
        # plan, stats = Planner(domain, wm, ltm).pure_recall(
        plan, stats = Planner(domain, wm, ltm).landmark_search(
            initial_state = states[0],
            goal_state = states[-1],
            max_rollouts = max_rollouts,
            max_actions=2*len(demo))
    
        # assess performance and "mental" effort
        result = lvd(plan, demo)
        dist, _, _, _, _, _, _ = result
        total_rollouts, total_nodes, total_time = stats

        print(dist, total_rollouts, total_nodes, total_time)
        trial_results.append((dist, total_rollouts, total_nodes, total_time))

    return trial_results

if __name__ == "__main__":

    import pickle as pk

    demo = demos["unstructured"]["p"]
    capacity = 7
    consolidation_rate = .2
    max_rollouts = 100
    trials_per_block = 1
    num_blocks = 25
    num_repetitions = 30

    num_trials = num_blocks * trials_per_block

    runexp = True
    showresults = True

    if runexp:

        all_results = []
        for rep in range(num_repetitions):
            results = run_experiment(demo, capacity, consolidation_rate, max_rollouts, num_trials)
            all_results.append(results)
    
        with open("trial_results.pkl","wb") as f: pk.dump(all_results, f)

    if showresults:

        import matplotlib as ml
        import matplotlib.pyplot as pt
        import numpy as np

        ml.rcParams.update({'font.size': 20, 'font.family': 'serif'})

        with open("trial_results.pkl","rb") as f: all_results = pk.load(f)
        all_dists, all_nodes, all_times = [], [], []
        blocks = np.arange(num_blocks) + 1

        pt.figure(figsize=(15,5))
        numplots = 3

        for results in all_results[:]:
            dists, rollouts, nodes, times = zip(*results)

            dists = np.array(dists).reshape(-1,trials_per_block).mean(axis=1)
            times = np.array(times).reshape(-1,trials_per_block).mean(axis=1)
            nodes = np.array(nodes).reshape(-1,trials_per_block).mean(axis=1)

            all_dists.append(dists)
            all_times.append(times)
            all_nodes.append(nodes)

            pt.subplot(1,numplots,1)
            pt.plot(blocks, dists, '-', color=(.75,.75,.75))
            pt.subplot(1,numplots,2)
            pt.plot(blocks, times, '-', color=(.75,.75,.75))
            pt.subplot(1,numplots,3)
            pt.plot(dists, times, '-', color=(.75,.75,.75))
            # pt.subplot(1,numplots,4)
            # pt.plot(blocks, nodes, '-', color=(.75,.75,.75))
            # pt.subplot(1,numplots,5)
            # pt.plot(dists, nodes, '-', color=(.75,.75,.75))

        all_dists = np.array(all_dists).mean(axis=0)
        all_times = np.array(all_times).mean(axis=0)
        all_nodes = np.array(all_nodes).mean(axis=0)

        pt.subplot(1,numplots,1)
        pt.plot(blocks, all_dists, 'ko-')
        # pt.xticks(blocks)
        pt.xlabel("Trial")
        pt.ylabel("LVD")

        pt.subplot(1,numplots,2)
        pt.plot(blocks, all_times, 'ko-')
        # pt.xticks(blocks)
        pt.xlabel("Trial")
        pt.ylabel("Runtime (s)")

        pt.subplot(1,numplots,3)
        pt.plot((all_dists[0], all_dists[-1]), (all_times[0], all_times[-1]), 'k--')
        pt.plot(all_dists, all_times, 'ko-')
        for b in [1] + list(range(5,len(blocks),5)):
            pt.text(all_dists[b] - 1, all_times[b] + 0.0025, str(b))
        pt.xlabel("LVD")
        pt.ylabel("Runtime (s)")

        # pt.subplot(1,numplots,4)
        # pt.plot(blocks, all_nodes, 'ko-')
        # pt.xticks(blocks)
        # pt.xlabel("Block")
        # pt.ylabel("Node count")

        # pt.subplot(1,numplots,5)
        # pt.plot(all_dists, all_nodes, 'ko-')
        # for b in range(len(blocks)):
        #     pt.text(all_dists[b]-1, all_nodes[b] + 250, str(b+1))
        # pt.xlabel("LVD")
        # pt.ylabel("Node count")

        pt.tight_layout()
        pt.savefig("learning_dynamics.pdf")
        pt.show()
