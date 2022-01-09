# state == num_slots, num_disks, support, toggle
# support == (...(object, support),...)
# toggle == (...(toggle, is_on),...)

def rehash(num_slots, num_disks, support, toggle):
    return (num_slots, num_disks) + tuple(
        tuple((key, d[key]) for key in sorted(d.keys()))
        for d in (support, toggle))
    
def unhash(state):
    num_slots, num_disks, support, toggle = state
    return num_slots, num_disks, dict(support), dict(toggle)

def initial_state(num_slots, num_disks):
    # assumes max(num_slots, num_disks) <= 10
    support = tuple((f"disk{i}", f"slot{i}") for i in range(num_slots))
    support += tuple((f"disk{i}", "dock") for i in range(num_slots, num_disks))
    toggle = tuple((f"toggle{i}", 0) for i in range(num_slots))
    return num_slots, num_disks, support, toggle

def valid_actions(state):
    num_slots, num_disks, support, toggle = unhash(state)
    actions = ()
    
    if "gripper" in support.values():
        # can't pick up or press
        # can put down
        gripped = [k for k in support if support[k] == "gripper"][0]
        free_slots = [f"slot{i}" for i in range(num_slots) if f"slot{i}" not in support.values() and toggle[f"toggle{i}"] % 2 == 1]
        actions += tuple(("put-down", (gripped, dest)) for dest in free_slots + ["dock","bin"])
    else:
        # can't put down
        # can press
        actions += tuple(("press", (f"toggle{i}",)) for i in range(num_slots))
        # can pick up
        for i in range(num_disks):
            d = f"disk{i}"
            if support[d] == "bin": continue
            if support[d][:4] == "slot":
                j = int(support[d][4])
                if toggle[f"toggle{j}"] % 2 == 0: continue
            actions += (("pick-up", (f"disk{i}",)),)

    return actions

def perform(action, state):
    # assumes action is valid in state

    name, args = action
    num_slots, num_disks, support, toggle = unhash(state)

    if name == "pick-up":
        obj, = args
        support[obj] = "gripper"

    elif name == "put-down":
        obj, dest = args
        support[obj] = dest

    elif name == "press":
        tog, = args
        toggle[tog] = (toggle[tog] + 1) % 4

    return rehash(num_slots, num_disks, support, toggle)        

if __name__ == "__main__":

    num_slots, num_disks = 2, 3 

    s0 = initial_state(num_slots, num_disks)
    print(s0)
    assert s0 == rehash(*unhash(s0))

    A0 = valid_actions(s0)
    for a in A0: print(a)

    plan = (
        ("pick-up", ("disk0",)),
        ("put-down", ("disk0","bin")),
        ("press", ("toggle1",)),
    )

    s = s0
    for action in plan:
        s = perform(action, s)
        print(s)
        A = valid_actions(s)
        for a in A: print(a)
