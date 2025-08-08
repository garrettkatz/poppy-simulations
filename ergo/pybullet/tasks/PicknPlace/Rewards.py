def rewards_potential(env, obj_pos, obj_id, use_right_hand, v_f, old_distance=None, scale_factor=1.0):
    max_dist = 1.0
    g_check = False
    gripper_link_indices = []

    if use_right_hand:
        tip1 = tr.tensor(pb.getLinkState(env.robot_id, env.joint_index["r_moving_tip"])[0])
        tip2 = tr.tensor(pb.getLinkState(env.robot_id, env.joint_index["r_fixed_tip"])[0])
        gripper_link_indices = [env.joint_index["r_moving_tip"], env.joint_index["r_fixed_tip"]]
    else:
        tip1 = tr.tensor(pb.getLinkState(env.robot_id, env.joint_index["l_moving_tip"])[0])
        tip2 = tr.tensor(pb.getLinkState(env.robot_id, env.joint_index["l_fixed_tip"])[0])
        gripper_link_indices = [env.joint_index["l_moving_tip"], env.joint_index["l_fixed_tip"]]

    tip_contacts = pb.getContactPoints(bodyA=env.robot_id, bodyB=env.robot_id,
                                       linkIndexA=gripper_link_indices[0], linkIndexB=gripper_link_indices[1])

    tip_distance = torch.norm(tip1 - tip2)
    distance_diff = torch.abs(tip_distance - 0.015)
    gripper_midpoint = (tip1 + tip2) / 2.0

    # Compute distance to nearest voxel
    v_f_np = np.array(v_f)
    v_f_tensor = torch.tensor(v_f_np, dtype=torch.float32)  # shape (n_voxels, 3)
    distances = torch.norm(v_f_tensor - gripper_midpoint, dim=1)
    new_distance = torch.min(distances)

    # → Compute approach reward as potential difference
    if old_distance is None:
        approach_reward = torch.tensor(0.0)
    else:
        diff = (old_distance - new_distance)*scale_factor #0.045
        diff_clamped = torch.clamp(diff, -1.0, 1.0)
        approach_reward = diff_clamped
        if new_distance <= 0.045:
            approach_reward+= 2*f(e**-new_distance) # 1.01 if new is 0.01 r =2 [0.97 , 1.03]

    voxel_distance_bonus = max(-1.0, 0.1 * (1 - distance_diff / 0.030))
   # if new_distance <=0.03:
       # pickup_reward + = 0.5
    obj_height = pb.getBasePositionAndOrientation(obj_id)[0][2]
    pickup_reward = (10) if obj_height > obj_pos[2] + 0.2 else (obj_height - obj_pos[2])*10
    pickup_reward = torch.tensor(pickup_reward)
    pickup_reward = torch.clamp(pickup_reward, min=0.0)
    pickup_reward += voxel_distance_bonus
    #pickup_reward = torch.clamp(pickup_reward, min=0.0)

    if len(tip_contacts) > 0:
        pickup_reward -= 0.1
        #print("penalty")

    for link_idx in gripper_link_indices:
        contact_points = pb.getContactPoints(env.robot_id, obj_id, link_idx, -1)
        if len(contact_points) == 1:
            pickup_reward += 1
            print("touch")
        if len(contact_points) > 1:
            pickup_reward += 2
            print("touch")
    pickup_reward+=(obj_height - obj_pos[2])*50
    if obj_height > obj_pos[2] + 0.2:
        g_check = True
        pickup_reward+=10

        print("Pick success")

    total_reward = approach_reward + pickup_reward
    #total_reward = torch.clamp(total_reward, min=0.0)

    return approach_reward, pickup_reward, g_check, new_distance, total_reward














    def update(self, states, actions, old_log_probs, rewards, approach_rewards, pickup_rewards,distance, values, next_values, dones):
        advantages, returns = self.compute_advantage(rewards, values, next_values, dones)

        # FIX: Detach to avoid backward-through-graph error
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-4)
        advantages = advantages.detach()
        returns = returns.detach()
        old_log_probs = old_log_probs.detach()
        states = states.detach()
        actions = actions.detach()
        for _ in range(self.epochs):
            mean, std, new_values = self.model(states)
            dist = tr.distributions.Normal(mean, std)

            # Unsquash actions using atanh (inverse tanh)
            #eps = 1e-6
            #raw_actions = 0.5 * torch.log((1 + actions) / (1 - actions + 1e-8))
            #safe_actions = torch.clamp(actions, -0.99, 0.99)
            squashed_actions = torch.tanh(actions)
            safe_actions = torch.clamp(squashed_actions, -0.99, 0.99)

            new_log_probs = dist.log_prob(actions).sum(dim=-1)
            new_log_probs -= torch.log(1 - safe_actions.pow(2) + 1e-6).sum(dim=-1)

            ratio = torch.exp(new_log_probs - old_log_probs)

            surrogate1 = ratio * advantages
            surrogate2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages
            policy_loss = -torch.min(surrogate1, surrogate2).mean()
            assert len(self.model.approach_joints_idx) > 0, "Approach joints not set"



            approach_mask  = ((normal_approach)>0.7).float()
            pick_mask = (distance<0.1).float() #(50,0)

            joint_entropies = dist.entropy() # (n_steps,action_space)
           # normalized_approach = torch.tensor(approach_rewards).sum(dim=-1).mean()
            normalized_approach = torch.clamp(torch.tensor(self.last_approach_reward_avg)/50, -1.0, 1.0) #shape 50,
            normalized_approach_pos = (normalized_approach + 1.0) / 2.0
            normalized_pickup = torch.clamp(torch.tensor(self.last_pickup_reward_avg) / 10.0, 0.0, 1.0)
            inv_approach = 1.0 - normalized_approach_pos
            inv_pickup = 1.0 - normalized_pickup
            min_beta = 0.001
            max_beta = 0.01


            if distance<0.1:
                current_pickup_entropy_coeff = min_beta + (max_beta - min_beta) * inv_pickup
            else:
                current_pickup_entropy_coeff = max_beta

            entropy_approach = joint_entropies[:, self.model.approach_joints_idx].sum(dim=-1)
            entropy_pickup = joint_entropies[:, self.model.pickup_joints_idx].sum(dim=-1)

            total_entropy_bonus = (current_approach_entropy_coeff * entropy_approach + current_pickup_entropy_coeff * entropy_pickup).mean()

            policy_loss = policy_loss - total_entropy_bonus

            value_loss = nn.functional.mse_loss(new_values.squeeze(), returns)
            loss = policy_loss + 0.5 * value_loss

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
            self.optimizer.step()



contacts = pb.getClosestPoints(bodyA=gripper_id, bodyB=obj_id, distance=0.02)