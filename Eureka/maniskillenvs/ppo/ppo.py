class PPO:
    def __init__(self, agent, optimizer, num_envs, num_steps, batch_size, minibatch_size,
                 epochs, learning_rate, gamma, gae_lambda, clip_coef, ent_coef, vf_coef,
                 max_grad_norm, target_kl, norm_adv, clip_vloss, reward_scale, device):
        self.agent = agent
        self.optimizer = optimizer
        self.num_envs = num_envs
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.minibatch_size = minibatch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_coef = clip_coef
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.target_kl = target_kl
        self.norm_adv = norm_adv
        self.clip_vloss = clip_vloss
        self.reward_scale = reward_scale
        self.device = device

    def compute_gae(self, rewards, values, dones, next_value):
        advantages = torch.zeros_like(rewards).to(self.device)
        lastgaelam = 0
        for t in reversed(range(self.num_steps)):
            if t == self.num_steps - 1:
                nextnonterminal = 1.0 - dones[-1]
                nextvalues = next_value
            else:
                nextnonterminal = 1.0 - dones[t + 1]
                nextvalues = values[t + 1]
            delta = rewards[t] + self.gamma * nextvalues * nextnonterminal - values[t]
            advantages[t] = lastgaelam = delta + self.gamma * self.gae_lambda * nextnonterminal * lastgaelam
        returns = advantages + values
        return advantages, returns

    def update(self, obs, actions, logprobs, rewards, dones, values):
        # Compute GAE
        with torch.no_grad():
            next_value = self.agent.get_value(obs[-1]).reshape(1, -1)
            advantages, returns = self.compute_gae(rewards, values, dones, next_value)

        # Flatten the batch
        b_obs = obs.reshape((-1,))
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + self.agent.action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimize policy for K epochs
        clipfracs = []
        for epoch in range(self.epochs):
            # Shuffle the batch
            permutation = torch.randperm(self.batch_size)
            
            for start in range(0, self.batch_size, self.minibatch_size):
                end = start + self.minibatch_size
                idx = permutation[start:end]

                mb_advantages = b_advantages[idx]
                if self.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Forward pass
                _, newlogprob, entropy, newvalue = self.agent.get_action_and_value(b_obs[idx], b_actions[idx])
                
                # Compute ratio (pi_theta / pi_theta_old)
                logratio = newlogprob - b_logprobs[idx]
                ratio = logratio.exp()

                # Compute loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                
                # Compute value loss
                newvalue = newvalue.view(-1)
                if self.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[idx]) ** 2
                    v_clipped = b_values[idx] + torch.clamp(
                        newvalue - b_values[idx],
                        -self.clip_coef,
                        self.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[idx]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[idx]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - self.ent_coef * entropy_loss + v_loss * self.vf_coef

                # Gradient step
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)
                self.optimizer.step()

                clipfracs += [((ratio - 1.0).abs() > self.clip_coef).float().mean().item()]

            if self.target_kl is not None:
                approx_kl = ((ratio - 1) - logratio).mean()
                if approx_kl > self.target_kl:
                    break

        return clipfracs