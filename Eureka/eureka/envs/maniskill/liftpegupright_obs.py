class LiftPegUpright(Env):
    def update_states(self):
        self.peg_pose = self.peg.pose.p  # batched positions of shape (N, 3) with batch size of N
        self.goal_pose = self.goal_region.pose.p # batched positions of shape (N, 3) with batch size of N
        self.agent_pose = self.agent.tcp.pose.p # batched positions of shape (N, 3) with batch size of N
        self.peg_half_length = self.peg_half_length #float