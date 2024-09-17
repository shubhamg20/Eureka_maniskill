class PushCubeObs(Env):
    def update_states(self):
        self.obj_pose = self.obj.pose.p  # batched positions of shape (N, 3) with batch size of N
        self.goal_pose = self.goal_region.pose.p # batched positions of shape (N, 3) with batch size of N
        self.agent_pose = self.agent.tcp.pose.p # batched positions of shape (N, 3) with batch size of N
        self.cube_half_size = self.cube_half_size #float