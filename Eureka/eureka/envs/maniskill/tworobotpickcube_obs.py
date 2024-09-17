class TwoRobotPickCube(Env):
    def update_states(self):
        self.obj_pose = self.obj.pose.p  # batched positions of shape (N, 3) with batch size of N
        self.right_agent = self.right_agent.tcp.pose.p # batched positions of shape (N, 3) with batch size of N
        self.left_agent = self.left_agent.tcp.pose.p # batched positions of shape (N, 3) with batch size of N
        self.right_agent_finger1_link = self.right_agent.finger1_link.pose.p # batched positions of shape (N, 3) with batch size of N
        self.right_agent_finger2_link = self.right_agent.finger2_link.pose.p # batched positions of shape (N, 3) with batch size of N
        self.cube = self.cube.pose.p # batched positions of shape (N, 3) with batch size of N
        self.right_agent_is_grasping = self.right_agent.is_grasping #bool
        self.goal_site = self.goal_site.pose.p # batched positions of shape (N, 3) with batch size of N
        self.left_agent_robot_qpos = self.left_agent.robot.get_qpos() # batched positions of shape (N, 3) with batch size of N
        self.left_init_qpos = self.left_init_qpos # batched positions of shape (N, 3) with batch size of N
        self.right_agent_robot_qvel = self.right_agent.robot.get_qvel() # batched positions of shape (N, 3) with batch size of N
        self.left_agent_robot_qvel = self.left_agent.robot.get_qvel() # batched positions of shape (N, 3) with batch size of N