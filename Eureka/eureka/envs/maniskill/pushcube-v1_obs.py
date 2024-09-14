class PushCube:
    def compute_observations(self):
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)

        self.object_pose = self.root_state_tensor[self.object_indices, 0:7]
        self.object_pos = self.root_state_tensor[self.object_indices, 0:3]
        self.object_rot = self.root_state_tensor[self.object_indices, 3:7]
        self.object_linvel = self.root_state_tensor[self.object_indices, 7:10]
        self.object_angvel = self.root_state_tensor[self.object_indices, 10:13]

        self.goal_pose = self.goal_states[:, 0:7]
        self.goal_pos = self.goal_states[:, 0:3]
        self.goal_rot = self.goal_states[:, 3:7]

        self.hand_pos = self.rigid_body_states[:, self.hand_handle][:, 0:3]
        self.hand_rot = self.rigid_body_states[:, self.hand_handle][:, 3:7]
        self.hand_vel = self.rigid_body_states[:, self.hand_handle][:, 7:13]

        self.dof_pos = self.dof_state.view(self.num_envs, -1, 2)[:, :, 0]
        self.dof_vel = self.dof_state.view(self.num_envs, -1, 2)[:, :, 1]