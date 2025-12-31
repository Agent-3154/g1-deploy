import torch
import onnxruntime as ort
import numpy as np
import yaml
from pathlib import Path
from g1_deploy.observation import Observation, Articulation

class ONNXModule:
    def __init__(self, onnx_path: Path | str):
        self.onnx_path = onnx_path
        self.session = ort.InferenceSession(onnx_path)
        self.input_names = [input.name for input in self.session.get_inputs()]
        self.output_names = [output.name for output in self.session.get_outputs()]
        self.input_shapes = [input.shape for input in self.session.get_inputs()]
        self.output_shapes = [output.shape for output in self.session.get_outputs()]
        self.input_types = [input.type for input in self.session.get_inputs()]
        self.output_types = [output.type for output in self.session.get_outputs()]

    def __repr__(self) -> str:
        """Return a string representation of the ONNXModule."""
        lines = [
            f"ONNXModule(onnx_path={self.onnx_path!r}",
            f"  inputs: {len(self.input_names)}",
        ]
        for name, shape, dtype in zip(self.input_names, self.input_shapes, self.input_types):
            lines.append(f"    {name}: shape={shape}, dtype={dtype}")
        lines.append(f"  outputs: {len(self.output_names)}")
        for name, shape, dtype in zip(self.output_names, self.output_shapes, self.output_types):
            lines.append(f"    {name}: shape={shape}, dtype={dtype}")
        lines.append(")")
        return "\n".join(lines)
    
    def dummy_input(self) -> dict[str, np.ndarray]:
        return {input.name: np.zeros(input.shape, dtype=np.float32) for input in self.session.get_inputs()}
    
    def forward(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        # inputs = {name: value[None, ...] for name, value in inputs.items()}
        outputs = self.session.run(self.output_names, inputs)
        return {name: output for name, output in zip(self.output_names, outputs)}

class TorchJitModule:
    def __init__(self, torchscript_path: Path | str):
        self.torchscript_path = torchscript_path
        self.module = torch.jit.load(torchscript_path)
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        outputs = self.module(input)
        return outputs

class Policy:
    def __init__(
        self, 
        name,
        robot: Articulation,
        config_path: Path | str,
        model_path: Path | str, 
        output_key: str = "action"
    ):
        self.name = name
        self.robot = robot

        with open(config_path, "r") as f:
            self.obs_config = yaml.load(f, Loader=yaml.FullLoader)["observation"]

        self.observation_groups = {}
        for group_name, group_config in self.obs_config.items():
            self.observation_groups[group_name] = []
            for observation_name, observation_config in group_config.items():
                observation_class = Observation.registry[observation_name]
                if observation_config is None:
                    observation = observation_class(robot)
                else:
                    observation = observation_class(robot, **observation_config)
                self.observation_groups[group_name].append(observation)

        if str(model_path).endswith(".pt"):
            self.module = TorchJitModule(model_path)
            self.module_type = "torch"
        elif str(model_path).endswith(".onnx"):
            self.module = ONNXModule(model_path)
            self.module_type = "onnxruntime"
            self.output_key = output_key
        else:
            raise ValueError(f"Unsupported model type: {model_path}")

    def compute_observations(self):
        results = {}
        for group_name, group_observations in self.observation_groups.items():
            group_results = []
            for observation in group_observations:
                group_results.append(observation())
            results[group_name] = np.concatenate(group_results, axis=-1, dtype=np.float32)[None, ...]
        return results

    def enter(self):
        pass

    def run(self) -> np.ndarray:
        observations = self.compute_observations()
        if self.module_type == "torch":
            assert len(observations) == 1, f"Torch module expects exactly one observation key, but got {list(observations.keys())}"
            observation = next(iter(observations.values()))
            with torch.no_grad():
                output = self.module.forward(torch.from_numpy(observation))
            return output.detach().cpu().numpy()
        elif self.module_type == "onnxruntime":
            inputs = self.module.dummy_input()
            inputs.update(observations)
            outputs = self.module.forward(inputs)
            return outputs[self.output_key]
        else:
            raise ValueError(f"Unsupported module type: {self.module_type}")

    def exit(self):
        pass

    def checkchange(self) -> str | None:
        return None

class SkillA(Policy):
    def __init__(self, name, robot, config_path, model_path, output_key="action"):
        super().__init__(name, robot, config_path, model_path, output_key)
        self.count = 0

    def enter(self):
        print("Entering SkillA")
        self.count = 0

    def run(self) -> np.ndarray:
        self.count += 1
        if self.count % 50 == 0:
            print(f"SkillA running, step {self.count}")
        return super().run()

    def checkchange(self):
        # Example: auto-switch to SkillB after 500 steps
        if self.count > 500:
            return "sB"
        return None

class SkillB(Policy):
    def __init__(self, name, robot, config_path, model_path, output_key="action"):
        super().__init__(name, robot, config_path, model_path, output_key)
        self.count = 0

    def enter(self):
        print("Entering SkillB")
        self.count = 0

    def run(self) -> np.ndarray:
        self.count += 1
        if self.count % 50 == 0:
            print(f"SkillB running, step {self.count}")
        return super().run()

    def checkchange(self):
        return None

class TrackMode(Policy):
    def __init__(self, name, robot, config_path, model_path, output_key="linear_4"):
        super().__init__(name, robot, config_path, model_path, output_key)

    def enter(self):
        print("Entering TrackMode")
    
    def run(self) -> np.ndarray:
        return super().run()

    def checkchange(self) -> str | None:
        if self.robot.t >= self.robot.ref_motion.motion_length:
            # return "loco"
            return "sA"
        return None

class FSM:
    def __init__(self, policies: dict[str, Policy], start_policy_name: str):
        self.policies = policies
        self.current_policy_name = start_policy_name
        self.current_policy = self.policies[self.current_policy_name]
        self.next_policy_name = None
        self.current_policy.enter()

    @property
    def module(self):
        return self.current_policy.module

    def set_next_policy(self, policy_name: str):
        if policy_name in self.policies:
            self.next_policy_name = policy_name
        else:
            print(f"FSM: Policy {policy_name} not found")

    def run(self) -> np.ndarray:
        # Priority: set_next_policy (external) > current_policy.checkchange (internal)
        target_policy_name = self.next_policy_name or self.current_policy.checkchange()

        if target_policy_name is not None and target_policy_name in self.policies and target_policy_name != self.current_policy_name:
            print(f"FSM: Transitioning from {self.current_policy_name} to {target_policy_name}")
            self.current_policy.exit()
            self.current_policy_name = target_policy_name
            self.current_policy = self.policies[self.current_policy_name]
            self.current_policy.enter()
            self.next_policy_name = None # Reset after transition

        return self.current_policy.run()