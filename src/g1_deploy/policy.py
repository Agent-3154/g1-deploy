import onnxruntime as ort
import numpy as np
from pathlib import Path


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


class FSM:
    def __init__(self, policies: dict[str, ONNXModule], start_policy_name: str):
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