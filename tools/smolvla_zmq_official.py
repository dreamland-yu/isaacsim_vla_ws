#!/usr/bin/env python3
"""Bridge isaacsim_vla_ws observations to a VLA-style action socket.

This is intentionally usable before LeRobot/SmolVLA is installed.  The default
``fake`` policy receives observations from ``vla_center`` on port 5556 and
publishes a fixed SO100 joint target on port 5555.  Once the ZMQ loop is proven,
the policy implementation can be replaced with official LeRobot SmolVLA
inference without touching the Isaac Sim or ROS2 side.
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
import time
from dataclasses import dataclass

import numpy as np
import zmq


SO100_JOINT_NAMES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)

HOME_ACTION = np.array([0.0009, 0.0258, 0.0, 0.0, 0.0, -0.0065], dtype=np.float32)
TEST_ACTION = np.array([0.0037, 1.7, -1.5, 0.9, -1.57, 0.0], dtype=np.float32)
FORWARD_REACH_ACTION = np.array([0.0, 1.25, -1.15, 0.75, 0.0, -0.0065], dtype=np.float32)
CUBE_HOVER_ACTION = np.array([0.0, 0.95, -0.85, 0.55, 0.0, -0.0065], dtype=np.float32)
CUBE_LEFT_HOVER_ACTION = np.array([-0.35, 1.10, -1.35, 0.45, 0.0, -0.0065], dtype=np.float32)
GRIPPER_CLOSED = 0.75
ISAAC_JOINT_LIMIT_MIN = np.array([-2.0, 0.0, -3.142, -2.5, -3.142, -0.2], dtype=np.float32)
ISAAC_JOINT_LIMIT_MAX = np.array([2.0, 3.5, 0.0, 1.2, 3.142, 2.0], dtype=np.float32)


def parse_float_vector(text: str, expected_len: int, name: str) -> np.ndarray:
    values = np.array([float(item.strip()) for item in text.split(",")], dtype=np.float32)
    if values.shape != (expected_len,):
        raise ValueError(f"Expected {name} to contain {expected_len} comma-separated values, got {text!r}")
    return values


@dataclass
class Observation:
    timestamp: float
    joints: np.ndarray
    img1_jpeg: bytes
    img2_jpeg: bytes
    img3_jpeg: bytes | None = None


def unpack_observation(payload: bytes) -> Observation:
    """Decode the npz payload produced by vla_center/send_observation_2cam.py."""
    with np.load(io.BytesIO(payload)) as data:
        timestamp = float(data["ts"][0])
        joints = data["joints"].astype(np.float32)
        img1_jpeg = data["img1"].astype(np.uint8).tobytes()
        img2_jpeg = data["img2"].astype(np.uint8).tobytes()
        img3_jpeg = data["img3"].astype(np.uint8).tobytes() if "img3" in data else None

    if joints.shape != (len(SO100_JOINT_NAMES),):
        raise ValueError(f"Expected joints shape {(len(SO100_JOINT_NAMES),)}, got {joints.shape}")

    return Observation(
        timestamp=timestamp,
        joints=joints,
        img1_jpeg=img1_jpeg,
        img2_jpeg=img2_jpeg,
        img3_jpeg=img3_jpeg,
    )


def select_fake_action(mode: str, observation: Observation, step: int) -> np.ndarray:
    """Return a simple target action for loop testing."""
    if mode == "home":
        return HOME_ACTION.copy()
    if mode == "test":
        return TEST_ACTION.copy()
    if mode == "mirror":
        return observation.joints.astype(np.float32).copy()
    if mode == "alternate":
        period = 40
        return TEST_ACTION.copy() if (step // period) % 2 == 0 else HOME_ACTION.copy()
    if mode == "reach_forward_grasp_return":
        return scripted_grasp_action(step, FORWARD_REACH_ACTION)
    if mode == "cube_hover_grasp_return":
        return scripted_grasp_action(step, CUBE_HOVER_ACTION)
    if mode == "cube_left_hover_grasp_return":
        return scripted_grasp_action(step, CUBE_LEFT_HOVER_ACTION)
    raise ValueError(f"Unknown fake action mode: {mode}")


def lerp_action(start: np.ndarray, end: np.ndarray, alpha: float) -> np.ndarray:
    alpha = float(np.clip(alpha, 0.0, 1.0))
    return (1.0 - alpha) * start + alpha * end


def scripted_grasp_action(
    step: int,
    reach_action: np.ndarray,
    start_action: np.ndarray | None = None,
) -> np.ndarray:
    """Open-loop reach, close gripper in the air, then return home."""
    start = HOME_ACTION if start_action is None else start_action.astype(np.float32)
    reach_open = reach_action.astype(np.float32).copy()
    reach_closed = reach_open.copy()
    reach_closed[5] = GRIPPER_CLOSED

    if step < 45:
        return lerp_action(start, reach_open, step / 44)
    if step < 75:
        return reach_open.copy()
    if step < 105:
        return lerp_action(reach_open, reach_closed, (step - 75) / 29)
    if step < 135:
        return reach_closed.copy()
    if step < 195:
        return lerp_action(reach_closed, HOME_ACTION, (step - 135) / 59)
    return HOME_ACTION.copy()


def decode_jpeg_rgb(jpeg_bytes: bytes) -> np.ndarray:
    """Decode JPEG bytes to an RGB uint8 image with shape (H, W, 3)."""
    import cv2

    image_bgr = cv2.imdecode(np.frombuffer(jpeg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError("Failed to decode JPEG image")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


class FakePolicy:
    def __init__(self, action_mode: str):
        self.action_mode = action_mode
        self.start_action: np.ndarray | None = None

    def select_action(self, observation: Observation, step: int) -> np.ndarray:
        if self.action_mode in ("reach_forward_grasp_return", "cube_hover_grasp_return"):
            if self.start_action is None:
                self.start_action = observation.joints.astype(np.float32).copy()
            reach_action = (
                FORWARD_REACH_ACTION
                if self.action_mode == "reach_forward_grasp_return"
                else CUBE_HOVER_ACTION
            )
            return scripted_grasp_action(step, reach_action, self.start_action)
        if self.action_mode == "cube_left_hover_grasp_return":
            if self.start_action is None:
                self.start_action = observation.joints.astype(np.float32).copy()
            return scripted_grasp_action(step, CUBE_LEFT_HOVER_ACTION, self.start_action)
        return select_fake_action(self.action_mode, observation, step)


class SmolVLAPolicyRunner:
    def __init__(self, args: argparse.Namespace):
        import torch
        from lerobot.policies import make_pre_post_processors
        from lerobot.policies.smolvla import SmolVLAPolicy

        self.torch = torch
        self.device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.task = args.task
        self.robot_type = args.robot_type
        self.action_clip = args.action_clip
        self.action_map = args.smolvla_action_map
        self.state_map = args.smolvla_state_map
        self.stats_key = args.smolvla_stats_key
        self.replan_interval = args.smolvla_replan_interval
        self.steps_since_replan = 0
        self.gripper_scale = args.smolvla_gripper_scale
        self.sim_joint_signs = parse_float_vector(
            args.smolvla_sim_joint_signs,
            len(SO100_JOINT_NAMES),
            "--smolvla-sim-joint-signs",
        )
        self.state_offsets_deg = parse_float_vector(
            args.smolvla_state_offsets_deg,
            len(SO100_JOINT_NAMES),
            "--smolvla-state-offsets-deg",
        )
        self.action_offsets_deg = parse_float_vector(
            args.smolvla_action_offsets_deg,
            len(SO100_JOINT_NAMES),
            "--smolvla-action-offsets-deg",
        )
        self.camera_order = tuple(int(item.strip()) for item in args.camera_order.split(","))
        if self.camera_order != tuple(item for item in self.camera_order if item in (1, 2, 3)):
            raise ValueError("--camera-order must only contain camera ids 1, 2, or 3")
        if len(self.camera_order) != 3:
            raise ValueError("--camera-order must contain exactly 3 comma-separated camera ids")
        self.last_model_action: np.ndarray | None = None
        self.last_model_state: np.ndarray | None = None

        print(f"Loading {args.model_id} on {self.device} ...")
        self.model = SmolVLAPolicy.from_pretrained(args.model_id)
        print("SmolVLA policy object created.")
        self.model.to(self.device)
        print(f"SmolVLA moved to {self.device}.")
        self.model.eval()
        print("SmolVLA set to eval mode.")

        print("Creating LeRobot pre/post processors ...")
        self.preprocess, self.postprocess = make_pre_post_processors(
            self.model.config,
            args.model_id,
            preprocessor_overrides={"device_processor": {"device": str(self.device)}},
        )
        print("LeRobot pre/post processors created.")
        self.model.reset()
        print("SmolVLA policy state reset.")
        self._action_mean_std = self._find_action_mean_std(self.stats_key)
        print("SmolVLA loaded.")

    def _find_action_mean_std(self, stats_key: str):
        full_key = f"{stats_key}.buffer.action"
        for step in getattr(self.postprocess, "steps", []):
            tensor_stats = getattr(step, "_tensor_stats", {})
            if full_key in tensor_stats:
                stats = tensor_stats[full_key]
                mean = stats["mean"].to(self.device)
                std = stats["std"].to(self.device)
                print(f"Using SmolVLA action stats: {full_key}")
                return mean, std
        print(f"Warning: action stats {full_key!r} not found; SO100 action mapping will be unavailable.")
        return None

    def observation_to_batch(self, observation: Observation) -> dict:
        observed_images = {
            1: decode_jpeg_rgb(observation.img1_jpeg),
            2: decode_jpeg_rgb(observation.img2_jpeg),
            3: decode_jpeg_rgb(observation.img3_jpeg) if observation.img3_jpeg is not None else None,
        }
        if observed_images[3] is None:
            observed_images[3] = observed_images[2]
        img1, img2, img3 = (observed_images[camera_id] for camera_id in self.camera_order)
        model_state = self.map_observation_state(observation.joints)
        self.last_model_state = model_state.copy()

        def image_to_tensor(image: np.ndarray):
            tensor = self.torch.from_numpy(image).to(dtype=self.torch.float32) / 255.0
            return tensor.permute(2, 0, 1).contiguous()

        return {
            "observation.state": self.torch.from_numpy(model_state.astype(np.float32)),
            "observation.images.camera1": image_to_tensor(img1),
            "observation.images.camera2": image_to_tensor(img2),
            "observation.images.camera3": image_to_tensor(img3),
            "task": self.task,
            "robot_type": self.robot_type,
        }

    def map_observation_state(self, joints: np.ndarray) -> np.ndarray:
        if self.state_map == "direct":
            return joints.astype(np.float32).copy()
        if self.state_map != "sim-rad-to-so100-deg":
            raise ValueError(f"Unknown SmolVLA state map: {self.state_map}")

        state = joints.astype(np.float32).copy()
        state[:5] = state[:5] / self.sim_joint_signs[:5] * (180.0 / np.pi) + self.state_offsets_deg[:5]
        state[5] = state[5] / (self.gripper_scale * self.sim_joint_signs[5]) + self.state_offsets_deg[5]
        return state

    def select_action(self, observation: Observation, step: int) -> np.ndarray:
        del step
        batch = self.preprocess(self.observation_to_batch(observation))
        if self.replan_interval > 0 and self.steps_since_replan >= self.replan_interval:
            self.model.reset()
            self.steps_since_replan = 0
        with self.torch.inference_mode():
            action = self.model.select_action(batch)
        self.steps_since_replan += 1
        self.last_model_action = action.squeeze(0).detach().cpu().numpy().astype(np.float32)
        action = self.map_model_action(action)
        action_np = action.squeeze(0).detach().cpu().numpy().astype(np.float32)
        return action_np

    def map_model_action(self, action):
        if self.action_map == "direct":
            return action
        if self.action_map != "so100-deg-to-sim-rad":
            raise ValueError(f"Unknown SmolVLA action map: {self.action_map}")
        if self._action_mean_std is None:
            raise ValueError(f"Missing action stats for {self.stats_key}.buffer.action")

        mean, std = self._action_mean_std
        so100_action = action * std + mean
        sim_action = so100_action.clone()
        signs = self.torch.as_tensor(self.sim_joint_signs, dtype=sim_action.dtype, device=sim_action.device)
        offsets = self.torch.as_tensor(
            self.action_offsets_deg, dtype=sim_action.dtype, device=sim_action.device
        )
        sim_action[..., :5] = (sim_action[..., :5] - offsets[:5]) * (np.pi / 180.0) * signs[:5]
        sim_action[..., 5] = (sim_action[..., 5] - offsets[5]) * self.gripper_scale * signs[5]
        return sim_action


def make_policy(args: argparse.Namespace):
    if args.policy == "fake":
        return FakePolicy(args.fake_action)
    if args.policy == "smolvla":
        return SmolVLAPolicyRunner(args)
    raise ValueError(f"Unknown policy: {args.policy}")


def transform_action(raw_action: np.ndarray, observation: Observation, args: argparse.Namespace) -> np.ndarray:
    if raw_action.shape != (len(SO100_JOINT_NAMES),):
        raise ValueError(f"Expected action shape {(len(SO100_JOINT_NAMES),)}, got {raw_action.shape}")

    if args.action_mode == "absolute":
        action = raw_action.astype(np.float32).copy()
    elif args.action_mode == "delta":
        action = observation.joints + raw_action.astype(np.float32) * args.delta_scale
    else:
        raise ValueError(f"Unknown action mode: {args.action_mode}")

    if args.max_delta > 0:
        delta = np.clip(action - observation.joints, -args.max_delta, args.max_delta)
        action = observation.joints + delta

    if args.action_clip > 0:
        action = np.clip(action, -args.action_clip, args.action_clip)

    if args.clip_to_isaac_limits:
        action = np.clip(action, ISAAC_JOINT_LIMIT_MIN, ISAAC_JOINT_LIMIT_MAX)

    return action.astype(np.float32)


def format_action_delta(action: np.ndarray, reference: np.ndarray) -> str:
    delta = action - reference
    return (
        f"ref={np.round(reference, 4).tolist()} "
        f"delta={np.round(delta, 4).tolist()} "
        f"mean_abs_delta={float(np.mean(np.abs(delta))):.4f}"
    )


def save_debug_observation(
    observation: Observation,
    raw_action: np.ndarray,
    action: np.ndarray,
    step: int,
    debug_dir: Path,
) -> None:
    import cv2

    debug_dir.mkdir(parents=True, exist_ok=True)
    img1 = decode_jpeg_rgb(observation.img1_jpeg)
    img2 = decode_jpeg_rgb(observation.img2_jpeg)
    img3 = decode_jpeg_rgb(observation.img3_jpeg) if observation.img3_jpeg is not None else img2
    for name, image in (("camera1", img1), ("camera2", img2), ("camera3", img3)):
        cv2.imwrite(str(debug_dir / f"{step:06d}_{name}.jpg"), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    meta = {
        "step": step,
        "timestamp": observation.timestamp,
        "joints": observation.joints.tolist(),
        "raw_action": raw_action.tolist(),
        "published_action": action.tolist(),
        "img_bytes": [
            len(observation.img1_jpeg),
            len(observation.img2_jpeg),
            len(observation.img3_jpeg) if observation.img3_jpeg is not None else 0,
        ],
    }
    (debug_dir / f"{step:06d}_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def run_bridge(args: argparse.Namespace) -> None:
    policy = make_policy(args)
    debug_dir = Path(args.save_debug_dir) if args.save_debug_dir else None
    context = zmq.Context.instance()

    obs_socket = context.socket(zmq.SUB)
    obs_socket.setsockopt(zmq.SUBSCRIBE, b"")
    obs_socket.setsockopt(zmq.RCVTIMEO, args.timeout_ms)
    obs_socket.connect(args.obs_addr)

    act_socket = context.socket(zmq.PUB)
    act_socket.bind(args.act_addr)

    # Give subscribers a moment to connect.  This avoids the usual PUB/SUB
    # first-message drop when get_action starts at nearly the same time.
    time.sleep(args.pub_warmup_s)

    print(f"Observation SUB connected to {args.obs_addr}")
    print(f"Action PUB bound to {args.act_addr}")
    print(f"Policy: {args.policy}")
    if args.policy == "fake":
        print(f"Fake action mode: {args.fake_action}")
    print("Waiting for observations. Press Ctrl+C to stop.")

    step = 0
    last_log = 0.0

    while True:
        try:
            payload = obs_socket.recv()
        except zmq.Again:
            print("No observation received yet. Is Isaac Sim playing and vla_center running?")
            continue

        try:
            observation = unpack_observation(payload)
        except Exception as exc:
            print(f"Skipping malformed observation: {exc}")
            continue

        try:
            raw_action = policy.select_action(observation, step)
            action = transform_action(raw_action, observation, args)
        except Exception as exc:
            print(f"Policy failed, skipping frame: {exc}")
            continue

        if not args.dry_run:
            act_socket.send(action.astype(np.float32).tobytes())

        if debug_dir is not None and (args.save_debug_every > 0) and (step % args.save_debug_every == 0):
            save_debug_observation(observation, raw_action, action, step, debug_dir)

        now = time.monotonic()
        if now - last_log >= args.log_interval_s:
            model_action = getattr(policy, "last_model_action", None)
            model_state = getattr(policy, "last_model_state", None)
            model_action_text = (
                f"model={np.round(model_action, 4).tolist()} " if model_action is not None else ""
            )
            model_state_text = (
                f"model_state={np.round(model_state, 4).tolist()} " if model_state is not None else ""
            )
            print(
                "dry-run action" if args.dry_run else "sent action",
                f"step={step}",
                f"ts={observation.timestamp:.3f}",
                f"joints={np.round(observation.joints, 4).tolist()}",
                model_state_text,
                model_action_text,
                f"raw={np.round(raw_action, 4).tolist()}",
                f"action={np.round(action, 4).tolist()}",
                format_action_delta(action, CUBE_LEFT_HOVER_ACTION) if args.compare_cube_hover else "",
                "img_bytes="
                f"({len(observation.img1_jpeg)}, {len(observation.img2_jpeg)}, "
                f"{len(observation.img3_jpeg) if observation.img3_jpeg is not None else 0})",
            )
            last_log = now

        step += 1
        if args.max_steps and step >= args.max_steps:
            print(f"Reached max steps: {args.max_steps}")
            return

        if args.rate_hz > 0:
            time.sleep(1.0 / args.rate_hz)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Receive isaacsim_vla_ws observations over ZMQ and publish SO100 actions."
    )
    parser.add_argument("--obs-addr", default="tcp://127.0.0.1:5556")
    parser.add_argument("--act-addr", default="tcp://127.0.0.1:5555")
    parser.add_argument("--timeout-ms", type=int, default=3000)
    parser.add_argument("--pub-warmup-s", type=float, default=0.5)
    parser.add_argument("--log-interval-s", type=float, default=1.0)
    parser.add_argument("--rate-hz", type=float, default=2.0, help="Maximum action publish rate; <=0 disables sleeping.")
    parser.add_argument("--max-steps", type=int, default=0, help="Stop after N observations; 0 means run forever.")
    parser.add_argument("--policy", choices=("fake", "smolvla"), default="fake")
    parser.add_argument(
        "--fake-action",
        choices=(
            "home",
            "test",
            "mirror",
            "alternate",
            "reach_forward_grasp_return",
            "cube_hover_grasp_return",
            "cube_left_hover_grasp_return",
        ),
        default="test",
        help="Temporary policy used before official SmolVLA inference is wired in.",
    )
    parser.add_argument("--model-id", default="lerobot/smolvla_base")
    parser.add_argument("--device", default=None, help="Torch device for SmolVLA, e.g. cuda or cpu.")
    parser.add_argument("--task", default="pick up the cube")
    parser.add_argument("--robot-type", default="so100_follower")
    parser.add_argument(
        "--smolvla-action-map",
        choices=("direct", "so100-deg-to-sim-rad"),
        default="direct",
        help="Map SmolVLA model output before sending it into the ROS/Isaac bridge.",
    )
    parser.add_argument(
        "--smolvla-state-map",
        choices=("direct", "sim-rad-to-so100-deg"),
        default="direct",
        help="Map Isaac observation.state before sending it into SmolVLA.",
    )
    parser.add_argument(
        "--smolvla-stats-key",
        choices=("so100", "so100-blue", "so100-red"),
        default="so100",
        help="Robot-specific stats key used by --smolvla-action-map=so100-deg-to-sim-rad.",
    )
    parser.add_argument(
        "--smolvla-replan-interval",
        type=int,
        default=0,
        help=(
            "Execute only the first N actions from each SmolVLA action chunk before replanning. "
            "Set 0 to keep the default policy queue behavior."
        ),
    )
    parser.add_argument(
        "--smolvla-gripper-scale",
        type=float,
        default=0.01,
        help="Scale SO100 gripper 0-100 style output into the Isaac gripper joint command.",
    )
    parser.add_argument(
        "--smolvla-sim-joint-signs",
        default="1,1,-1,1,1,1",
        help="Signs used between SO100 degree space and Isaac radian space.",
    )
    parser.add_argument(
        "--smolvla-state-offsets-deg",
        default="0,0,0,0,0,0",
        help="Offsets added after mapping Isaac joint radians into SmolVLA SO100 degree state.",
    )
    parser.add_argument(
        "--smolvla-action-offsets-deg",
        default="0,0,0,0,0,0",
        help="Offsets subtracted before mapping SmolVLA SO100 degree action into Isaac joint radians.",
    )
    parser.add_argument(
        "--camera-order",
        default="1,2,3",
        help="Map observed cameras into SmolVLA camera1,camera2,camera3 slots, e.g. 1,2,1.",
    )
    parser.add_argument(
        "--action-mode",
        choices=("absolute", "delta"),
        default="absolute",
        help="Interpret policy output as absolute joint targets or scaled joint deltas.",
    )
    parser.add_argument("--delta-scale", type=float, default=0.05)
    parser.add_argument(
        "--max-delta",
        type=float,
        default=0.15,
        help="Limit each published target to current_joints +/- max_delta. Set <=0 to disable.",
    )
    parser.add_argument(
        "--action-clip",
        type=float,
        default=3.14,
        help="Clip SmolVLA actions to [-value, value]. Set <=0 to disable.",
    )
    parser.add_argument(
        "--clip-to-isaac-limits",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Clip published joint targets to known Isaac SO100 joint limits.",
    )
    parser.add_argument("--save-debug-dir", default=None)
    parser.add_argument("--save-debug-every", type=int, default=10)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run policy and logging without publishing actions to the robot.",
    )
    parser.add_argument(
        "--compare-cube-hover",
        action="store_true",
        help="Print delta from the scripted cube-left hover reference action.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        run_bridge(args)
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
