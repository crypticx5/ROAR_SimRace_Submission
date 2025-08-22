"""
ROAR Competition Solution
-------------------------

This module contains the logic for controlling your autonomous racing agent.
Only this file should be modified for the competition.  The provided
`competition_runner.py` and `infrastructure.py` files set up the race
environment and call into the functions defined here.

The implementation below uses a simple look‑ahead waypoint follower with
a proportional controller for steering and throttle.  Compared to the
starter code, the look‑ahead distance has been increased and the target
speed is adjusted dynamically based on how sharply the next turn
approaches.  Feel free to experiment with the constants to tune your
agent’s behaviour; just be careful not to alter the function signatures.
"""

from typing import List, Dict, Optional
import roar_py_interface
import numpy as np


def normalize_rad(rad: float) -> float:
    """Normalize an angle to the range [−π, π)."""
    return (rad + np.pi) % (2 * np.pi) - np.pi


def filter_waypoints(location: np.ndarray, current_idx: int, waypoints: List[roar_py_interface.RoarPyWaypoint]) -> int:
    """
    Given the vehicle’s current location, find the index of the closest
    waypoint ahead of the vehicle.  Starting the search from
    `current_idx` helps avoid scanning the entire list every step.
    """

    def dist_to_waypoint(waypoint: roar_py_interface.RoarPyWaypoint) -> float:
        return np.linalg.norm(location[:2] - waypoint.location[:2])

    for i in range(current_idx, len(waypoints) + current_idx):
        if dist_to_waypoint(waypoints[i % len(waypoints)]) < 3:
            return i % len(waypoints)
    return current_idx


class RoarCompetitionSolution:
    """
    User‑defined solution class.  The competition runner will instantiate
    this class once at the beginning of the evaluation and call its
    asynchronous methods to control the vehicle.
    """

    def __init__(
        self,
        maneuverable_waypoints: List[roar_py_interface.RoarPyWaypoint],
        vehicle: roar_py_interface.RoarPyActor,
        camera_sensor: roar_py_interface.RoarPyCameraSensor = None,
        location_sensor: roar_py_interface.RoarPyLocationInWorldSensor = None,
        velocity_sensor: roar_py_interface.RoarPyVelocimeterSensor = None,
        rpy_sensor: roar_py_interface.RoarPyRollPitchYawSensor = None,
        occupancy_map_sensor: roar_py_interface.RoarPyOccupancyMapSensor = None,
        collision_sensor: roar_py_interface.RoarPyCollisionSensor = None,
    ) -> None:
        self.maneuverable_waypoints = maneuverable_waypoints
        self.vehicle = vehicle
        self.camera_sensor = camera_sensor
        self.location_sensor = location_sensor
        self.velocity_sensor = velocity_sensor
        self.rpy_sensor = rpy_sensor
        self.occupancy_map_sensor = occupancy_map_sensor
        self.collision_sensor = collision_sensor
        # state variables
        self.current_waypoint_idx = 0

    async def initialize(self) -> None:
        """
        Called once before the race starts.  Use this hook for any
        expensive pre‑computation.  Here we simply pick an initial
        waypoint index based on the vehicle’s starting position.
        """
        # Receive location, rotation and velocity data
        vehicle_location = self.location_sensor.get_last_gym_observation()
        # Set an initial guess for the waypoint index – this need not be
        # perfect because filter_waypoints will correct it on the first step.
        self.current_waypoint_idx = 10
        self.current_waypoint_idx = filter_waypoints(
            vehicle_location,
            self.current_waypoint_idx,
            self.maneuverable_waypoints
        )

    async def step(self) -> Optional[Dict[str, float]]:
        """
        Called every simulation tick.  Reads the latest observations and
        returns a control dictionary.  This controller steers towards a
        look‑ahead waypoint and adjusts its target speed based on the
        upcoming curvature.
        """
        # Read last observations (do not call receive_observation here)
        vehicle_location = self.location_sensor.get_last_gym_observation()
        vehicle_rotation = self.rpy_sensor.get_last_gym_observation()
        vehicle_velocity = self.velocity_sensor.get_last_gym_observation()
        vehicle_velocity_norm = np.linalg.norm(vehicle_velocity)
        # Update waypoint index to the closest waypoint ahead
        self.current_waypoint_idx = filter_waypoints(
            vehicle_location,
            self.current_waypoint_idx,
            self.maneuverable_waypoints
        )
        # Choose a look‑ahead waypoint a few steps ahead to smooth steering
        look_ahead = 5
        waypoint_to_follow = self.maneuverable_waypoints[(self.current_waypoint_idx + look_ahead) % len(self.maneuverable_waypoints)]
        # Vector from vehicle to the target waypoint (in the ground plane)
        vector_to_waypoint = (waypoint_to_follow.location - vehicle_location)[:2]
        heading_to_waypoint = np.arctan2(vector_to_waypoint[1], vector_to_waypoint[0])
        # Compute heading error between current yaw and desired heading
        delta_heading = normalize_rad(heading_to_waypoint - vehicle_rotation[2])
        # Steering controller: scale inverse with speed to reduce over‑steer
        if vehicle_velocity_norm > 1e-2:
            steer_control = -5.0 / np.sqrt(vehicle_velocity_norm) * delta_heading / np.pi
        else:
            steer_control = -np.sign(delta_heading)
        steer_control = np.clip(steer_control, -1.0, 1.0)
        # Dynamic target speed: slow down for sharp turns
        abs_delta = abs(delta_heading)
        target_speed = 25.0  # m/s nominal
        if abs_delta > 0.4:
            target_speed = 15.0
        elif abs_delta > 0.2:
            target_speed = 20.0
        # Proportional throttle controller towards target speed
        throttle_control = 0.05 * (target_speed - vehicle_velocity_norm)
        # Build the control dictionary
        control = {
            "throttle": float(np.clip(throttle_control, 0.0, 1.0)),
            "steer": float(steer_control),
            "brake": float(np.clip(-throttle_control, 0.0, 1.0)),
            "hand_brake": 0.0,
            "reverse": 0,
            "target_gear": 0
        }
        # Apply action asynchronously
        await self.vehicle.apply_action(control)
        return control
