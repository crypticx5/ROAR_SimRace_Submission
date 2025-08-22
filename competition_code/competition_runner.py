import roar_py_interface
import roar_py_carla
from submission import RoarCompetitionSolution
from infrastructure import RoarCompetitionAgentWrapper, ManualControlViewer
from typing import List, Type, Optional, Dict, Any
import carla
import numpy as np
import gymnasium as gym
import asyncio


class RoarCompetitionRule:
    """
    Rule engine for the ROAR competition.  The rule class tracks the
    vehicle’s progress around the track, determines when a lap has finished
    and handles respawns after major collisions.  This is copied verbatim
    from the official competition skeleton and should not be modified.
    """

    def __init__(
        self,
        waypoints: List[roar_py_interface.RoarPyWaypoint],
        vehicle: roar_py_carla.RoarPyCarlaActor,
        world: roar_py_carla.RoarPyCarlaWorld
    ) -> None:
        self.waypoints = waypoints
        self.vehicle = vehicle
        self.world = world
        self._last_vehicle_location = vehicle.get_3d_location()
        self._respawn_location = None
        self._respawn_rpy = None

    def initialize_race(self) -> None:
        """
        Reset state at the start of a race and rotate the waypoint list so
        that the first waypoint is ahead of the vehicle’s starting position.
        """
        self._last_vehicle_location = self.vehicle.get_3d_location()
        vehicle_location = self._last_vehicle_location
        closest_waypoint_dist = np.inf
        closest_waypoint_idx = 0
        for i, waypoint in enumerate(self.waypoints):
            waypoint_dist = np.linalg.norm(vehicle_location - waypoint.location)
            if waypoint_dist < closest_waypoint_dist:
                closest_waypoint_dist = waypoint_dist
                closest_waypoint_idx = i
        # rotate waypoints so that the closest point becomes the start
        self.waypoints = self.waypoints[closest_waypoint_idx + 1:] + self.waypoints[:closest_waypoint_idx + 1]
        self.furthest_waypoints_index = 0
        print(f"total length: {len(self.waypoints)}")
        # save respawn state
        self._respawn_location = self._last_vehicle_location.copy()
        self._respawn_rpy = self.vehicle.get_roll_pitch_yaw().copy()

    def lap_finished(self, check_step: int = 5) -> bool:
        """Return True when the vehicle has visited all waypoints."""
        return self.furthest_waypoints_index + check_step >= len(self.waypoints)

    async def tick(self, check_step: int = 15) -> None:
        """
        Advance the rule engine by one world step.  It projects the vehicle’s
        movement onto a short segment of upcoming waypoints and updates
        `furthest_waypoints_index` accordingly.
        """
        current_location = self.vehicle.get_3d_location()
        delta_vector = current_location - self._last_vehicle_location
        delta_vector_norm = np.linalg.norm(delta_vector)
        delta_vector_unit = (delta_vector / delta_vector_norm) if delta_vector_norm >= 1e-5 else np.zeros(3)
        previous_furthest_index = self.furthest_waypoints_index
        min_dis = np.inf
        min_index = 0
        endind_index = previous_furthest_index + check_step if (previous_furthest_index + check_step <= len(self.waypoints)) else len(self.waypoints)
        for i, waypoint in enumerate(self.waypoints[previous_furthest_index:endind_index]):
            waypoint_delta = waypoint.location - current_location
            projection = np.dot(waypoint_delta, delta_vector_unit)
            projection = np.clip(projection, 0, delta_vector_norm)
            closest_point_on_segment = current_location + projection * delta_vector_unit
            distance = np.linalg.norm(waypoint.location - closest_point_on_segment)
            if distance < min_dis:
                min_dis = distance
                min_index = i
        # update index and state
        self.furthest_waypoints_index += min_index
        self._last_vehicle_location = current_location
        print(f"reach waypoints {self.furthest_waypoints_index} at {self.waypoints[self.furthest_waypoints_index].location}")

    async def respawn(self) -> None:
        """
        Move the vehicle back to its last saved respawn point.  The engine will
        call this function after a major collision.  It waits for a few
        simulation steps to allow the physics to stabilise.
        """
        self.vehicle.set_transform(
            self._respawn_location, self._respawn_rpy
        )
        self.vehicle.set_linear_3d_velocity(np.zeros(3))
        self.vehicle.set_angular_velocity(np.zeros(3))
        for _ in range(20):
            await self.world.step()
        self._last_vehicle_location = self.vehicle.get_3d_location()
        self.furthest_waypoints_index = 0


async def evaluate_solution(
    world: roar_py_carla.RoarPyCarlaWorld,
    solution_constructor: Type[RoarCompetitionSolution],
    max_seconds: int = 12000,
    enable_visualization: bool = False,
) -> Optional[Dict[str, Any]]:
    """
    Spawn a vehicle, attach sensors and run the provided solution.  The
    evaluation ends when the solution completes three laps, times out or
    experiences a blocking error.  If `enable_visualization` is True a
    simple Pygame window will be displayed.
    """
    if enable_visualization:
        viewer = ManualControlViewer()
    # spawn the vehicle and attach sensors
    waypoints = world.maneuverable_waypoints
    vehicle = world.spawn_vehicle(
        "vehicle.tesla.model3",
        waypoints[0].location + np.array([0, 0, 1]),
        waypoints[0].roll_pitch_yaw,
        True,
    )
    assert vehicle is not None
    camera = vehicle.attach_camera_sensor(
        roar_py_interface.RoarPyCameraSensorDataRGB,
        np.array([-2.0 * vehicle.bounding_box.extent[0], 0.0, 3.0 * vehicle.bounding_box.extent[2]]),
        np.array([0, 10 / 180.0 * np.pi, 0]),
        image_width=1024,
        image_height=768
    )
    location_sensor = vehicle.attach_location_in_world_sensor()
    velocity_sensor = vehicle.attach_velocimeter_sensor()
    rpy_sensor = vehicle.attach_roll_pitch_yaw_sensor()
    occupancy_map_sensor = vehicle.attach_occupancy_map_sensor(
        50,
        50,
        2.0,
        2.0
    )
    collision_sensor = vehicle.attach_collision_sensor(
        np.zeros(3),
        np.zeros(3)
    )
    # sanity checks
    assert camera is not None
    assert location_sensor is not None
    assert velocity_sensor is not None
    assert rpy_sensor is not None
    assert occupancy_map_sensor is not None
    assert collision_sensor is not None
    # build the agent and rule
    solution: RoarCompetitionSolution = solution_constructor(
        waypoints,
        RoarCompetitionAgentWrapper(vehicle),
        camera,
        location_sensor,
        velocity_sensor,
        rpy_sensor,
        occupancy_map_sensor,
        collision_sensor
    )
    # repeat waypoints three times to run three laps
    rule = RoarCompetitionRule(waypoints * 3, vehicle, world)
    # warm up simulation
    for _ in range(20):
        await world.step()
    rule.initialize_race()
    # start timer
    start_time = world.last_tick_elapsed_seconds
    current_time = start_time
    await vehicle.receive_observation()
    await solution.initialize()
    while True:
        # terminate if time out
        current_time = world.last_tick_elapsed_seconds
        if current_time - start_time > max_seconds:
            vehicle.close()
            return None
        # receive sensors' data
        await vehicle.receive_observation()
        await rule.tick()
        # major collision detection
        collision_impulse_norm = np.linalg.norm(collision_sensor.get_last_observation().impulse_normal)
        if collision_impulse_norm > 100.0:
            print(f"major collision of tensity {collision_impulse_norm}")
            await rule.respawn()
        # check completion
        if rule.lap_finished():
            break
        # optional visualization
        if enable_visualization:
            if viewer.render(camera.get_last_observation()) is None:
                vehicle.close()
                return None
        # step agent and world
        await solution.step()
        await world.step()
    print("end of the loop")
    end_time = world.last_tick_elapsed_seconds
    vehicle.close()
    if enable_visualization:
        viewer.close()
    return {
        "elapsed_time": end_time - start_time,
    }


async def main() -> None:
    """Convenience entry point for running this module standalone."""
    carla_client = carla.Client('127.0.0.1', 2000)
    carla_client.set_timeout(5.0)
    roar_py_instance = roar_py_carla.RoarPyCarlaInstance(carla_client)
    world = roar_py_instance.world
    world.set_control_steps(0.05, 0.005)
    world.set_asynchronous(False)
    evaluation_result = await evaluate_solution(
        world,
        RoarCompetitionSolution,
        max_seconds=5000,
        enable_visualization=True
    )
    if evaluation_result is not None:
        print("Solution finished in {} seconds".format(evaluation_result["elapsed_time"]))
    else:
        print("Solution failed to finish in time")


if __name__ == "__main__":
    asyncio.run(main())
