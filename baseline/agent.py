from __future__ import annotations

from env import Action, ActionType, CarType, TaxiEnv, Tier


class HeuristicAgent:
    """
    Deterministic rule-based agent.
    Provides a reproducible lower-bound score for each task.
    """

    def __init__(self, env: TaxiEnv) -> None:
        self.env = env

    def act(self, obs) -> Action:
        env = self.env
        waiting = env.waiting_riders()
        cars = env.available_cars()
        drivers = env.available_drivers()

        if not waiting:
            if env._task and env._task.enable_repositioning and drivers:
                hot_zones = env._demand_heatmap()
                if hot_zones and hot_zones[0][1] > 0:
                    return Action(
                        action_type=ActionType.REPOSITION,
                        driver_id=drivers[0].driver_id,
                        target_zone=hot_zones[0][0],
                    )
            return Action(action_type=ActionType.NO_OP)

        if env._task and env._task.enable_pooling and len(waiting) >= 2 and cars and drivers:
            for i in range(len(waiting)):
                for j in range(i + 1, len(waiting)):
                    r1, r2 = waiting[i], waiting[j]
                    if (
                        r1.pickup_zone == r2.pickup_zone
                        and r1.share_allowed
                        and r2.share_allowed
                        and r1.group_size + r2.group_size <= max(c.capacity for c in cars)
                        and abs(r1.drop_zone - r2.drop_zone) <= 2
                    ):
                        car = self._best_car_for_pair(r1, r2, cars)
                        if car is not None:
                            driver = self._driver_for_car(car, drivers)
                            if driver is not None:
                                return Action(
                                    action_type=ActionType.POOL,
                                    rider_ids=[r1.rider_id, r2.rider_id],
                                    car_id=car.car_id,
                                    driver_id=driver.driver_id,
                                )

        rider = waiting[0]
        car = self._best_car_for_rider(rider, cars)
        if car is not None:
            driver = self._driver_for_car(car, drivers)
            if driver is not None:
                return Action(
                    action_type=ActionType.ASSIGN,
                    rider_id=rider.rider_id,
                    car_id=car.car_id,
                    driver_id=driver.driver_id,
                )

        if env._task and env._task.enable_repositioning and drivers:
            hot_zones = env._demand_heatmap()
            if hot_zones and hot_zones[0][1] > 0:
                return Action(
                    action_type=ActionType.REPOSITION,
                    driver_id=drivers[0].driver_id,
                    target_zone=hot_zones[0][0],
                )

        return Action(action_type=ActionType.NO_OP)

    def _best_car_for_rider(self, rider, cars):
        preferred = []
        if rider.tier == Tier.VIP:
            preferred = [CarType.ARMORED]
        elif rider.group_size >= 4:
            preferred = [CarType.XUV]
        elif rider.tier == Tier.PREMIUM:
            preferred = [CarType.SEDAN]
        else:
            preferred = [CarType.BASIC]

        for ctype in preferred:
            for car in cars:
                if car.car_type == ctype and car.capacity >= rider.group_size:
                    return car

        for car in cars:
            if car.capacity >= rider.group_size:
                return car
        return None

    def _best_car_for_pair(self, r1, r2, cars):
        total = r1.group_size + r2.group_size
        for car in cars:
            if car.capacity >= total:
                return car
        return None

    def _driver_for_car(self, car, drivers):
        if car.driver_id:
            for d in drivers:
                if d.driver_id == car.driver_id:
                    return d
        return drivers[0] if drivers else None
