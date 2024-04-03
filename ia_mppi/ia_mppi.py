from mppi_torch.mppi import MPPIPlanner

class IAMPPIPlanner:
    def __init__(self, agents_cfg, sys_cfg) -> None:
        self._device = sys_cfg["device"]
        self._dt = sys_cfg["dt"]
        self._environment = self._initalize_environment(agents_cfg, sys_cfg)
        dynamics = OmnidirectionalPointRobotDynamics(
        dt=CONFIG["dt"], device=CONFIG["device"]
        )
        objective = Objective(goal=CONFIG["goal"], device=CONFIG["device"])
        planner = MPPIPlanner(
            cfg=CONFIG["mppi"],
            nx=6,
            dynamics=dynamics.step,
            running_cost=objective.compute_running_cost,
        )