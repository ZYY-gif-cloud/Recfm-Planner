from diffusers.optimization import Union, SchedulerType, Optional, Optimizer, TYPE_TO_SCHEDULER_FUNCTION
def get_scheduler(name: Union[str, SchedulerType], optimizer: Optimizer, num_warmup_steps: Optional[int] = None, num_training_steps: Optional[int] = None, **kwargs):
    name = SchedulerType(name)
    schedule_func = TYPE_TO_SCHEDULER_FUNCTION[name]
    if name == SchedulerType.CONSTANT:
        return schedule_func(optimizer, **kwargs)
    if num_warmup_steps is None:
        raise ValueError(f"{name} 需要 `num_warmup_steps`，请提供该参数。")
    if name == SchedulerType.CONSTANT_WITH_WARMUP:
        return schedule_func(optimizer, num_warmup_steps=num_warmup_steps, **kwargs)
    if num_training_steps is None:
        raise ValueError(f"{name} 需要 `num_training_steps`，请提供该参数。")
    return schedule_func(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps, **kwargs)

