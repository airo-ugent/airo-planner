class PlannerError(RuntimeError):
    """Base class for all things that can go wrong during planning.
    Mostly intended for problems that can happen, but are not per se
    the user's fault, e.g. no valid IK solutions, no valid paths etc.

    Note that not all planner problems need to be a subclass of this
    error, e.g. you can still use ValueError, AttributeError etc.
    """
