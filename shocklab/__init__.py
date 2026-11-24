"""Shock Lab: viscous Burgers' equation, from a PINN to an FNO to a PINO.

Governing equation (periodic x in [-1, 1), t in [0, 1]):

    u_t + u u_x = nu u_xx,   nu = 0.01 / pi

See CLAUDE.md for the full specification. The physics core (solver, exact
solution, initial conditions) is ported verbatim from 01_physics_and_data.ipynb.
"""

__version__ = "0.1.0"
