# pid.py
import math

class PIDController:
    """
    Standard 1D PID Controller.
    """

    def __init__(self, kp, ki, kd, *, out_min=None, out_max=None, i_min=-1.0, i_max=1.0, d_alpha=1.0):
        self.kp = float(kp)
        self.ki = float(ki)
        self.kd = float(kd)

        self._i = 0.0
        self._e_prev = 0.0
        self._d_f = 0.0
        self._primed = False

        self.out_min = out_min
        self.out_max = out_max
        self.i_min = float(i_min)
        self.i_max = float(i_max)
        self.d_alpha = max(1e-6, min(1.0, float(d_alpha)))

    def compute(self, error, dt):
        if dt <= 0 or not math.isfinite(dt):
            return self._apply(self.kp * error)

        if not self._primed:
            self._e_prev = float(error)
            self._d_f = 0.0
            self._primed = True

        self._i += float(error) * dt
        if self._i < self.i_min: self._i = self.i_min
        if self._i > self.i_max: self._i = self.i_max

        d_raw = (float(error) - self._e_prev) / dt
        self._d_f = (1.0 - self.d_alpha) * self._d_f + self.d_alpha * d_raw
        self._e_prev = float(error)

        u = self.kp * float(error) + self.ki * self._i + self.kd * self._d_f
        return self._apply(u)

    def _apply(self, u):
        if self.out_min is not None and u < self.out_min:
            return self.out_min
        if self.out_max is not None and u > self.out_max:
            return self.out_max
        return u

    def reset(self):
        self._i = 0.0
        self._e_prev = 0.0
        self._d_f = 0.0
        self._primed = False
