import numpy as np
from Simulation import jit_solver_helpers

class Solver:
    def __init__(self, args=None):
        """
        class for strains
        :param args: overrides defaults
        """

        # Default Settings
        defaults =  {
            "beta": 20,
            "theta": 0.2,
            "v_max": 10,
            "gamma": np.log(10) / 2,
            "sigma": 0.25,
            "N": 40,
            "N_t": 8,
            "epsilon": 0.01,
        }
        # Merge default settings with arguments
        if args is not None:
            defaults = {**defaults, **args}

        self.beta = defaults["beta"]
        self.gamma = defaults["gamma"]
        self.theta = defaults["theta"]
        self.N_t = defaults["N_t"]
        self.N = defaults["N"]
        self.v_max = defaults["v_max"]
        self.sigma = defaults["sigma"]
        self.epsilon = defaults["epsilon"]
        self.defaults = defaults

    def f(self, x):
        """
        activation function as in model
        :param x:
        :return:
        """
        return np.tanh(self.beta * (x - self.theta))

    def rhs(self, mat, v, x):
        """
        non-stochastic part of the model.
        :param mat: Regulatory Matrix
        :param v: Growth Rate
        :param x: State
        :return:
        """
        # remve environment genes from the regulation matrix
        x_copy = x.copy()
        x_copy[:,:self.N_t] = 0

        # do the matrix multiplication to calculate the regulatory effect
        prod = x_copy @ mat.T
        prod = self.f(prod)

        results = np.zeros_like(x)
        results[:,:prod.shape[1]]  = v[:,np.newaxis]* (prod - x[:,:prod.shape[1]])
        return results

    def euler_ma_step(self, dt, x, v, mat, rng: np.random.Generator):
        """
        A single step of euler-maruyama
        :param dt: step size
        :param x: current state
        :param v: growth rate
        :param mat: regulatory matrix
        :param rng: random number provider
        :return:
        """

        dxdt_c = self.rhs(mat, v, x) * dt

        dxdt_s = np.zeros_like(dxdt_c)

        if not self.sigma == 0:
            dxdt_s[:,:mat.shape[0]] = rng.normal(0, self.sigma * np.sqrt(dt), (x.shape[0],mat.shape[0]))
            # mat.shape[0] is the number of genes excluding the inputs
        return dxdt_c + dxdt_s

    def v_f(self, x, t_values):
        """
        Growth Rate function as in the model
        :param x: state
        :param t_values: target genes
        :return: growth rate
        """
        x = x[:,:self.N_t]
        sum = np.sum(((x - t_values) / 2) ** 2,axis=1)
        return self.v_max * np.exp(-1 * self.gamma * sum)+self.epsilon

    def _integrate_internal_jit(self, x_0, steps, dt, mat, target_values):
        """
        Solve without any history with jit
        :param x_0: starting state
        :param steps: number of steps
        :param dt: step size
        :param mat: regulatory matrix
        :param target_values: environmental target
        :return: latest gene state, latest growth rate
        """
        rng = np.random.default_rng()
        return jit_solver_helpers.integrate_internal_jit(self.N_t,self.v_max,self.gamma,self.epsilon,self.beta,self.theta,self.sigma,
                                                         x_0, steps, dt, mat, target_values,rng)

    def _integrate_internal_hist_jit(self, x_0, steps, dt, mat, target_values, x_hist = False):
        """
        Solve with history. With jit.
        :param x_0: starting state
        :param steps: number of steps
        :param dt: step size
        :param mat: regulatory matrix
        :param target_values: environmental target
        :param x_hist: (Optional) return also gene state history. Very memory intensive!
        :return: (latest gene state, growth rate history) or (gene state history, growth rate history)
        """
        rng = np.random.default_rng()
        return jit_solver_helpers.integrate_internal_history_jit(self.N_t,self.v_max,self.gamma,self.epsilon,self.beta,self.theta,self.sigma,
                                                                 x_0, steps, dt, mat, target_values,rng,x_hist=x_hist)

    def _integrate_internal(self, x_0, steps, dt, mat, target_values, pbar=None):
        """
        Solve without any history
        :param x_0: starting state
        :param steps: number of steps
        :param dt: step size
        :param mat: regulatory matrix
        :param target_values: environmental target
        :param pbar: (Optional) progress bar (tqdm)
        :return: latest gene state, latest growth rate
        """
        x_i = x_0 #dimensions: time, cells, gene

        # worker thread needs to have its own rng not based on the main thread:
        # If not (when using np.random.normal directly) this might causes a fringe case by chance where the np.random gets deadlocked
        # Since this np.random.normal will be executed 10^6 times in a single generation this causes a fail after 100 generations
        # only happens when using fork instead of spawn
        rng = np.random.default_rng()
        for i in range(steps):
            v = self.v_f(x_i, target_values)
            step = self.euler_ma_step(dt, x_i, v, mat,rng)
            x_i += step

            if pbar is not None:
                pbar.update(1)

        return x_i, self.v_f(x_i, target_values)


    def _integrate_internal_history(self, x_0, steps, dt, mat, target_values, pbar=None, x_hist = False):
        """
        Solve with history
        :param x_0: starting state
        :param steps: number of steps
        :param dt: step size
        :param mat: regulatory matrix
        :param target_values: environmental target
        :param pbar: (Optional) progress bar (tqdm)
        :param x_hist: (Optional) return also gene state history. Very memory intensive!
        :return: (latest gene state, growth rate history) or (gene state history, growth rate history)
        """
        x_s = None
        if x_hist:
            x_s = np.empty((steps,x_0.shape[0],x_0.shape[1]))

        v_hist = np.empty((steps,x_0.shape[0]))
        x_i = x_0 #dimensions: time, cells, gene



        # worker thread needs to have its own rng not based on the main thread:
        # If not (when using np.random.normal directly) this might causes a fringe case by chance where the np.random gets deadlocked
        # Since this np.random.normal will be executed 10^6 times in a single generation this causes a fail after 100 generations
        # only happens when using fork instead of spawn

        rng = np.random.default_rng()

        for i in range(steps):
            v = self.v_f(x_i, target_values)
            v_hist[i,:] = v
            if x_hist:
                x_s[i,:,:] = x_i

            step = self.euler_ma_step(dt, x_i, v, mat,rng)
            x_i += step

            if pbar is not None:
                pbar.update(1)
        if x_hist:
            return x_s, v_hist
        else :
            return x_i, v_hist


    def integrate(self, x_0, t, dt, mat, environment,hist = "none",pbar = None):
        """
        Solve the model from t[0] to t[1] starting with x_0
        :param x_0: starting point. Shape should be (number_of_copies, number_of_genes]
        :param t: (starting time, end time) tuple
        :param dt: step size
        :param mat: regulatory matrix
        :param target_values: (inputs, target gene values)
        :param hist: "none" for no history fastest
                    "growth-rate-only" for growth rate history and latest gene state
                    "both" both histories
        :param pbar: (Optional) progress bar (tqdm)
        :return: t,x,v history or latest values depending on the configuration
        """
        _input = environment[0]
        target_values = environment[1]

        # append input values to the starting point
        x_0_0 = np.zeros((x_0.shape[0], x_0.shape[1] + _input.shape[0]))
        x_0_0[:, :x_0.shape[1]] = x_0
        x_0_0[:, x_0.shape[1]:] = _input  #

        timeline = np.arange(t[0], t[1], dt)
        if hist == "none":
            x,v = self._integrate_internal(x_0_0, len(timeline), dt, mat, target_values, pbar)
            return timeline,x, v

        elif hist == "growth-rate-only":
            x,v= self._integrate_internal_history(x_0_0, len(timeline), dt, mat, target_values, pbar, x_hist =False)
            return timeline, x, v

        elif hist == "both":
            x, v = self._integrate_internal_history(x_0_0, len(timeline), dt, mat, target_values, pbar, x_hist=True)
            return timeline, x, v
        else:
            raise NotImplementedError(f"{hist} is not implemented yet!")

    def integrate_jit(self, x_0, t, dt, mat, environment, hist="none"):
        """
        Solve the model from t[0] to t[1] starting with x_0 with jit
        :param x_0: starting point. Shape should be (number_of_copies, number_of_genes]
        :param t: (starting time, end time) tuple
        :param dt: step size
        :param mat: regulatory matrix
        :param environment: (inputs, target gene values)
        :param hist: "none" for no history fastest
                    "growth-rate-only" for growth rate history and latest gene state
                    "both" both histories
        :param pbar: (Optional) progress bar (tqdm)
        :return: t,x,v history or latest values depending on the configuration
        """
        _input = environment[0]
        target_values = environment[1]

        #append input values to the starting point
        x_0_0 = np.zeros((x_0.shape[0], x_0.shape[1] + _input.shape[0]))
        x_0_0[:, :x_0.shape[1]] = x_0
        x_0_0[:, x_0.shape[1]:] = _input #

        timeline = np.arange(t[0], t[1], dt)
        if hist == "none":
            x,v = self._integrate_internal_jit(x_0_0, len(timeline), dt, mat, target_values)
            return timeline, x, v

        elif hist == "growth-rate-only":
            x, v = self._integrate_internal_hist_jit(x_0_0, len(timeline), dt, mat, target_values, x_hist=False)
            return timeline, x[0], v

        elif hist == "both":
            x, v = self._integrate_internal_hist_jit(x_0_0, len(timeline), dt, mat, target_values, x_hist=True)
            return timeline, x, v
        else:
            raise NotImplementedError(f"{hist} is not implemented yet!")

    def copy(self):
        return Solver(self.defaults)