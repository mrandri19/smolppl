from scipy.stats import norm


class Distribution:
    @staticmethod
    def log_density(point, params):
        raise NotImplementedError("Must be implemented by a subclass")

    @staticmethod
    def sample(params):
        raise NotImplementedError("Must be implemented by a subclass")


class Normal(Distribution):
    @staticmethod
    def log_density(point, params):
        return float(norm.logpdf(point, params[0], params[1]))

    @staticmethod
    def sample(params):
        return float(norm.rvs(loc=params[0], scale=params[1]))


class LatentVariable:
    def __init__(self, name, dist_class, dist_args):
        self.name = name
        self.dist_class = dist_class
        self.dist_args = dist_args


class ObservedVariable:
    def __init__(self, name, dist_class, dist_args, observed):
        self.name = name
        self.dist_class = dist_class
        self.dist_args = dist_args
        self.observed = observed


def evaluate_log_density(variable, latent_values):
    visited = set()
    variables = []

    def collect_variables(variable):
        if isinstance(variable, float):
            return

        visited.add(variable)
        variables.append(variable)

        for arg in variable.dist_args:
            if arg not in visited:
                collect_variables(arg)

    collect_variables(variable)

    log_density = 0.0
    for variable in variables:
        dist_params = []
        for dist_arg in variable.dist_args:
            if isinstance(dist_arg, float):
                dist_params.append(dist_arg)
            if isinstance(dist_arg, LatentVariable):
                dist_params.append(latent_values[dist_arg.name])

        if isinstance(variable, LatentVariable):
            log_density += variable.dist_class.log_density(
                latent_values[variable.name], dist_params
            )
        if isinstance(variable, ObservedVariable):
            log_density += variable.dist_class.log_density(
                variable.observed, dist_params
            )

    return log_density


def prior_sample(root):
    visited = set()
    variables = []

    def collect_variables(variable):
        if isinstance(variable, float):
            return

        visited.add(variable)

        for arg in variable.dist_args:
            if arg not in visited:
                collect_variables(arg)

        # post-order
        variables.append(variable)

    collect_variables(root)

    sampled_values = {}
    for variable in variables:
        dist_params = []
        for dist_arg in variable.dist_args:
            if isinstance(dist_arg, float):
                dist_params.append(dist_arg)
            else:
                dist_params.append(sampled_values[dist_arg.name])

        sampled_values[variable.name] = variable.dist_class.sample(dist_params)

    return sampled_values[root.name]


def posterior_sample(root, latent_values):
    visited = set()
    variables = []

    def collect_variables(variable):
        if isinstance(variable, float):
            return

        visited.add(variable)

        for arg in variable.dist_args:
            if arg not in visited:
                collect_variables(arg)

        # post-order
        variables.append(variable)

    collect_variables(root)

    sampled_values = {}
    for variable in variables:
        dist_params = []
        for dist_arg in variable.dist_args:
            if isinstance(dist_arg, float):
                dist_params.append(dist_arg)
            else:
                dist_params.append(sampled_values[dist_arg.name])

        if isinstance(variable, LatentVariable):
            sampled_values[variable.name] = latent_values[variable.name]
        if isinstance(variable, ObservedVariable):
            sampled_values[variable.name] = variable.dist_class.sample(dist_params)

    return sampled_values[root.name]
