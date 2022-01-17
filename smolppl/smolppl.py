from scipy.stats import norm


class Distribution:
    @staticmethod
    def log_density(point, params):
        raise NotImplementedError("Must be implemented by a subclass")


class Normal(Distribution):
    @staticmethod
    def log_density(point, params):
        return float(norm.logpdf(point, params[0], params[1]))


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
