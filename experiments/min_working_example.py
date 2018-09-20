import GPyOpt
import numpy as np

np.random.seed(1234)


def long_evaluate(x):
    # long operation here
    return ((x ** 2) - (2 * x) + 1)


def get_n_suggestions(n, domain, trials):
    suggestions = []
    X_step, Y_step = trials

    for i in range(0, n):
        bo_step = GPyOpt.methods.BayesianOptimization(f=None,
                                                      domain=domain,
                                                      X=X_step,
                                                      Y=Y_step,
                                                      de_duplication=True)
        x_next = bo_step.suggest_next_locations(pending_X=suggestions)
        if len(suggestions) == 0:
            suggestions = x_next
        else:
            suggestions = np.vstack((suggestions, x_next))

    print('_______________')
    print(X_step.flatten().tolist())
    print(Y_step.flatten().tolist())
    print(suggestions)
    print('_______________')
    return suggestions


num_iters = 30
n = 3

performed_iters = 0
X = np.array([[-50], [25]])
Y = np.array([[long_evaluate(-5)], [long_evaluate(10)]])
domain = [{'name': 'var_1', 'type': 'discrete', 'domain': range(-100, 100)}]

while performed_iters < num_iters:
    suggestions = get_n_suggestions(n, domain, (X, Y))
    y_s = list(map(long_evaluate, suggestions.flatten().tolist()))
    y_array = np.array([[v] for v in y_s])
    X = np.vstack((X, suggestions))
    Y = np.vstack((Y, y_array))
    performed_iters += n
    print(suggestions.T)