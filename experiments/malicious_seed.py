import GPyOpt
import numpy as np

np.random.seed(1234)

X_step = np.array([[x] for x in [-50.0, 25.0, -51.0, -52.0, -49.0,
                                 24.0, 26.0, 41.0, 23.0,27.0, 22.0,
                                 20.0, 21.0, 28.0, 18.0, 17.0, 19.0,
                                 11.0, 10.0, 9.0, 5.0, 6.0, 4.0, 3.0,
                                 2.0, 1.0]])

Y_step = np.array([[y] for y in [36.0, 81.0, 2704.0, 2809.0, 2500.0,
                                 529.0, 625.0, 1600.0, 484.0, 676.0,
                                 441.0, 361.0, 400.0, 729.0, 289.0,
                                 256.0, 324.0, 100.0, 81.0, 64.0, 16.0,
                                 25.0, 9.0, 4.0, 1.0, 0.0]])

domain = [{'name': 'var_1', 'type': 'discrete', 'domain': range(-100, 100)}]
suggestions = None

for i in range(0,30):
    if i == 28:
        print('aha')
    bo_step = GPyOpt.methods.BayesianOptimization(f=None,
                                                  domain=domain,
                                                  X=X_step,
                                                  Y=Y_step,
                                                  de_duplication=True)
    x_next = bo_step.suggest_next_locations(pending_X=suggestions)
    if suggestions is None:
        suggestions = x_next
    else:
        suggestions = np.vstack((suggestions, x_next))
print(suggestions.T)
