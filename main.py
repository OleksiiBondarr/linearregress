import numpy as np
def generate_wave_set(n_support=1000, n_train=25, std=0.3):
    data = {}
    # выберем некоторое количество точек из промежутка от 0 до 2*pi
    data['support'] = np.linspace(0, 2*np.pi, num=n_support)
    # для каждой посчитаем значение sin(x) + 1
    # это будет ground truth
    data['values'] = np.sin(data['support']) + 1
    # из support посемплируем некоторое количество точек с возвратом, это будут признаки
    data['x_train'] = np.sort(np.random.choice(data['support'], size=n_train, replace=True))
    # опять посчитаем sin(x) + 1 и добавим шум, получим целевую переменную
    data['y_train'] = np.sin(data['x_train']) + 1 + np.random.normal(0, std, size=data['x_train'].shape[0])
    return data
data = generate_wave_set(1000, 250)
X = np.array([np.ones(data['x_train'].shape[0]), data['x_train']]).T
w = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), data['y_train'])
y_hat = np.dot(w, X.T)
print(y_hat)

import matplotlib.pyplot as plt
plt.plot(data['support'], data['values'])
plt.scatter(data['x_train'], data['y_train'])

plt.plot(data['x_train'], y_hat, 'r')
plt.show()