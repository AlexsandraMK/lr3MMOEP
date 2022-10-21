# -*- coding: cp1251 -*-
import numpy as np
from matplotlib import pyplot as plt
from itertools import product
from collections import Counter
from functools import lru_cache

class Grid:
    
    def __init__(self, size: list, x_bounds: list , y_bounds: list):
        self.size = size
        self.x_bounds = x_bounds
        self.y_bounds = y_bounds
        self.fill_grid()
    
    def fill_grid(self):
        self.x_coords = np.array(np.linspace(self.x_bounds[0], self.x_bounds[1], self.size[0]), dtype = np.float32)
        self.y_coords = np.array(np.linspace(self.y_bounds[0], self.y_bounds[1], self.size[1]), dtype = np.float32)
        # Цыганские фокусы
        self.points = lambda: ((x, y) for x,y in product(self.x_coords, self.y_coords))

class Plan:
    
    def __init__(self, grid:Grid, num_of_observations: int):
        self.spectrum_x  = np.array(np.random.choice(grid.x_coords, size = num_of_observations), dtype = np.float32) 
        self.spectrum_y = np.array(np.random.choice(grid.y_coords, size = num_of_observations), dtype = np.float32)
        self.__establish_points_and_probs()
        
    def replace_point(self, index, new_point: tuple):
        self.spectrum_x[index] = new_point[0]
        self.spectrum_x = np.array(self.spectrum_x, dtype = np.float32) 
        self.spectrum_y[index] = new_point[1]
        self.spectrum_y = np.array(self.spectrum_y, dtype = np.float32)
        self.__establish_points_and_probs()
        
    def sum_repeating_points(self):
        counter = Counter([(x,y) for x, y in zip(self.spectrum_x, self.spectrum_y)])
        self.points = lambda: (point for point in counter.keys())
        self.probs =  1/len(self.spectrum_x) * np.array([*counter.values()])
        
    def __establish_points_and_probs(self):
        self.points = lambda: ((x,y) for x, y in zip(self.spectrum_x, self.spectrum_y))
        self.probs = [1/len(self.spectrum_x) for _ in range(len(self.spectrum_x))]

    
class Model:
    
    def __init__(self, plan: Plan, theta: np.array):
        self.plan = plan
        self.theta = theta
        self.M_matrix = self.get_M()
        self.D_matrix = self.get_D()
        self.D_criterion = self.calculate_D_criterion()
        self.D_functional = self.calculate_D_functional()
        
    
    @staticmethod
    def f(x1: float, x2: float, theta: np.array) -> np.array:
        """
        В данном случае реализовано для двухфакторной квадратичной модели.
        :param x1: входные параметры эксперимента.
        :param x2: входные параметры эксперимента
        :param theta: текущие параметры модели.
        :return: np.array
        """
        return np.array([1,x1,x2,x1*x2,x1**2,x2**2])*theta
        
    def get_M(self) -> np.array:
        M_matrix = np.zeros([len(self.theta), len(self.theta)])
        for idx, point in enumerate(self.plan.points()):
            fx: np.array = self.f(point[0], point[1], self.theta).reshape(1, -1)
            Mi = np.matmul(fx.T, fx)
            M_matrix += self.plan.probs[idx] * Mi
        return M_matrix
    
    def get_D(self) -> np.array:
        D_matrix = np.linalg.inv(self.M_matrix)
        return D_matrix
                                                              
    def calculate_D_criterion(self):
        D_criterion = np.linalg.det(self.M_matrix)
        return D_criterion
    
    def calculate_D_functional(self):
        D_functional = np.log(np.linalg.det(self.M_matrix))
        return D_functional
    
    @lru_cache(maxsize = 60)
    def calculate_variance(self, point_1: tuple, point_2: tuple,) -> float:
        fx_1: np.array = self.f(point_1[0], point_1[1], self.theta).reshape(1, -1)
        if point_1 == point_2:
            fx_2 = fx_1
        else:
            fx_2: np.array = self.f(point_2[0], point_2[1], self.theta).reshape(1, -1)
        return fx_1 @ self.D_matrix @ fx_2.T
    
class Fedorov_optimiser:
    
    def __init__(self, plan = None, *, grid: Grid, num_of_terms = 6,
                 num_of_observations = 20, num_of_iters = 5000, eps = 0.000001):
        # Если план не подан в функцию, генерируется случайный план с указанным количеством наблюдений
        self.plan = plan or Plan(grid, num_of_observations)
        self.grid = grid
        self.theta = np.ones(num_of_terms)
        self.num_of_observations = num_of_observations
        self.num_of_iters = num_of_iters
        self.eps = eps
        
        
    def optimise(self):
        
        self.last_grid_point = None
        j = 0
        for i in range(self.num_of_iters):
            
            
            self.model = Model(self.plan, self.theta)
            # Убрать
            
            self.max_delta = -np.inf
            
            #self.max_grid_x_coord, self.max_grid_y_coord, self.max_plan_spectrum_x, \
             #                                       self.max_plan_spectrum_y = None, None, None, None
            # 2-й шаг
            self.find_max_delta()
            
            # Убрать
            #print((self.plan.spectrum_x[self.idx_max_plan_point], self.plan.spectrum_y[self.idx_max_plan_point]))
            #print((self.grid.x_coords[self.idx_max_grid_point], self.grid.y_coords[self.idx_max_grid_point]))
            if self.max_delta != -np.inf :
                j = 0
                print(f'points {list(self.plan.points())}')
                print(f"Номер итерации: {i}")
                print(f'max_delta{self.max_delta}')
                print(f'D-Критерий: {self.model.D_criterion}')
                print(f'D-Функционал: {self.model.D_functional}')
            else: 
                j = j + 1
               
            # 3-й шаг
            # eps — порог (маленькая дельта)
            if np.abs(self.max_delta) < self.eps or j > 100:
                    self.plan.sum_repeating_points()
                    return list(self.plan.points()), self.plan.probs
                    
            # 4-й шаг
            else:
                self.step_4()
               
            
        else:
             print("Превышено количество итераций")
            
    
    def step_4(self):
        self.last_plan_point = self.max_plan_point
        self.last_grid_point = self.max_grid_point
        #for idx_plan,plan_point in enumerate(self.plan.points()):
            #if plan_point == self.max_plan_point:
        self.plan.replace_point(index = self.idx_max_plan_point, new_point = (self.max_grid_point[0], self.max_grid_point[1]))
    
    
    def find_max_delta(self):
        for idx_grid, grid_point in enumerate(self.grid.points()):
                for idx_plan, plan_point in enumerate(self.plan.points()):
                    if grid_point not in self.plan.points():
                        # Убрать
                        #print(f'grid_point {grid_point}')
                        #print(f'plan_point {plan_point}')
                        delta = self.calculate_delta(grid_point = grid_point, plan_point = plan_point)
                        #print(f'delta {delta}')

                        if delta > self.max_delta and delta > 0:
                            #if self.last_grid_point != None :
                            #    if grid_point[0] == self.last_plan_point[0] and grid_point[1] == self.last_plan_point[1] and plan_point[0] == self.last_grid_point[0] and plan_point[1] == self.last_grid_point[1] :
                            #        continue
                            self.max_delta = delta
                            self.max_grid_point = grid_point
                            self.max_plan_point = plan_point
                            self.idx_max_plan_point = idx_plan
                        
                    else:
                       continue
        
                                                        
    def calculate_delta(self, *, grid_point: tuple, plan_point: tuple) -> float:
        grid_variance = self.model.calculate_variance(grid_point, grid_point)
        plan_variance = self.model.calculate_variance(plan_point, plan_point)
        grid_plan_variance = self.model.calculate_variance(grid_point, plan_point)
        # Убрать
        #print(f'grid_variance: {grid_variance}')
        #print(f'plan_variance: {plan_variance}')
        #print(f'grid_plan_variance: {grid_plan_variance}')
        #print(f'num_of_observations: {self.num_of_observations}')
        return (1/self.num_of_observations) * (grid_variance - \
                                             plan_variance) - (1/(self.num_of_observations**2)) * \
                                                              (grid_variance*plan_variance - grid_plan_variance**2)



grid_size = [11,11]
x_bounds = [-1,1]
y_bounds = [-1,1]

grid = Grid(grid_size, x_bounds, y_bounds)
num_of_observations = 20
plan = Plan(grid, num_of_observations)

for i in range(0, len(plan.spectrum_x)):
    for j in range(0, len(plan.spectrum_y)):
        plt.scatter(plan.spectrum_x[i], plan.spectrum_y[j])
 # scatter - метод для нанесения маркера в точке (x1, x2)
plt.plot()
plt.show()




opt = Fedorov_optimiser(grid=grid, plan = plan, num_of_observations = num_of_observations)
opt.optimise()



for i in range(0, len(opt.plan.spectrum_x)):
    for j in range(0, len(opt.plan.spectrum_y)):
        plt.scatter(opt.plan.spectrum_x[i], opt.plan.spectrum_y[j])
 # scatter - метод для нанесения маркера в точке (x1, x2)
plt.plot()
plt.show()









