class react_dummy:
    name = 'react_dummy'
    
    def __init__(self):
        self.react_on = 0
        self.limit_reserve = 0

    def react_dist(self, tank, u):
        self.react_on = 0
    
    def adjust_setpoint(self, data, tank):
        tank.pi["0"].correct = 0