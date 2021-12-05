class PI:
    """ class for pid controller """
    
    def __init__(self, kp, ki, kd, y_s, dt, u_max, AW=1):
        """ controll parameters """

        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.y_s0 = y_s # original setpoint during initialisation
        self.y_s = y_s # setpoint
        self.dt = dt # sample rate
        self.u_max = u_max # maximum u
        self.AW = AW # anti-windup

        # init var for integral
        self.e_old = 0
        self.e_i = 0
        self.i_add = 0
        
    def get_u(self, y):
        """ determine u """
        
        e = y - self.y_s # calculate error

        # integral part
        self.e_i += self.i_add * self.dt 
        self.e_d = e - self.e_old
        self.e_old = e
        self.i_add = e

        u = - (self.kp*e + self.ki*self.e_i + self.kd*self.e_d)

        #  Sollgrößenbeschränkung
        if u > self.u_max:
            u = self.u_max

            # clamping: stop integrator
            if self.AW==1:
                self.i_add = 0

        elif u < 0:
            u = 0

            # clamping: stop integrator
            if self.AW==1:
                self.i_add = 0

        return u