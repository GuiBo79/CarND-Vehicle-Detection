import numpy as np

class Line():
    def __init__(self):
	
        # was the line detected in the last iteration?
        self.detected = False  
        # x values of the last n fits of the line
        self.recent_xfitted = [] 
        #average x values of the fitted line over the last n iterations
        self.bestx = None     
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None  
        #polynomial coefficients for the most recent fit
        self.current_fit = []
        #radius of curvature of the line in some units
        self.radius_of_curvature = None 
        #distance in meters of vehicle center from the line
        self.line_base_pos = None 
        #difference in fit coefficients between last and new fits
        self.diffs = [] 
        #x values for detected line pixels
        self.allx = None  
        #y values for detected line pixels
        self.ally = []
        #Center position of the Lane
        self.midpos = 0
        #Position of the car
        self.carpos = 0
        #Pixels to meters value
        self.pix_to_meters = None
        #Lane Width 
        self.width = 0

