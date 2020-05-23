

class Point:
    """ Point class with additional features so that it can be used with the A-star algorithm """
    
    X = None
    Y = None

    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def not_diagonal(self):
        if self.X == 0:
            return True
        elif self.Y == 0:
            return  True
        return False
        
    def abs_sum(self):
        return abs(self.X) + abs(self.Y)
        
    def xstr(self,s):
        if s is None:
            return 'NULL'
        return str(s)

    def __str__(self):
        return "(" + self.xstr(self.X) + "," + self.xstr(self.Y) + ")"
    
    def toTuple(self):
        return (self.X, self.Y)
    
    def __eq__(self, other):
        #try catch because sometimes comparing actions compars a point to None
        try:
            return self.X == other.X and self.Y == other.Y
        except:
            return False
    
    # Needed for A-star
    def __hash__(self):
        return hash(self.toTuple())
    
    def __add__(self, other):
        return Point(self.X + other.X, self.Y + other.Y)
    
    # Needed for A-star
    def __lt__(self, other):
        return self.X < other.X or (self.X == other.X and self.Y < other.Y)
    
    def __repr__(self):
        return  "Point (" + str(self.X) + ", " + str(self.Y) + ")"