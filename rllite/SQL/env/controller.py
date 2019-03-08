
class Player:
    def __init__(self, x, y, terrain):
        self.x = x
        self.y = y
        self.terrain = terrain

    def getposition(self):
        return (self.x, self.y)

    def action(self, xy_m):
        self.x += xy_m[0]
        self.y += xy_m[1]
        self.x = min(self.terrain.bounds_x[1], max(self.x, self.terrain.bounds_x[0]))
        self.y = min(self.terrain.bounds_y[1], max(self.y, self.terrain.bounds_y[0]))
        return self.terrain.getreward()
