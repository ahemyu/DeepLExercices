from .pattern import Checker, Circle

if __name__ == "__main__":
    print("Checker: ")
    checker = Checker(resolution=16, tile_size=3)
    checker.draw()
    checker.show()

    print("Circle: ") 
    circle = Circle()
    circle.draw()
    circle.show()
    

