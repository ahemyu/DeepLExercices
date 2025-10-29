from .pattern import Checker, Circle

if __name__ == "__main__":
    print("Checker: ")
    checker = Checker(resolution=4, tile_size=2)
    checker.draw()
    checker.show()

    print("Circle: ") 
    circle = Circle()
    circle.draw()
    circle.show()
    

