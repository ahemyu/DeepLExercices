from .pattern import Checker, Circle, Spectrum

if __name__ == "__main__":
    print("Checker: \n")
    # checker = Checker(resolution=16, tile_size=3)
    # checker.draw()
    # checker.show()

    print("Circle: \n") 
    circle = Circle(resolution=10, radius=4, position=(5,5))
    # circle.draw()
    # circle.show()
    
    print("Spectrum: \n")
    spectrum  = Spectrum(resolution=500)
    spectrum.draw()
    spectrum.show()