from .pattern import Checker, Circle, Spectrum

from .generator import ImageGenerator
if __name__ == "__main__":
    #TODO: comment out
    # print("Checker: \n")
    # checker = Checker(resolution=16, tile_size=3)
    # checker.draw()
    # checker.show()

    # print("Circle: \n") 
    # circle = Circle(resolution=10, radius=4, position=(5,5))
    # circle.draw()
    # circle.show()
    
    # print("Spectrum: \n")
    # spectrum  = Spectrum(resolution=500)
    # spectrum.draw()
    # spectrum.show()   

    generator = ImageGenerator(
                            file_path="/home/ahemyu/drive/Master/Erst/DL/Exercices/exercise0_material/src_to_implement/exercise_data",
                            label_path="/home/ahemyu/drive/Master/Erst/DL/Exercices/exercise0_material/src_to_implement/Labels.json", 
                            batch_size=35,
                            image_size=[32,32, 3],
                            rotation=False,
                            mirroring=False,
                            shuffle=False
                            )

    generator.next()
    generator.next()
    generator.next()
    generator.next()