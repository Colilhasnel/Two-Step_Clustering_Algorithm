import os


class dataset:
    INPUT_PATH = "ordered_images"
    os.path.join("ordered_images", "ordered_description.csv")

    OUTPUT_PATH = "output_images"

    # Parameters of Images
    class Variables:
        AccelaratingVoltage = "AcceleratingVoltage"
        Magnification = "Magnification"
        WorkingDistance = "WorkingDistance"
        EmissionCurrent = "EmissionCurrent"
        LensMode = "LensMode"
        Area = "Area"
        Sample = "Sample"
        Filename = "Filename"
        Centers = "Centers"

    class Sample:
        S1 = "S1"
        S2 = "S2"
        S3 = "S3"
        S4 = "S4"
        S5 = "S5"

    class Magnification:
        x50 = 50000
        x100 = 100000
        x200 = 200000

    class AccelaratingVoltage:
        kV10 = 10000
        kV20 = 20000
        kV30 = 30000

    # A function to check if a path already exists or not
    def check_path(path):
        if not os.path.exists(path):
            raise Exception("The input file cannot be find")

    # A function to make new path, if it not already exists
    def make_path(path):
        if not os.path.isdir(path):
            os.mkdir(path)

    # A function to get and set output path
    def output_dir(path=None, join=True):
        if path == None:
            return dataset.OUTPUT_PATH

        if not join:
            dataset.OUTPUT_PATH = path
        else:
            dataset.OUTPUT_PATH = os.path.join(dataset.OUTPUT_PATH, path)

        dataset.make_path(dataset.OUTPUT_PATH)

    # Defining Function to get specific images (Returns a dataframe with only images of specified set)
    def get_images(image_file, sample, mag, notes=""):
        dataset.output_dir("%s_%dk_%s" % (sample, mag / 1000, notes))

        requirements = (image_file[dataset.Variables.Sample] == sample) & (
            image_file[dataset.Variables.Magnification] == mag
        )

        return image_file.loc[requirements]

    check_path(INPUT_PATH)
    make_path(OUTPUT_PATH)


# WorkingDistance, EmissionCurrent can also be added
