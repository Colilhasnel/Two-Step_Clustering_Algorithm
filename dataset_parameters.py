import os


class dataset:
    INPUT_PATH = os.path.join("ordered_images", "ordered_description.csv")

    OUTPUT_PATH = "output_images"

    class Variables:
        AccelaratingVoltage = "AcceleratingVoltage"
        Magnification = "Magnification"
        WorkingDistance = "WorkingDistance"
        EmissionCurrent = "EmissionCurrent"
        LensMode = "LensMode"
        Area = "Area"
        Sample = "Sample"
        Filename = "Filename"

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


# WorkingDistance, EmissionCurrent can also be added
