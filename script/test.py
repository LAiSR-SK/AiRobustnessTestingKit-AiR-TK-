from airtk.defense import GairatTraining


if __name__ == "__main__":
    GairatTraining("cifar10", "res18", epochs=2)()
