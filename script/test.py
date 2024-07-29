from adversarial_training_toolkit.defense import GairatTraining

if __name__ == "__main__":
    GairatTraining("cifar100", "res18", epochs=1)()
