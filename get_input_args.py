import argparse

def get_input_args():
    # Create Parse using ArgumentParser
    parser = argparse.ArgumentParser()

    try:
    # Create 3 command line arguments as mentioned above using add_argument() from ArguementParser method
        parser.add_argument('--data_dir', type=str, default='flower_data', help='path to the folder of pet images')
        parser.add_argument('--arch', type=str, default='densenet121', help='choices a type of CNN resnet, alexnet, or vgg')
        parser.add_argument('--hidden_units ', type=int, default='512', help='path to the folder of pet images')
        in_args = parser.parse_args()

        if check_arch(in_args.arch):
            return in_args
        else:raise
    except Exception as err:
        print("Error with arch chose between vgg13, densenet161 and densenet121")
        return err

    # Replace None with parsed argument collection that
    # you created with this function

def check_arch(arch):
    if arch == "vgg13":
        return True
    elif arch == "densenet161":
        return True
    elif arch == "densenet121":
        return True
    else:
        return False


