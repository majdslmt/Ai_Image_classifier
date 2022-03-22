import argparse

def get_input_args():
    # Create Parse using ArgumentParser
    parser = argparse.ArgumentParser()

    try:
    # Create 8 command line arguments as mentioned above using add_argument() from ArguementParser method
        parser.add_argument('--data_dir', type=str, default='flower_data', help='path to the folder of pet images')
        parser.add_argument('--arch', type=str, default='densenet121', help='choices a type of models, or vgg')
        parser.add_argument('--hidden_units ', type=int, default='512', help='hidden_units')
        parser.add_argument('--input', type=str, default='', help='path to the checkpoint')
        parser.add_argument('--top_k', type=int, default='5', help='top_k')
        parser.add_argument('--category_names', type=str, default='cat_to_name.json', help='path to the category_names file')
        parser.add_argument('--image_path', type=str, default='cat_to_name.json', help='path to the image')
        parser.add_argument('--gpu', type=int, default='1', help='choices type of device')
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

def check_device(gpu):
    if gpu == 1:
        return "cuda"
    elif gpu == 0:
        return "cpu"

