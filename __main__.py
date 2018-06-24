import argparse
import mnist_tflearn
import mnist_keras

parser = argparse.ArgumentParser(description='run classification on mnist database using different libraries')
parser.add_argument('--library', type=str,
                    help='please choose from <tflearn,>')

def main():
    args = parser.parse_args()
    print('library to be used is [... ', args.library, ' ...]')
    if args.library == "tflearn":
        mnist_tflearn.apply()
    if args.library == "keras":
        mnist_keras.apply()
    else:
        print('library...',  args.library, ' not supported yet.')


if __name__ == '__main__':
    main()
