import argparse

parser = argparse.ArgumentParser(description='run classification on mnist database using different libraries')
parser.add_argument('--library', type=str,
                    help='please choose from <tflearn,>')


def main():
    args = parser.parse_args()
    print('library to be used is [... ', args.library, ' ...]')


if __name__ == '__main__':
    main()
