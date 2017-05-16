import sys

from yelp_ds.yelp_runner import main

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Please enter tag small or big')
        sys.exit(-1)

    tag = sys.argv[1]
    main(tag)