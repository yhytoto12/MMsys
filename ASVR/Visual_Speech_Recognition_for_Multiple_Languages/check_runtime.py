import json
import argparse

def main(args):
    with open(args.filename,'r') as f:
        data = json.load(f)
        total_time=0
        for event in data['traceEvents']:
            if 'load_data' in event['name']:
                print(f"Time for function call load_data: {event['dur']}")
                total_time +=event['dur']
            if 'process_landmarks' in event['name']:
                print(f"Time for function call process_landmarks: {event['dur']}")
                total_time +=event['dur']
        print(f"Total time for data loading: {total_time}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--filename', type=str)
    args = parser.parse_args()
    main(args)