import argparse
import os
from datetime import datetime
import gym_super_mario_bros
import human_playing
import pickle

AGENTS = ['human', 'genetic']
DEFAULT_ENVIRONMENT = 'SuperMarioBros-v0'

def parse_arguments():
    parser = argparse.ArgumentParser(description="Script to train agents.")
    parser.add_argument("-agent", dest='agent', choices=AGENTS, default=AGENTS[1],
                        help="Chose kind of agent for testing.")
    parser.add_argument("i", "input_dir", dest="input_dir", help="Path for the input data.")
    parser.add_argument("-o", "-output_dir", dest="output_dir", default="", help="Path for the output data.")
    parser.add_argument("-e", "-env", dest="env", default=DEFAULT_ENVIRONMENT, help="The environment ID to play")
    parser.add_argument("-r", "-record", dest="record", action='store_true', default=False,
                        help="Record gameplay options")

    args = parser.parse_args()
    if not args.input_dir:
        parser.error("You must give a input directory path.")
    elif not os.path.isdir(args.input_dir):
        parser.error("The input path is not valid. Check if the directory was removed.")
    if not args.output_dir:
        # make default output directory
        default_output_dir = "test_output_{agent}_{date}".format(agent=args.agent, date=datetime.now().strftime("%d-%m_%H-%M"))
        if not os.path.isdir(default_output_dir):
            os.mkdir(default_output_dir)
        args.output_dir = default_output_dir
    elif not os.path.isdir(args.output_dir):
        parser.error("The output path is not valid. Check if the directory was removed.")
    return args


if __name__ == "__main__":
    args = parse_arguments()
    if args.record:
        vids_path = os.path.join(args.output_dir, "vid")
        if not os.path.isdir(vids_path):
            os.mkdir(vids_path)
    else:
        vids_path = ""
    if args.agent == "human":
        print("Stating human agent test:")
        if args.record:
            current_record_path = os.path.join(vids_path, "human_record_test.mp4")
        else:
            current_record_path = ""
        outcome = human_playing.run(env=gym_super_mario_bros.make(args.env),
                                    max_steps=args.steps_limit,
                                    standing_steps_limit=args.standing_steps_limit,
                                    allow_dying=args.allow_dying,
                                    record=current_record_path)