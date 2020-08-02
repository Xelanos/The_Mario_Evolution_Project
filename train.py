import argparse
import os
from datetime import datetime
from pandas import DataFrame
import gym_super_mario_bros
import human_playing


DEFAULT_ENVIRONMENT = 'SuperMarioBros-v0'
DEFAULT_STEP_LIMIT = 2000
DEFAULT_NO_ADVANCE_STEP_LIMIT = 100
AGENTS = ['human', 'genetic']
TRIALS = 10
RECORDE_OPTIONS = ["none", 'some', 'all']
DEFAULT_RECORDE_FREQUENCY = 100

def parse_arguments():
    parser = argparse.ArgumentParser(description="Script to train agents.")
    parser.add_argument("-agent", dest='agent', choices=AGENTS, default=AGENTS[0],
                        help="Chose kind of agent for training.")
    parser.add_argument("-o", "-output_dir", dest="output_dir", default="", help="Path for the output data.")
    parser.add_argument("-n", "-num_of_trials", dest='num_of_trials', type=int, default=TRIALS, help="Number of trials.")
    parser.add_argument("-s", '-time_scale', "-steps_limit", dest='steps_limit', default=DEFAULT_STEP_LIMIT, type=int,
                        help="The maximal frames to a trail.")
    parser.add_argument("-no_action_limit", "-no_advance_limit", "-standing_limit", "-no_progress_limit",
                        dest='standing_steps_limit', default=DEFAULT_NO_ADVANCE_STEP_LIMIT, help="Limit the number of"
                        " steps the agent allow not to change the x position.")
    parser.add_argument("-d","-allow_death", "-allow_dying", dest="allow_dying", action='store_false',
                        help="Allow agent to die in a trail.")
    parser.add_argument("-e", "-env", dest="env", default=DEFAULT_ENVIRONMENT, help="The environment ID to play")
    parser.add_argument("-r", "-record", dest="record", choices=RECORDE_OPTIONS, default=RECORDE_OPTIONS[0],
                        help="Record gameplay options")
    parser.add_argument("-rf", "-record_frequency", dest="record_frequency", default=DEFAULT_RECORDE_FREQUENCY, type=int,
                        help="The frequency of trails that will be recoded if 'some' was chosen for record option.")


    args = parser.parse_args()
    if args.num_of_trials < 0:
        parser.error("number of trials must to be positive.")
    if args.steps_limit < 1:
        parser.error("steps limit must to be positive.")
    if not args.output_dir:
        # make default output directory
        default_output_dir = "train_output_{agent}_{date}".format(agent=args.agent, date=datetime.now().strftime("%d-%m_%H-%M"))
        if not os.path.isdir(default_output_dir):
            os.mkdir(default_output_dir)
        args.output_dir = default_output_dir
    elif not os.path.isdir(args.output_dir):
        parser.error("The output path is not valid. Check if the directory was removed.")
    if args.record_frequency < 1 and args.record == RECORDE_OPTIONS[1]:
        print("Invalid record frequency argument. No record will be made.")
        args.record = RECORDE_OPTIONS[0]
    return args


if __name__ == "__main__":
    #output_data = DataFrame(columns=['avg_reward', 'steps', 'score', 'deaths', 'finish_level'])
    args = parse_arguments()
    if args.record != RECORDE_OPTIONS[0]:
        vids_path = os.path.join(args.output_dir, "vid")
        if not os.path.isdir(vids_path):
            os.mkdir(vids_path)
    else:
        vids_path = ""
    if args.agent == "human":
        outcomes = []
        for trial in range(args.num_of_trials):
            print("Stating human trial {}:".format(trial))
            if RECORDE_OPTIONS[2] or (RECORDE_OPTIONS[1] and trial % args.record_frequency == 0):
                current_record_path = os.path.join(vids_path, "human_record_trial_{}.mp4".format(trial))
            else:
                current_record_path = ""
            outcome = human_playing.run(env=gym_super_mario_bros.make(args.env),
                                        max_steps=args.steps_limit,
                                        standing_steps_limit=args.standing_steps_limit,
                                        allow_dying=args.allow_dying,
                                        record=current_record_path)
            outcomes.append(outcome)
        df = DataFrame(outcomes)
        df.to_csv(os.path.join(args.output_dir, "summary.csv"))

