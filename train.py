import argparse
import os
from datetime import datetime
from pandas import DataFrame
import gym_super_mario_bros
from gym_super_mario_bros import actions
from mario_evolution import GeneticMario
import human_playing


DEFAULT_ENVIRONMENT = 'SuperMarioBros-v0'
DEFAULT_STEP_LIMIT = 2000
DEFAULT_NO_ADVANCE_STEP_LIMIT = 100
AGENTS = ['human', 'genetic']
ACTION_SET = {"right_only": actions.RIGHT_ONLY, "simple": actions.SIMPLE_MOVEMENT, "complex": actions.COMPLEX_MOVEMENT}
DEFAULT_ACTION_SET = "simple"
TRIALS = 10
INITIAL_POP = 150
RECORDE_OPTIONS = ["none", 'some', 'all']
DEFAULT_RECORDE_FREQUENCY = 100

def parse_arguments():
    parser = argparse.ArgumentParser(description="Script to train agents.")
    parser.add_argument("-agent", dest='agent', choices=AGENTS, default=AGENTS[0],
                        help="Chose kind of agent for training.")
    parser.add_argument("-o", "-output_dir", dest="output_dir", default="", help="Path for the output data.")
    parser.add_argument("-n", "-num_of_trials", "-g", "-generations", dest='loop_times', type=int, default=TRIALS,
                        help="Number of trials or generations to run.")
    parser.add_argument("-initial_population", "-i_p", dest="initial_population", type=int, default=INITIAL_POP,
                        help="The size of initial population for genetic agent.")
    parser.add_argument("-s", '-time_scale', "-steps_scale", dest='steps_limit', default=DEFAULT_STEP_LIMIT, type=int,
                        help="The maximal frames for a trial.")
    parser.add_argument("-a", "-action_set", dest="action_set", choices=ACTION_SET.keys(), default=DEFAULT_ACTION_SET,
                        help="The set of action the agent can use. Isn't relevant for human agent.")
    parser.add_argument("-no_action_limit", "-no_advance_limit", "-standing_limit", "-no_progress_limit",
                        dest='standing_steps_limit', default=DEFAULT_NO_ADVANCE_STEP_LIMIT, help="Limit the number of"
                        " steps the agent allow not to change the x position.")
    parser.add_argument("-d", "-allow_death", "-allow_dying", dest="allow_dying", action='store_false',
                        help="Allow agent to die in a trail.")
    parser.add_argument("-e", "-env", dest="env", default=DEFAULT_ENVIRONMENT, help="The environment ID to play")
    parser.add_argument("-r", "-record", dest="record", choices=RECORDE_OPTIONS, default=RECORDE_OPTIONS[0],
                        help="Record gameplay options")
    parser.add_argument("-rf", "-record_frequency", dest="record_frequency", default=DEFAULT_RECORDE_FREQUENCY, type=int,
                        help="The frequency of trails that will be recoded if 'some' was chosen for record option.")


    args = parser.parse_args()
    if args.loop_times < 0:
        parser.error("number of trials\generations must to be positive.")
    if args.steps_limit < 1:
        parser.error("steps limit must to be positive.")
    if args.initial_population < 1:
        parser.error("Initial population size have to be positive.")
    if args.standing_steps_limit < 1:
        parser.error("The limit for standing must be positive.")
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


def write_summary(args, output_data_frame: DataFrame):
    with open(os.path.join(args.output_dir, "summary.txt"), 'w') as summary_file:
        summary_file.write("Summary for training {agent} agent on {env}".format(agent=args.agent, env=args.env))
        if args.agent != "human":
            summary_file.write(" with {action_set} action set".format(action_set=args.action_set))
        summary_file.write(".\n")
        summary_file.write("Ran for {num_of_loops} {g_or_t} with at most {steps_limit} steps per game.\n".format(
            num_of_loops=args.loop_times, g_or_t="generations" if args.agent == "genetic" else "trials",
            steps_limit=args.steps_limit))
        summary_file.write("Limit on no changing the x position is: {standing_limit}\n"
                           "{allow_dying} allow player to die.\n".format(standing_limit=args.standing_steps_limit,
                                                                allow_dying="Didn't" if args.allow_dying else "Did"))
        if args.agent == "genetic":
            summary_file.write("Initial population is: {i_p}\n".format(i_p=args.inital_poplation))
        if any(output_data_frame['finish_level']):
            summary_file.write("Agent successfully win the level in some games.\n")
        else:
            summary_file.write("Agent failed to win any games.\n")
        # add best result


if __name__ == "__main__":
    args = parse_arguments()
    if args.record != RECORDE_OPTIONS[0]:
        vids_path = os.path.join(args.output_dir, "vid")
        if not os.path.isdir(vids_path):
            os.mkdir(vids_path)
    else:
        vids_path = ""
    if args.agent == "human":
        outcomes = []
        for trial in range(args.loop_times):
            print("Stating human trial {}:".format(trial))
            if args.record == RECORDE_OPTIONS[2] or (args.record == RECORDE_OPTIONS[1] and trial % args.record_frequency == 0):
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
        df.to_csv(os.path.join(args.output_dir, "output.csv"))
    elif args.agent == "genetic":
        model = GeneticMario(actions=ACTION_SET[args.action_set],
                             generations=args.loop_times,
                             initial_pop=args.inital_poplation,
                             steps_scale=args.steps_limit,
                             allow_death=args.allow_dying,
                             standing_steps_limit=args.standing_steps_limit,
                             output_dir=args.output_dir,
                             record=args.record == RECORDE_OPTIONS[1] or args.record == RECORDE_OPTIONS[2])
        model.run()

