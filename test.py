import argparse
import os
from datetime import datetime
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym import Wrapper
import gym.wrappers.monitor as monitor
from wrappers import *
import human_playing
import time
from population_manger import MarioBasicPopulationManger
from player import MarioPlayer
from pandas import DataFrame, read_csv
import json
from train import ACTION_SET

AGENTS = ['human', 'genetic', "random_nn"]
DEFAULT_ENVIRONMENT = 'SuperMarioBros-v0'
DEFAULT_STEP_LIMIT = 2000


def parse_arguments():
    parser = argparse.ArgumentParser(description="Script to train agents.")
    parser.add_argument("-agent", dest='agent', choices=AGENTS, default=AGENTS[1],
                        help="Chose kind of agent for testing.")
    parser.add_argument("-i", "-input_dir", dest="input_dir", help="Path for the train output data.")
    parser.add_argument("-o", "-output_dir", dest="output_dir", default="", help="Path for the output data.")
    parser.add_argument("-s", '-time_scale', "-steps_scale", dest='steps_limit', type=int,
                        help="The maximal frames for a trial.")
    parser.add_argument("-e", "-env", dest="env", help="The environment ID to play")
    parser.add_argument("-r", "-record", dest="record", action='store_true', default=False,
                        help="Record gameplay options")

    args = parser.parse_args()
    if args.agent != "human":
        if not args.input_dir:
            parser.error("You must give a input directory path.")
        elif not os.path.isdir(args.input_dir):
            parser.error("The input path is not valid. Check if the directory was removed.")
        elif not os.path.isfile(os.path.join(args.input_dir, "train_arguments.json")):
            parser.error("Missing train_arguments.json file from input directory.")
        elif args.agent == "genetic" and not os.path.isfile(os.path.join(args.input_dir, "manager_values.json")):
            parser.error("Missing manager_values.json file from input directory.")
    else:
        if not args.input_dir or not os.path.isdir(args.input_dir):
            # No input directory is given for human - must have values for env and steps_limit.
            if args.steps_limit is None:
                args.steps_limit = DEFAULT_STEP_LIMIT
            if args.env is None:
                args.env = DEFAULT_ENVIRONMENT
    if not args.output_dir:
        # make default output directory
        default_output_dir = "test_output_{agent}_{date}".format(agent=args.agent, date=datetime.now().strftime("%d-%m_%H-%M"))
        if not os.path.isdir(default_output_dir):
            os.mkdir(default_output_dir)
        args.output_dir = default_output_dir
    elif not os.path.isdir(args.output_dir):
        parser.error("The output path is not valid. Check if the directory was removed.")
    return args


def get_input_args(args):
    if args.input_dir is not None:
        with open(os.path.join(args.input_dir, "train_arguments.json"), "r") as arguments_file:
            args_dict = json.load(arguments_file)
    elif args.agent == "human":
        args_dict = {"agent": args.agent,
                     "steps_limit": args.steps_limit,
                     "allow_death": True,
                     "env": args.env}
    else:
        args_dict = dict()
    return args_dict


def write_summary(args, input_args, output_data_frame: DataFrame):
    with open(os.path.join(args.output_dir, "summary.txt"), 'w') as summary_file:
        summary_file.write("Summary for testing {agent} agent on {env}".format(agent=args.agent, env=args.env))
        if args.agent != "human":
            summary_file.write(" with {action_set} action set.\n".format(action_set=input_args["action_set"]))
            summary_file.write("The agent originally was trained with {num_of_loops} {g_or_t} with at most "
                               "{steps_limit} steps per game on {env}. ".format(num_of_loops=input_args["num_of_loops"],
                                g_or_t="generations" if args.agent == "genetic" else "trials",
                                steps_limit=input_args["steps_limit"],
                                env=input_args["env"]))
            summary_file.write("{standing_limit} - standing steps limit. {allow_dying} death".
                               format(standing_limit=input_args["standing_steps_limit"],
                                allow_dying="Didn't allowed" if input_args["allow_death"] else "allowed"))
            if args.agent == "genetic":
                summary_file.write(". {i_p} - initial population. {e_s} - Elite size. {p} - pick size. "
                                   "{r_m} - random members number".
                                   format(i_p=input_args["initial_population"], e_s=input_args["elite_size"],
                                          p=input_args["pick_size"], r_m=input_args["random_members"]))
        summary_file.write(".\n")
        summary_file.write("Test ran {steps_limit} steps limit.\n".format(steps_limit=args.steps_limit))
        if any(output_data_frame['finish_level']):
            summary_file.write("Agent successfully win the level.\n")
        else:
            summary_file.write("Agent failed to win the level.\n")
        if args.agent == "human" or args.agent == "random_nn":
            info = output_data_frame.iloc[0]
            summary_file.write("Agent ran for {steps} steps and the performance score is {performance_score}.\n"
                .format(steps=info["steps"], performance_score=info['performance_score']))
        elif args.agent == "genetic":
            summary_file.write("Average performance score of the elite is {avg_score}\n".
                               format(avg_score=output_data_frame.mean(axis=0)['performance_score']))
            best_result_index = output_data_frame['performance_score'].idxmax()
            info = output_data_frame.iloc[best_result_index]
            summary_file.write("Best performance: {best_index}.\n"
                               "ran for {steps} steps with performance score of {performance_score}. ".
                               format(best_index=info['index'], steps=info['steps'],
                                      performance_score=info['performance_score']))
        if info['finish_level']:
            summary_file.write("Finish level with size {size}.\n".format(size=info['finish_status']))
        summary_file.write("Finish with {deaths} deaths, game score of {score} and collected {coins} coins.\n".format(
            steps=info['steps'], deaths=info['deaths'], score=info['score'], coins=info['coins']))


def run_agent(player: MarioPlayer, env: Wrapper, record: bool, vids_path: str, index):
    if record:
        rec_output_path = os.path.join(vids_path, "{name}.mp4".
                                       format(name=index))
        rec = monitor.video_recorder.VideoRecorder(env, path=rec_output_path)

    state = env.reset()
    done = False

    for step in range(steps_limit):
        if done:
            break
        action = player.act(state)
        state, reward, done, info = env.step(action)
        env.render()
        if record:
            rec.capture_frame()
        player.update_info(info)
        player.update_reward(reward)
        if info['flag_get']:  # if got to the flag - run is ended.
            done = True

    if record:
        rec.close()
    player.calculate_fitness()
    outcome = player.get_run_info()
    outcome['index'] = index
    return outcome


if __name__ == "__main__":
    args = parse_arguments()
    input_args = get_input_args(args)

    if args.env is not None:
        if args.env != input_args['env']:
            print("Attention: The agent was trained {org_env} and will perform on {env}. "
                  "That can influence the results.".format(org_env=input_args['env'], env=args.env))
        env = gym_super_mario_bros.make(args.env)
    else:
        args.env = input_args['env']
        env = gym_super_mario_bros.make(input_args['env'])
    if args.steps_limit:
        if args.steps_limit != input_args['steps_limit']:
            print("Attention: The agent was trained with a limit of {org_steps_limit} steps and will "
                  "perform with a limit of {steps_limit}. That can influence the results.".
                  format(org_steps_limit=input_args['steps_limit'], steps_limit=args.steps_limit))
        steps_limit = args.steps_limit
    else:
        args.steps_limit = input_args['steps_limit']
        steps_limit = input_args['steps_limit']
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
        outcome = human_playing.run(env=env,
                                    max_steps=steps_limit,
                                    standing_steps_limit=steps_limit,
                                    allow_dying=True,
                                    record=current_record_path)
        df = DataFrame([outcome])

    elif args.agent == "genetic":
        actions = ACTION_SET[input_args['action_set']]
        population_manager = MarioBasicPopulationManger()
        population_manager.load_population(args.input_dir)
        elite = population_manager.get_elite()
        env = JoypadSpace(env, actions)
        env = WarpFrame(env)
        env = FrameStack(env, 4)
        outcomes = []
        print("Stating genetic agent test:")
        t = time.time()
        for index, member in enumerate(elite):
            print("running member {index} of the elite.".format(index=index))
            player = MarioPlayer(len(actions), member.genes)
            outcome = run_agent(player, env, args.record, vids_path, member.get_name())
            outcomes.append(outcome)
        env.close()
        print(f"finish test in {time.time() - t}")
        df = DataFrame(outcomes)

    elif args.agent == "random_nn":
        actions = ACTION_SET[input_args['action_set']]
        env = JoypadSpace(env, actions)
        env = WarpFrame(env)
        env = FrameStack(env, 4)
        df = read_csv(os.path.join(args.input_dir, "random_output.csv"))
        best_result_index = df['performance_score'].idxmax()
        player = MarioPlayer(len(actions))
        player.load_player(os.path.join(args.input_dir, f"trail_{best_result_index}_info.json"))
        print("Stating random_nn agent test:")
        t = time.time()
        outcome = run_agent(player, env, args.record, vids_path,best_result_index)
        env.close()
        print(f"finish test in {time.time() - t}")
        t = time.time()
        df = DataFrame([outcome])

    df.to_csv(os.path.join(args.output_dir, "output.csv"))
    write_summary(args, input_args, df)

