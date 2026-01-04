import json
import os
import threading
import pyximport; pyximport.install()

from torch import multiprocessing as mp

from alphazero import CALLABLE_PREFIX
from alphazero.Coach import Coach, get_args
from alphazero.NNetWrapper import NNetWrapper as nn
from alphazero.envs.crazyhouse.crazyhouse import Game
from alphazero.GenericPlayers import RawMCTSPlayer
from alphazero.utils import dotdict

from torch.optim import *
from torch.optim.lr_scheduler import *
from alphazero.GenericPlayers import *
from alphazero.utils import default_temp_scaling, const_temp_scaling

ARGS_DIR = "alphazero/envs/crazyhouse/args"

class TrainingManager:

    def __init__(self, train_args_name):
        self.coach = None
        self.args = dotdict()
        self.train_args_name = train_args_name

    def start_train(self):

        try:
            self._load_args_into(self.args, ARGS_DIR + "/" + self.train_args_name)
        except (IOError, OSError) as e:
            print('Failed to load the selected args: ' + str(e), self)
            return

        nnet = nn(Game, self.args)
        self.coach = Coach(Game, nnet, self.args)
        self.train_thread = threading.Thread(target=self.coach.learn, daemon=True)
        self.train_thread.start()

    def stop_train(self):
        self.coach.stop_train.set()
        if self.coach.arena:
            self.coach.arena.stop_event.set()

        self.train_thread.join()


        self.args.selfPlayModelIter = self.coach.self_play_iter
        try:
            self._save_args_from(self.args, ARGS_DIR + "/" + self.train_args_name)
        except (IOError, OSError) as e:
            print('Unable to save args after training was stopped: ' + str(e))

        self.coach = None

    def pause_train(self):
        self.coach.pause_train.set()
        if self.coach.arena:
            self.coach.arena.pause_event.set()

    def _load_args_into(self, args: dotdict, filepath, clear_args=True):
        if clear_args: args.clear()

        load_args = dotdict()
        with open(filepath) as f:
            load_args.update(json.load(f))

        args.update(self._parse_str_args(load_args))  # TODO: TypeError: 'NoneType' object is not iterable
        return args

    def _save_args_from(self, args: dotdict, filepath, replace=True):
        if not replace and os.path.exists(filepath): return

        save_args = self._get_str_args(args)
        with open(filepath, 'w') as f:
            json.dump(save_args, f)

        return save_args
    
    def _parse_str_args(self, args: dotdict):
        new_args = dotdict()
        for k, v in args.items():
            if isinstance(v, str) and CALLABLE_PREFIX in v:
                try:
                    v = eval(v.replace(CALLABLE_PREFIX, ''))
                except Exception as e:
                    print('Failed to parse argument file: ' + str(e))
                    return

            elif isinstance(v, dict):
                v = dotdict(v)

            new_args.update({k: v})

        return new_args
    
    @staticmethod
    def _get_str_args(args: dotdict):
        save_args = dict()
        for k, v in args.items():
            if callable(v):
                v = CALLABLE_PREFIX + v.__name__
            save_args.update({k: v})

        return save_args


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python train.py <train_args_name>")
        sys.exit(1)
    
    tm = TrainingManager(sys.argv[1])
    
    print("Training control commands:")
    print("  s - start training")
    print("  p - pause training")
    print("  t - stop training")
    print("  q - quit")
    print("Enter command and press Enter:")
    
    while True:
        try:
            command = input("> ").strip().lower()
            
            if command == 's':
                if tm.coach is None:
                    print("Starting training...")
                    tm.start_train()
                else:
                    print("Training is already running!")
                    
            elif command == 't':
                if tm.coach is not None:
                    print("Stopping training...")
                    tm.stop_train()
                else:
                    print("Training is not running!")
                    
            elif command == 'q':
                if tm.coach is not None:
                    print("Stopping training before quit...")
                    tm.stop_train()
                print("Quitting...")
                break
                
            else:
                print(f"Unknown command: '{command}'. Available: s(start), p(pause), t(stop), q(quit)")
                
        except KeyboardInterrupt:
            print("\nInterrupted by user")
            if tm.coach is not None:
                print("Stopping training...")
                tm.stop_train()
            break
        except Exception as e:
            print(f"Error: {e}")

