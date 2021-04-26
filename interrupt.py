import sys
import signal
import pdb

def ctrlc_handler(sig, frame):
    pdb.Pdb().set_trace(frame)

signal.signal(signal.SIGINT, ctrlc_handler)

