import os
import torch
from torchvision import transforms
from termcolor import colored


os.system("color")
def breaker(num=50, char="*") -> None:
    print(colored("\n" + num*char + "\n", "magenta"))


def myprint(text: str, color: str) -> None:
    print(colored(text, color))


SEED = 0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRANSFORM = transforms.Compose([transforms.ToTensor(),])
DATA_PATH = "./Data"
CHECKPOINT_PATH = "./Checkpoints"
if not os.path.exists(CHECKPOINT_PATH):
    os.makedirs(CHECKPOINT_PATH)
