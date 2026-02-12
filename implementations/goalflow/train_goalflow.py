import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from models.goal_point_scorer import GoalPointScorer
from models.goal_flow_matcher import GoalFlowMatcher
from models.trajectory_selector import TrajectorySelector

class GoalFlowTrain():
    """ 
    Train of Goal Flow
    
    """
    def __init__(self,
                 model):
        self.goal_point_scorer = 

    def train(self, 
              epoch: int):
        