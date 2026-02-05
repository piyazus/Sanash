
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.optimize import linear_sum_assignment

class P2PLoss(nn.Module):
    def __init__(self, w_cls=1.0, w_reg=1.0):
        super(P2PLoss, self).__init__()
        self.w_cls = w_cls
        self.w_reg = w_reg
        self.matcher = HungarianMatcher()
        self.loss_cls = nn.BCEWithLogitsLoss()
        self.loss_reg = nn.MSELoss() # Or SmoothL1
        
    def forward(self, pred_points, pred_logits, targets):
        # pred_points: B x N x 2
        # pred_logits: B x N x 1
        # targets: List of dicts {'points': tensor}
        
        indices = self.matcher(pred_points, pred_logits, targets)
        
        loss_c = 0.0
        loss_r = 0.0
        
        for b, (idx_pred, idx_gt) in enumerate(indices):
            # Targets for this batch
            gt_points = targets[b]['points'].to(pred_points.device)
            
            # Regression Loss (Matched points)
            if len(idx_pred) > 0:
                pred_pts_b = pred_points[b][idx_pred]
                gt_pts_b = gt_points[idx_gt]
                loss_r += self.loss_reg(pred_pts_b, gt_pts_b)
            
            # Classification Loss
            # Matched are 1, Unmatched are 0
            # pred_logits[b] shape N x 1
            target_cls = torch.zeros_like(pred_logits[b])
            if len(idx_pred) > 0:
                target_cls[idx_pred] = 1.0
            
            loss_c += self.loss_cls(pred_logits[b], target_cls)
            
        loss_c /= len(targets)
        loss_r /= len(targets)
        
        return self.w_cls * loss_c + self.w_reg * loss_r

class HungarianMatcher(nn.Module):
    def __init__(self):
        super(HungarianMatcher, self).__init__()
        
    @torch.no_grad()
    def forward(self, pred_points, pred_logits, targets):
        # Compute Cost Matrix
        # For P2PNet, cost is usually: - prob + L2_dist
        
        indices = []
        for b in range(len(targets)):
            out_prob = pred_logits[b].sigmoid().flatten() # N
            out_pts = pred_points[b] # N x 2
            gt_pts = targets[b]['points'].to(out_pts.device) # M x 2
            
            if len(gt_pts) == 0:
                indices.append(([], []))
                continue
                
            # Cost Matrix: N x M
            # Distance cost
            # Expand for broadcasting
            # N x 1 x 2 - 1 x M x 2
            dist = torch.norm(out_pts.unsqueeze(1) - gt_pts.unsqueeze(0), p=2, dim=2) # N x M
            
            # Prob cost (inverse prob => 1 - prob)
            # Or just - prob.
            prob_cost = - out_prob.unsqueeze(1).repeat(1, len(gt_pts))
            
            C = 0.05 * dist + 1.0 * prob_cost # weights need tuning
            
            C = C.cpu().numpy()
            
            # Hungarian matching
            # We want to match every GT point to one Prediction
            # But not every Prediction needs a GT.
            # scipy linear_sum_assignment finds min cost matching
            # It matches min(N, M) pairs.
            # In P2PNet, N (proposals) >> M (GT).
            # So every GT gets a match.
            
            row_ind, col_ind = linear_sum_assignment(C)
            
            # row_ind are indices in Prediction (N)
            # col_ind are indices in GT (M)
            
            indices.append((row_ind, col_ind))
            
        return indices
