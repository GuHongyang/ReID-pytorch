from abc import ABCMeta, abstractmethod
from ignite._six import with_metaclass
from ignite.engine import Events
from ignite.metrics import Metric
import torch
import numpy as np
from .metrics import cmc_map
from .re_ranking import re_ranking

def toNP(tensor):
    return tensor.numpy()

def toCPU(tensor):
    return tensor.detach().cpu()

class CMC_MAP(Metric):
    """
    Base class for all Metrics.

    Args:
        output_transform (callable, optional): a callable that is used to transform the
            :class:`ignite.engine.Engine`'s `process_function`'s output into the
            form expected by the metric. This can be useful if, for example, you have a multi-output model and
            you want to compute the metric with respect to one of the outputs.

    """

    def __init__(self,data, L2=True, re_ranking=False, output_transform=lambda x: x):
        self._output_transform = output_transform
        super(CMC_MAP,self).__init__(output_transform)
        self.query_data=data.dataset['query']
        self.test_data=data.dataset['test']
        self.L2=L2
        self.re_ranking=re_ranking
        self.reset()

    def reset(self):
        """
        Resets the metric to to it's initial state.

        This is called at the start of each epoch.
        """
        self.fa=np.array([])

    def update(self, output):
        """
        Updates the metric's state using the passed batch output.

        This is called once for each batch.

        Args:
            output: the is the output from the engine's process function
        """
        ff, _=output
        ff = toCPU(ff)
        if self.L2:
            fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
            ff = ff.div(fnorm.expand_as(ff))
        self.fa=np.vstack((self.fa, toNP(ff))) if self.fa.shape[0] else toNP(ff)


    def compute(self):
        """
        Computes the metric based on it's accumulated state.

        This is called at the end of each epoch.

        Returns:
            Any: the actual quantity of interest

        Raises:
            NotComputableError: raised when the metric cannot be computed
        """
        gf=self.fa[:self.test_data.__len__(),:]
        qf=self.fa[self.test_data.__len__():,:]

        if self.re_ranking:
            q_g_dist = np.dot(qf, np.transpose(gf))
            q_q_dist = np.dot(qf, np.transpose(qf))
            g_g_dist = np.dot(gf, np.transpose(gf))
            dist = re_ranking(q_g_dist, q_q_dist, g_g_dist)
        else:
            dist1 = np.sum(qf ** 2, 1, keepdims=True)
            dist2 = np.sum(gf.T * gf.T, 0, keepdims=True)
            dist = dist1 + dist2 - 2 * np.dot(qf, gf.T)

        r, m_ap = cmc_map(dist, self.query_data.ids, self.test_data.ids, self.query_data.cameras,
                          self.test_data.cameras,
                          separate_camera_set=False,
                          single_gallery_shot=False,
                          first_match_break=True)

        return {'mAP': m_ap,
                'Rank-1': r[0],
                'Rank-3': r[2],
                'Rank-5': r[4],
                'Rank-10':r[9]}
