# Coding by BAEK(01153450@hyundai-autoever.com)

from utils.types import Dictconfigs
from architecture.engine.test_loop import TesterBase
from . import TESTER_REGISTRY

from architecture.modeling.build import build_model
from architecture.solver.build import build_criterion, build_optimizer
from architecture.evaluation.build import build_evaluator
from architecture.data.build import build_dataloader


@TESTER_REGISTRY.register()
class DefaultTrainer(TesterBase):
    '''
    Default-class for trainer.
    '''
    def __init__(self, configs: Dictconfigs):
        super(DefaultTrainer, self).__init__()

        # build for tester.
        self.test_loader = self.build_dataloader(configs, train=False)


        self.model = build_model(configs)
        self.criterion = build_criterion(configs)
        self.optimizer = build_optimizer(configs, self.model)

    def before_test(self):
        # todo : add checkpointer
        pass

    def before_step(self):
        pass

    def run_step(self):
        self.model.test()

        # todo : add max-iterations > 기준을 epoch 으로 할건지 iter 로 할건지
        for data in self.test_loader:

            # y zero the parameter gradients.
            self.optimizer.zero_grad()

            # forward + backward + optimize
            ds, _ = self.model()
            loss2, loss = self.model

    def after_step(self):
        # todo : add save weights
        # todo : add logger
        pass

    def after_train(self):
        # todo : add logger
        pass

    @classmethod
    def build_model(cls):
        # It now calls :func: 'architecture.modeling.build_model'.
        return build_model()

    @classmethod
    def build_criterion(cls):
        # It now calls :func: 'architecture.solver.build_criterion'.
        return build_criterion()

    @classmethod
    def build_optimizer(cls):
        # It now calls :func: 'architecture.solver.build_optimizer'.
        return build_optimizer()

    @classmethod
    def build_evaluator(cls):
        # It now calls :func: 'architecture.evaluation.build_evaluator'.
        return build_evaluator()

    @classmethod
    def build_dataloader(cls, configs: Dictconfigs, train: bool):
        # It now calls :func: 'architecture.data.build_dataloader'.
        return build_dataloader(configs, train)
