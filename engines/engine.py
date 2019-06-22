import torch
from ignite.engine.engine import Engine, State, Events
from ignite._utils import convert_tensor



def mean(tensors):
    s=0
    for t in tensors:
        s+=t
    return s/len(tensors)

def list_sum(l1,l2):
    return [l1[i]+l2[i] for i in range(len(l1))]


def _prepare_batch(batch, device=None, non_blocking=False):
    """Prepare batch for training: pass to a device with options

    """
    x, y = batch
    return (convert_tensor(x, device=device, non_blocking=non_blocking),
            convert_tensor(y, device=device, non_blocking=non_blocking))

def create_supervised_trainer(model, optimizer, loss_fn, device='cuda'):
    """

    :param model:
    :param optimizer:
    :param lr_scheduler:
    :param loss_fn: {'CE_Loss':{
            'function':nn.CrossEntropyLoss(),
            'weight':self.args.weight_softmax,
        }}
    :param device:
    :return:
    """

    def _update(engine, batch):
        model.train()
        model.get_module().mode='train'
        optimizer.zero_grad()
        x, y = _prepare_batch(batch, device=device)
        y_pred = model(x)
        output={}
        loss=0
        for k in loss_fn.keys():
            if k == 'Center_Loss':
                output[k] = [loss_fn[k]['function'](y_pred[k][i], y, model.get_module().centers[i]) for i in range(len(y_pred[k]))]
            else:
                output[k]=[loss_fn[k]['function'](y_pred[k][i], y) for i in range(len(y_pred[k]))]
            loss+=mean(output[k])*loss_fn[k]['weight']
        loss.backward()
        optimizer.step()

        return output
    engine=Engine(_update)
    return engine




def create_supervised_evaluator(model, metrics={}, flip_=True,device='cuda'):
    def fliphor(x):
        inv_idx = torch.arange(x.size(3) - 1, -1, -1).long().to(device)  # N x C x H x W
        return x.index_select(3, inv_idx)

    def _inference(engine, batch):
        model.eval()
        model.get_module().mode='test'
        with torch.no_grad():
            x, y = _prepare_batch(batch, device=device)
            y_pred = model(x)
            if flip_:
                y_pred+=model(fliphor(x))
            return y_pred, y

    engine = Engine(_inference)

    for name, metric in metrics.items():
        metric.attach(engine, name)

    return engine

