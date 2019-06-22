from config import args
from data import Data
from model import Model
from loss import Loss
from opti import Opti
from utils.Metric import CMC_MAP
from engines.engine import create_supervised_evaluator,create_supervised_trainer,mean,list_sum
from ignite.engine import Events
from tqdm import tqdm
from utils.saver import Saver
from tensorboardX import SummaryWriter


data=Data(args)
model=Model(args)
loss_fn=Loss(args)
opti, lr_s=Opti(args,model)
saver=Saver(args,model,opti)
writer=SummaryWriter(log_dir=saver.save_dir)
print(saver.save_dir)

trainer=create_supervised_trainer(model, opti, loss_fn)
tester=create_supervised_evaluator(model,
                                   metrics={'CMC_MAP':CMC_MAP(data,
                                                              L2=True,
                                                              re_ranking=False)},
                                   flip_=True)



@trainer.on(Events.STARTED)
def started(engine):
    saver.load()
    print(args.loss)

    engine.state.epoch_bar=tqdm(total=engine.state.max_epochs,desc='')


@trainer.on(Events.EPOCH_STARTED)
def epoch_started(engine):
    lrs=lr_s(engine.state.epoch-1)
    writer.add_scalars('training/lrs',lrs,engine.state.epoch)
    engine.state.lr=lrs
    engine.state.batch_bar=tqdm(total=len(data.dataloader['train']),desc='',leave=False)
    engine.state.batch_desc=''

    engine.state.batch_losses = {k: 0 for k in loss_fn.keys()}
    if loss_fn.__len__() > 1:
        engine.state.batch_losses['Total_Loss'] = 0

    engine.state.batch_losses2 = {k:[] for k in loss_fn.keys()}


@trainer.on(Events.ITERATION_COMPLETED)
def iteration_completed(engine):

    for k in loss_fn.keys():
        engine.state.batch_losses[k] += mean(engine.state.output[k])
        if loss_fn.__len__()>1:
            engine.state.batch_losses['Total_Loss'] += mean(engine.state.output[k]) * loss_fn[k]['weight']

    batch_desc = ['[{}={:.4e}]'.format(k, engine.state.lr[k]) for k in engine.state.lr.keys()]
    engine.state.batch_desc = ''
    for s in batch_desc:
        engine.state.batch_desc += s
    if loss_fn.__len__() > 1:
        engine.state.batch_desc += '[Total_Loss={:.4f}]'.format(engine.state.batch_losses['Total_Loss'])

    for k in loss_fn.keys():
        engine.state.batch_desc += '[{}={:.4f}]'.format(k, mean(engine.state.output[k]))
        engine.state.batch_losses2[k] = engine.state.output[k] if len(engine.state.batch_losses2[k]) == 0 else list_sum(
            engine.state.batch_losses2[k], engine.state.output[k])

    engine.state.batch_bar.desc=engine.state.batch_desc
    engine.state.batch_bar.update(1)

@trainer.on(Events.EPOCH_COMPLETED)
def epoch_completed(engine):
    engine.state.batch_bar.close()
    desc='[Train Epoch {}]'.format(engine.state.epoch)
    for k in engine.state.batch_losses.keys():
        desc+='[{}={:.4f}]'.format(k,engine.state.batch_losses[k]/len(data.dataloader['train']))
    engine.state.epoch_bar.write(desc)
    writer.add_scalars('training/loss',
                       {k:engine.state.batch_losses[k]/len(data.dataloader['train']) for k in engine.state.batch_losses.keys()},
                       global_step=engine.state.epoch)

    for k in loss_fn.keys():
        if len(engine.state.batch_losses2[k])>1:
            scalars={'{}_{}'.format(k,i):engine.state.batch_losses2[k][i]/len(data.dataloader['train']) for i in range(len(engine.state.batch_losses2[k]))}
            scalars['mean']=engine.state.batch_losses[k]/len(data.dataloader['train'])
            writer.add_scalars('training/{}_s'.format(k),scalars,global_step=engine.state.epoch)

    engine.state.epoch_bar.update(1)

    if engine.state.epoch % args.test_every==0:
        tester.run(data.dataloader['test_query'])
        writer.add_scalars('testing/metrics',
                           tester.state.metrics['CMC_MAP'],
                           global_step=engine.state.epoch)
        engine.state.epoch_bar.write('[Test Epoch {}][mAP={:.4f}][Rank-1={:.4f}][Rank-3={:.4f}][Rank-5={:.4f}][Rank-10={:.4f}]'.format(engine.state.epoch,
                                                                                                                                       tester.state.metrics['CMC_MAP'][
                                                                                                                                            'mAP'],
                                                                                                                                       tester.state.metrics['CMC_MAP'][
                                                                                                                                            'Rank-1'],
                                                                                                                                       tester.state.metrics['CMC_MAP'][
                                                                                                                                            'Rank-3'],
                                                                                                                                       tester.state.metrics['CMC_MAP'][
                                                                                                                                            'Rank-5'],
                                                                                                                                       tester.state.metrics['CMC_MAP'][
                                                                                                                                            'Rank-10']))
        saver.save(engine.state.epoch)


@tester.on(Events.EPOCH_STARTED)
def epoch_started2(engine):
    engine.state.batch_bar=tqdm(total=len(data.dataloader['test_query']))


@tester.on(Events.ITERATION_COMPLETED)
def iteration_completed(engine):
    engine.state.batch_bar.update(1)





trainer.run(data.dataloader['train'],max_epochs=args.epochs)