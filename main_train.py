import os
import time
from tensorboardX import SummaryWriter
from data.datasets import create_dataloader
from earlystop import EarlyStopping
from options.train_options import TrainOptions

from networks.qbatrainer_cls import Trainer as qbatrainer_cls
from networks.qbatrainer_od import Trainer as qbatrainer_od
from networks.qbatrainer_dfd import Trainer as qbatrainer_dfd
from networks.trainer_cls import Trainer as trainer_cls
from networks.trainer_od import Trainer as trainer_od
from networks.trainer_dfd import Trainer as trainer_dfd

from evaluation.validate_cls import validate as validate_cls
from evaluation.validate_cls import validate as validate_od


def get_trainer(opt):
    if opt.is_QBATrain:
        if opt.task == "CLS":
            return qbatrainer_cls(opt)
        elif opt.task == "OD":
            return qbatrainer_od(opt)
        elif opt.task == "DFD":
            return qbatrainer_dfd(opt)
    else:
        if opt.task == "CLS":
            return trainer_cls(opt)
        elif opt.task == "OD":
            return trainer_od(opt)
        elif opt.task == "DFD":
           return trainer_dfd(opt)


if __name__ == '__main__':
    opt = TrainOptions().parse()
    val_opt = TrainOptions().parse(print_options=False)

    data_loader, _ = create_dataloader(opt)

    dataset_size = len(data_loader)
    print('#training images = %d' % dataset_size)

    train_writer = SummaryWriter(os.path.join(opt.checkpoints_dir, "train"))
    valid_writer_1 = SummaryWriter(os.path.join(opt.checkpoints_dir, "valid_1"))
    valid_writer_2 = SummaryWriter(os.path.join(opt.checkpoints_dir, "valid_2"))

    model = get_trainer(opt)

    early_stopping = EarlyStopping(patience=opt.earlystop_epoch, delta=-0.001, verbose=True)
    for epoch in range(opt.niter):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0

        for i, data in enumerate(data_loader):
            model.total_steps += 1
            epoch_iter += opt.batch_size

            model.set_input(data)
            model.optimize_parameters()

            if model.total_steps % opt.loss_freq == 0:
                print("Train loss: {} at step: {}".format(model.loss, model.total_steps))
                train_writer.add_scalar('loss', model.loss, model.total_steps)

            if model.total_steps % opt.save_latest_freq == 0:
                print('saving the latest model %s (epoch %d, model.total_steps %d)' %
                      (opt.name, epoch, model.total_steps))
                model.save_networks('latest')

        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' %
                  (epoch, model.total_steps))
            model.save_networks('latest')
            model.save_networks(epoch)

        # Validation
        model.eval()
        if opt.task == "OD":
            map_score, avg_f1_score = validate_od(model, val_opt)
            valid_writer_1.add_scalar('map_score', map_score, model.total_steps)
            valid_writer_2.add_scalar('avg_f1_score', avg_f1_score, model.total_steps)

            print("(Val @ epoch {}) map_score: {}; avg_f1_score: {}".format(epoch, map_score, avg_f1_score))

        elif opt.is_QBATrain:
            acc, target_acc = validate_cls(model.model, val_opt)
            valid_writer_1.add_scalar('accuracy', acc, model.total_steps)
            valid_writer_2.add_scalar('target_accuracy', target_acc, model.total_steps)
            print("(Val @ epoch {}) acc: {}, target_acc: {}".format(epoch, acc, target_acc))

        else:
            acc = validate_cls(model.model, val_opt)
            valid_writer_1.add_scalar('accuracy', acc, model.total_steps)
            print("(Val @ epoch {}) acc: {}".format(epoch, acc))

        if opt.task == "OD":
            early_stopping(map_score)
        else:
            early_stopping(acc, model)

        if early_stopping.early_stop:
            cont_train = model.adjust_learning_rate()
            if cont_train:
                print("Learning rate dropped by 10, continue training...")
                early_stopping = EarlyStopping(patience=opt.earlystop_epoch, delta=-0.002, verbose=True)
            else:
                print("Early stopping.")
                break
        model.train()