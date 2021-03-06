from __future__ import print_function
import argparse
import random
import numpy as np
import torch
import torch.optim as optim
import sys
sys.path.append('./auxiliary/')
from dataset import *
from model import *
from utils import *
from ply import *
import os
import json
import time, datetime
import visdom
from tensorboardX import SummaryWriter

best_val_loss = 10

# =============PARAMETERS======================================== #
parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=32, help='input batch size')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=12)
parser.add_argument('--nepoch', type=int, default=400, help='number of epochs to train for')
# parser.add_argument('--model_preTrained_AE', type=str, default = 'trained_models/ae_atlasnet_25.pth',  help='model path')
parser.add_argument('--model_preTrained_AE', type=str, default = None,  help='model path')
parser.add_argument('--model', type=str, default = '',  help='model path')
parser.add_argument('--num_points', type=int, default = 2048,  help='number of points')
parser.add_argument('--nb_primitives', type=int, default = 25,  help='number of primitives')
parser.add_argument('--env', type=str, default ="SVR_AtlasNet"   ,  help='visdom env')
parser.add_argument('--fix_decoder', type=bool, default = True   ,  help='if set to True, on the the resnet encoder is trained')
parser.add_argument('--accelerated_chamfer', type=int, default =1   ,  help='use custom build accelarated chamfer')
parser.add_argument('--visdom_port', type=int, default = 48481,  help='Port use for visdom server.')
parser.add_argument('--class_choice', type=str, default = None, nargs='+', help='Class choice')
opt = parser.parse_args()
print (opt)
# ========================================================== #

# =============DEFINE CHAMFER LOSS======================================== #
if opt.accelerated_chamfer:
    sys.path.append("./extension/")
    import dist_chamfer as ext
    distChamfer =  ext.chamferDist()

else:
    def pairwise_dist(x, y):
        xx, yy, zz = torch.mm(x,x.t()), torch.mm(y,y.t()), torch.mm(x, y.t())
        rx = (xx.diag().unsqueeze(0).expand_as(xx))
        ry = (yy.diag().unsqueeze(0).expand_as(yy))
        P = (rx.t() + ry - 2*zz)
        return P


    def NN_loss(x, y, dim=0):
        dist = pairwise_dist(x, y)
        values, indices = dist.min(dim=dim)
        return values.mean()


    def distChamfer(a,b):
        x,y = a,b
        bs, num_points, points_dim = x.size()
        xx = torch.bmm(x, x.transpose(2,1))
        yy = torch.bmm(y, y.transpose(2,1))
        zz = torch.bmm(x, y.transpose(2,1))
        diag_ind = torch.arange(0, num_points).type(torch.cuda.LongTensor)
        rx = xx[:, diag_ind, diag_ind].unsqueeze(1).expand_as(xx)
        ry = yy[:, diag_ind, diag_ind].unsqueeze(1).expand_as(yy)
        P = (rx.transpose(2,1) + ry - 2*zz)
        return P.min(1)[0], P.min(2)[0]
# ========================================================== #

# =============DEFINE stuff for logs ======================================== #
#Launch visdom for visualization

vis = visdom.Visdom(port = opt.visdom_port, env=opt.env)
import time
# now = datetime.datetime.now()
now = time.time()
# save_path = now.isoformat()
save_path = opt.env + "_%d"%int(now)
log_dir = os.path.join("runs", "SVR", save_path)
dir_name =  os.path.join('log', "SVR", save_path)
os.makedirs(dir_name, exist_ok=True)
logname = os.path.join(dir_name, 'log.txt')
writer = SummaryWriter(log_dir=log_dir)

blue = lambda x:'\033[94m' + x + '\033[0m'

opt.manualSeed = random.randint(1, 10000) # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

# ========================================================== #


# ===================CREATE DATASET================================= #
#Create train/test dataloader on new views and test dataset on new models
dataset = ShapeNet( SVR=True, normal = False, class_choice = opt.class_choice, train=True, npoints = opt.num_points )
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                          shuffle=True, num_workers=int(opt.workers))

dataset_test = ShapeNet( SVR=True, normal = False, class_choice = opt.class_choice, train=False, npoints = opt.num_points)
dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=opt.batchSize,
                                          shuffle=False, num_workers=int(opt.workers))

dataset_test_view = ShapeNet( SVR=True, normal = False, class_choice = opt.class_choice, train=True, gen_view=True, npoints = opt.num_points)
dataloader_test_view = torch.utils.data.DataLoader(dataset_test, batch_size=opt.batchSize,
                                          shuffle=False, num_workers=int(opt.workers))


print('training set', len(dataset.datapath))
print('testing set', len(dataset_test.datapath))
len_dataset = len(dataset)
# ========================================================== #

# ===================CREATE network================================= #
# LOAD Pretrained autoencoder and check its performance on testing set
network_preTrained_autoencoder = AE_AtlasNet(num_points = opt.num_points, nb_primitives = opt.nb_primitives)
network_preTrained_autoencoder.cuda()
if opt.model_preTrained_AE is not None:
    network_preTrained_autoencoder.load_state_dict(torch.load(opt.model_preTrained_AE ))
val_loss = AverageValueMeter()
val_loss.reset()
network_preTrained_autoencoder.eval()
for i, data in enumerate(dataloader_test, 0):
    img, points, cat = data[:3]
    points = points.transpose(2,1).contiguous()
    points = points.cuda()
    pointsReconstructed  = network_preTrained_autoencoder(points)
    dist1, dist2 = distChamfer(points.transpose(2,1).contiguous(), pointsReconstructed)
    loss_net = (torch.mean(dist1)) + (torch.mean(dist2))
    val_loss.update(loss_net.item())
    # This is neccesary to avoid a memory leak issue

print("Previous decoder performances : ", val_loss.avg)

#Create network
network = SVR_AtlasNet(num_points = opt.num_points, nb_primitives = opt.nb_primitives)
network.apply(weights_init)
network.cuda()
if opt.model != '':
    network.load_state_dict(torch.load(opt.model))
    print(" Previous weight loaded ")

if opt.fix_decoder:
    network.decoder = network_preTrained_autoencoder.decoder

print(network)
network_preTrained_autoencoder.cpu()
network.cuda()
# ========================================================== #



# ===================CREATE optimizer================================= #
lrate = 0.001
params_dict = dict(network.named_parameters())
params = []

if opt.fix_decoder:
    optimizer = optim.Adam(network.encoder.parameters(), lr = lrate)
else:
    optimizer = optim.Adam(network.parameters(), lr = lrate)
# ========================================================== #

# =============DEFINE stuff for logs ======================================== #
num_batch = len(dataset) / opt.batchSize

train_loss = AverageValueMeter()
val_loss = AverageValueMeter()
val_view_loss = AverageValueMeter()
with open(logname, 'a') as f: #open and append
        f.write(str(network) + '\n')


trainloss_acc0 = 1e-9
trainloss_accs = 0

train_curve = []
val_curve = []
val_view_curve = []
labels_generated_points = torch.Tensor(range(1, (opt.nb_primitives+1)*(opt.num_points//opt.nb_primitives)+1)).view(opt.num_points//opt.nb_primitives,(opt.nb_primitives+1)).transpose(0,1)
labels_generated_points = (labels_generated_points)%(opt.nb_primitives+1)
labels_generated_points = labels_generated_points.contiguous().view(-1)
# ========================================================== #

# =============start of the learning loop ======================================== #
for epoch in range(opt.nepoch):
    #TRAIN MODE
    train_loss.reset()
    network.train()
    # learning rate schedule
    if epoch==100:
        if opt.fix_decoder:
            optimizer = optim.Adam(network.encoder.parameters(), lr = lrate/10.0)
        else:
            optimizer = optim.Adam(network.parameters(), lr = lrate/10.0)

    for i, data in enumerate(dataloader, 0):
        optimizer.zero_grad()

        img, points, cat = data[:3]
        img = img.cuda()
        points = points.cuda()

        pointsReconstructed  = network(img)
        dist1, dist2 = distChamfer(points, pointsReconstructed)
        loss_net = (torch.mean(dist1)) + (torch.mean(dist2))
        trainloss_accs = trainloss_accs * 0.99 + loss_net.item()
        trainloss_acc0 = trainloss_acc0 * 0.99 + 1
        loss_net.backward()
        train_loss.update(loss_net.item())
        optimizer.step()
        # This is neccesary to avoid a memory leak issue

        # VIZUALIZE
        if i%50 <= 0:
            vis.image(img[0].data.cpu().contiguous(), win = 'INPUT IMAGE TRAIN',opts = dict( title = "INPUT IMAGE TRAIN"))
            vis.scatter(X = points[0].data.cpu(),
                    win = 'TRAIN_INPUT',
                    opts = dict(
                        title = "TRAIN_INPUT",
                        markersize = 2,
                        ),
                    )
            vis.scatter(X = pointsReconstructed[0].data.cpu(),
                    Y = labels_generated_points[0:pointsReconstructed.size(1)],
                    win = 'TRAIN_INPUT_RECONSTRUCTED',
                    opts = dict(
                        title="TRAIN_INPUT_RECONSTRUCTED",
                        markersize=2,
                        ),
                    )
        print('[%d: %d/%d] train loss:  %f , %f ' %(epoch, i, len_dataset/32, loss_net.item(), trainloss_accs/trainloss_acc0))
        num_steps = i + len(dataloader) * epoch
        writer.add_scalar("train/loss", loss_net.cpu().detach().item(), num_steps)


    #UPDATE CURVES
    train_curve.append(train_loss.avg)

    with torch.no_grad():

        #VALIDATION on same models new views
        if epoch%10==0:
            val_view_loss.reset()
            network.eval()
            for i, data in enumerate(dataloader_test_view, 0):
                img, points, cat = data[:3]
                img = img.cuda()
                points = points.cuda()
                pointsReconstructed  = network(img)
                dist1, dist2 = distChamfer(points, pointsReconstructed)
                loss_net = (torch.mean(dist1)) + (torch.mean(dist2))
                # This is neccesary to avoid a memory leak issue
                val_view_loss.update(loss_net.item())
            #UPDATE CURVES
        val_view_curve.append(val_view_loss.avg)
        writer.add_scalar("val/same_model_new_view_loss", float(val_view_loss.avg), epoch)

        #VALIDATION
        val_loss.reset()
        for item in dataset_test.cat:
            dataset_test.perCatValueMeter[item].reset()

        network.eval()
        for i, data in enumerate(dataloader_test, 0):
            img, points, cat = data[:3]
            img = img.cuda()
            points = points.cuda()
            pointsReconstructed  = network(img)
            dist1, dist2 = distChamfer(points, pointsReconstructed)
            loss_net = (torch.mean(dist1)) + (torch.mean(dist2))
            val_loss.update(loss_net.item())
            # This is neccesary to avoid a memory leak issue
            dataset_test.perCatValueMeter[cat[0]].update(loss_net.item())
            if i%25 ==0 :
                vis.image(img[0].data.cpu().contiguous(), win = 'INPUT IMAGE VAL', opts = dict( title = "INPUT IMAGE TRAIN"))
                vis.scatter(X = points[0].data.cpu(),
                        win = 'VAL_INPUT',
                        opts = dict(
                            title = "VAL_INPUT",
                            markersize = 2,
                            ),
                        )
                vis.scatter(X = pointsReconstructed[0].data.cpu(),
                        Y = labels_generated_points[0:pointsReconstructed.size(1)],
                        win = 'VAL_INPUT_RECONSTRUCTED',
                        opts = dict(
                            title = "VAL_INPUT_RECONSTRUCTED",
                            markersize = 2,
                            ),
                        )
            print('[%d: %d/%d] val loss:  %f ' %(epoch, i, len(dataset_test), loss_net.item()))

        #UPDATE CURVES
        val_curve.append(val_loss.avg)

    vis.line(X=np.column_stack((np.arange(len(train_curve)),np.arange(len(val_curve)), np.arange(len(val_view_curve)))),
                 Y=np.column_stack((np.array(train_curve),np.array(val_curve), np.array(val_view_curve))),
                 win='loss',
                 opts=dict(title="loss", legend=["train_curve" + opt.env, "val_curve" + opt.env, "val_view_curve" + opt.env], markersize=2, ), )
    vis.line(X=np.column_stack((np.arange(len(train_curve)),np.arange(len(val_curve)), np.arange(len(val_view_curve)))),
                 Y=np.log(np.column_stack((np.array(train_curve),np.array(val_curve), np.array(val_view_curve)))),
                 win='log',
                 opts=dict(title="log", legend=["train_curve"+ opt.env, "val_curve"+ opt.env, "val_view_curve" + opt.env], markersize=2, ), )




    log_table = {
      "train_loss" : train_loss.avg,
      "val_loss" : val_loss.avg,
      "val_loss_new_views_same_models" : val_view_loss.avg,
      "epoch" : epoch,
      "lr" : lrate,
      "bestval" : best_val_loss,
    }
    print(log_table)
    for item in dataset_test.cat:
        print(item, dataset_test.perCatValueMeter[item].avg)
        log_table.update({item: dataset_test.perCatValueMeter[item].avg})
        writer.add_scalar("val/cate-%s-loss"%item, float(dataset_test.perCatValueMeter[item].avg), epoch)
    with open(logname, 'a') as f: #open and append
        f.write('json_stats: ' + json.dumps(log_table) + '\n')
    #save last network
    print('saving net...')
    torch.save(network.state_dict(), '%s/network.pth' % (dir_name))
