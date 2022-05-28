import torch
import torchvision
import time
from torchvision import transforms
#from torch.utils.tensorboard import SummaryWriter
from model_9layers_5x5 import *
from torch.utils.data import DataLoader

#定义训练设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

trans = transforms.Compose([
    transforms.Grayscale(1),
    transforms.ToTensor()
])
#数据集
train_data = torchvision.datasets.ImageFolder(root='Dataset/train',transform=trans)
test_data = torchvision.datasets.ImageFolder(root='Dataset/test',transform=trans)

test_data_size = len(test_data)
train_data_size = len(train_data)

train_dataloader = DataLoader(train_data,batch_size=32)
test_dataloader = DataLoader(test_data,batch_size=32)


#创建实例
billy = Billy()
billy.to(device)
#创建损失函数，优化器
loss_fn = nn.CrossEntropyLoss()
loss_fn.to(device)
learning_rate = 1e-2
optimizer = torch.optim.SGD(billy.parameters(),lr=learning_rate)
#设立步骤计数器
total_train_step = 0
total_test_step = 1
#设定训练轮数
epoch = 201
#设定计时器
start_time = time.time()
#writer = SummaryWriter('logs')

#训练x轮
with open('Reconding.csv', 'w') as optput:
    for i in range(epoch):
        print('----------------the No.{} training has started-----------------'.format(i+1))
        for data in train_dataloader:
            imgs,targets=data
            imgs = imgs.to(device)
            targets = targets.to(device)
            #输入神经网络
            output = billy(imgs)
            #损失值
            loss = loss_fn(output,targets)
            #优化器
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #训练次数计数器+1
            total_train_step += 1
            if total_train_step % 100 == 0:
                end_time = time.time()
                print('time_used: {}'.format(end_time - start_time))
                print('train times:{},loss:{}'.format(total_train_step,loss.item()))
                #writer.add_scalar("train_loss",loss.item(),total_train_step)

        total_test_lost = 0
        total_accuracy = 0
        total_test_time = 0
        with torch.no_grad():
            for data in test_dataloader:
                imgs,targets = data
                imgs = imgs.to(device)
                targets = targets.to(device)
                Start_test_time = time.time()
                output = billy(imgs)
                End_test_time = time.time()
                total_test_time += End_test_time-Start_test_time
                loss = loss_fn(output,targets)
                total_test_lost += loss.item()
                accuracy = (output.argmax(1) == targets).sum()
                total_accuracy += accuracy

        print('total loss:{}'.format(total_test_lost))
        print('total accuracy:{}'.format(total_accuracy/test_data_size))
        print('average test time:{}'.format(total_test_time / test_data_size))
        #writer.add_scalar("test_loss",total_test_lost,total_test_step)
        #writer.add_scalar('test_accuracy',total_accuracy/test_data_size,total_test_step)
        # 将结果写入文本
        optput.write(str(total_test_step)+r','+str(total_test_lost)+r','+str(float(total_accuracy/test_data_size))+'\n')
        total_test_step += 1

        if i % 50 == 0:
            torch.save(billy,'models_tem/billy_train_{}.pth'.format(i+1))
            print('model has been saved')

#writer.close()
with open('timeused.txt', 'w') as timeoutput:
    timeoutput.write('time_used: {}'.format(end_time - start_time)+'\n')
    timeoutput.write('time_used: {}'.format(total_test_time / test_data_size))