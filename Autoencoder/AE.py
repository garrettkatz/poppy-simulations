if __name__ == '__main__':
    import torch
    from torch.utils.data import TensorDataset, DataLoader, random_split
    import numpy as np
    import cv2
    import os


    class AE(torch.nn.Module):
        def __init__(self, img_size, num_nodes):
            super(AE, self).__init__()
            print('The number of input nodes =', img_size)
            self.encoder = torch.nn.ModuleList([
            torch.nn.Linear(img_size, num_nodes),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(num_nodes, 24),
            ])

        def forward(self, x):
            for i, _ in enumerate(self.encoder):
                x = self.encoder[i](x)
            return x

    if not os.path.exists('./data'):
        os.mkdir('./data')
    if os.path.exists('imgs.npy') and os.path.exists('targets.npy'):
        imgs = np.load('imgs.npy')
        targets = np.load('targets.npy')
    else:
        imgs, targets = [], []
        for count in range(3000):
            pos = np.load('./pred_img/downsampled_1/pos_{}.npy'.format(count))
            for i in range(3):
                image = cv2.imread('./pred_img/downsampled_1/ds_{}_{}.png'.format(i, count))
                # flatten image
                image = image.flatten()
                # concatenation
                concat = np.concatenate((image, pos[i]), axis=0)
                target = np.load('./pred_img/downsampled_1/ds_{}_{}.npy'.format(i, count))
                target = target.flatten().reshape(4,6)
                targets.append(target)
                imgs.append(concat)
        np.save('imgs.npy', imgs)
        np.save('targets.npy', targets)

    imgs = torch.from_numpy(np.array(imgs)).float()
    targets = torch.from_numpy(targets).float()
    
    # targets = torch.from_numpy((targets - targets.min())*255).float()

    epochs, bs = 1000, 5
    # size of hidden layer: 25, 50, 100, 200, 300, 1000
    num_nodes = [12, 20, 25, 50, 100, 200, 300, 1000]
    image_size = imgs.shape[1]
    dataset = TensorDataset(imgs, targets)
    train_dataset, test_dataset = random_split(dataset, [2000, 1000])
    train_loader = DataLoader(train_dataset, batch_size=bs)
    test_loader = DataLoader(test_dataset, batch_size=bs)
    loss_function = torch.nn.MSELoss()
    
    # Remeber to substitute parameter 'path' with your correct path.
    path = 'C:\\Users\\hbrch\\Desktop\\poppy-simulations-main\\plots\\'

    print('----------------------Training----------------------')
    for z in num_nodes:
        model = AE(image_size, z)
        optimizer = torch.optim.Adam(model.parameters(), lr = 1e-5 , weight_decay = 1e-8)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
        plot_loss = []
        
        name = 'H_{}_'.format(z)
        for epoch in range(epochs):
            for i, (image, target) in enumerate(train_loader):
                image = image.squeeze()
                target = target.flatten()
                pre_ten = model(image)
                pre_ten = pre_ten.flatten()
                loss = loss_function(pre_ten, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                diff = (pre_ten-target)**2
                Euclidean_ls, sum = [], 0
                for i in range(pre_ten.shape[0]):
                    # calculate sum of squared difference of coordinates(size: 3)
                    if i % 3 == 0:
                        sum = 0
                    sum += diff[i]
                    # calculate Euclidean distance
                    if (i+1) % 3 == 0: 
                        Euclidean_ls.append(torch.sqrt(sum).item())
                sqrt_sum = 0
                for ele in Euclidean_ls:
                    sqrt_sum += ele
                ave_sqrt_sum = sqrt_sum/len(Euclidean_ls)

            scheduler.step()
            plot_loss.append(ave_sqrt_sum)
            # elements in plot_loss are pytorch tensors
            print('MSE:', loss.item(), 'average Euclidean distance:', plot_loss[-1], 'epoch:', epoch, 'LR:', optimizer.param_groups[0]['lr'])

        np.save('train_data_{}'.format(z), np.array(plot_loss))
        
        model.eval()
        print('----------------------Testing----------------------')
        te_loss = []
        for i, (image, target) in enumerate(test_loader):
            image = image.squeeze()
            target = target.flatten()
            pre_ten = model(image)
            pre_ten = pre_ten.flatten()
            loss = loss_function(pre_ten, target)

            diff = (pre_ten-target)**2
            Euclidean_ls, sum = [], 0
            for i in range(pre_ten.shape[0]):
                # calculate sum of squared difference of coordinates(size: 3)
                if i % 3 == 0:
                    sum = 0
                sum += diff[i]
                # calculate Euclidean distance
                if (i+1) % 3 == 0: 
                    Euclidean_ls.append(torch.sqrt(sum).item())
            sqrt_sum = 0
            for ele in Euclidean_ls:
                sqrt_sum += ele
            ave_sqrt_sum = sqrt_sum/len(Euclidean_ls)
            te_loss.append(ave_sqrt_sum)
            print('average Euclidean distance', te_loss[-1])
            # diff = pre_ten-target
            # print('top 3 largest values:', list(torch.topk(diff, 3)[0].detach().numpy()))
            # print('top 3 smallest values:', list(torch.topk(diff, 3, largest=False)[0].detach().numpy()))

        np.save('./data/test_data_{}'.format(z), np.array(te_loss))
    pass