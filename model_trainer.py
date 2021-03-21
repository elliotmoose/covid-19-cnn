from torchsummary import summary
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from datetime import datetime
from matplotlib import pyplot as plt
import metrics
# def validate_model(model, dataloader, criterion, device):
#   running_loss = 0
#   correct_count = 0
#   for images, labels in dataloader:
#     images, labels = images.to(device), labels.to(device)    
#     prediction = model(images)
#     loss = criterion(prediction, labels).item()
#     running_loss += loss
#     equality = (labels.data == prediction.max(dim=1)[1])
#     correct_count += equality.type(torch.FloatTensor).sum()
#   accuracy = correct_count/len(dataloader.dataset)
#   running_loss /= len(dataloader)
#   return running_loss, accuracy

def validation(model, testloader, criterion, device, num_classes=2):
    test_loss = 0
    accuracy = 0
    confusion_matrix = torch.zeros(num_classes, num_classes)
    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)
                
        output = model(images)
        test_loss += criterion(output, labels).item()
        
        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()

        for label, prediction in zip(labels.view(-1), output.view(-1)):
            confusion_matrix[label.long(), prediction.long()] += 1

    return test_loss, accuracy, confusion_matrix

def test(model, testloader, device='cuda', num_classes=2):  
    model.to(device)
    accuracy = 0
    confusion_matrix = torch.zeros(num_classes, num_classes)

    with torch.no_grad():
        model.eval()
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
                    
            output = model(images)
            
            ps = torch.exp(output)
            equality = (labels.data == ps.max(dim=1)[1])
            accuracy += equality.type(torch.FloatTensor).mean()

            for t, p in zip(labels.view(-1), output.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
        
        print('Testing Accuracy: {:.3f}'.format(accuracy/len(testloader)))

    return accuracy, confusion_matrix


def train(model, model_name, batch_size, n_epochs, lr, train_loader, val_loader, saved_model_path, device = "cuda", num_classes=2):
    input_sample, _ =  next(iter(train_loader))
    print(summary(model, tuple(input_sample.shape[1:]), device=device))

    start_time = datetime.now()


    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr= lr) 

    train_loss_ls = []
    val_loss_ls = []
    
    running_loss = 0.0
    for e in range(n_epochs):  # loop over the dataset multiple times

        # Training
        model.train()
    #     train_loader1 = DataLoader(ld_train1, batch_size=batch_size, shuffle=True)
        with tqdm(train_loader, position=0, leave=False) as progress_bar:          
            for images, labels in progress_bar:
                images, labels = images.to(device), labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                output = model(images)
                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
        
        # Eval mode for predictions
        model.eval()

        # Turn off gradients for validation
        with torch.no_grad():
            test_loss, accuracy, confusion_matrix = validation(model, val_loader, criterion, device, num_classes)
            recall = metrics.recall(confusion_matrix, num_classes)
            precision = metrics.precision(confusion_matrix, num_classes)
            f1 = metrics.f1(confusion_matrix, num_classes)
 
        filepath = saved_model_path + f"{model_name}-{start_time}-b{batch_size}-e{e}.pt"
        torch.save(model, filepath)

        running_loss /= len(train_loader)

        time_elapsed = (datetime.now() - start_time)
        tqdm.write(f'\n===Epoch: {e+1}===')
        tqdm.write(f'== Loss: {running_loss:.3f} Time: {datetime.now()} Elapsed: {time_elapsed}')    
        tqdm.write(f'== Val Loss: {test_loss/len(val_loader):.3f} Val Accuracy: {accuracy/len(val_loader):.3f}') 
        tqdm.write(f'== Val Recall: {recall:.3f} Val Precision: {precision:.3f} Val F1: {f1:.3f}')

        train_loss_ls.append(running_loss) #/print_every
        val_loss_ls.append(test_loss/len(val_loader))
        running_loss = 0        

        # Make sure training is back on
        model.train()
                    

    print("Finished training")
    plt.plot(train_loss_ls, label = "train_loss")
    plt.plot(val_loss_ls, label = "val_loss")
    plt.legend()
    plt.savefig(saved_model_path+'train_val_loss.png')
    plt.show()
    return model
