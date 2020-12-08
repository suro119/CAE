import os

def print_losses(epoch, iters, losses, t_comp, t_data, opt):
    message = '(epoch: %d, iters: %d, time: %.3f, data: %.3f) ' % (epoch, iters, t_comp, t_data)
    for k, v in losses.items():
        message += '%s: %.6f ' % (k, v)

    print(message)  # print the message
    log_name = os.path.join(opt.checkpoint_dir, opt.name, 'loss_log.txt')
    with open(log_name, "a") as log_file:
        log_file.write('%s\n' % message)  # save the message
        

def print_val_losses(epoch, losses, opt):
    message = '(epoch: %d) ' % (epoch)
    for k, v in losses.items():
        message += '%s: %.6f ' % ('val_' + k, v)

    print(message)  # print the message
    log_name = os.path.join(opt.checkpoint_dir, opt.name, 'loss_log.txt')
    with open(log_name, "a") as log_file:
        log_file.write('%s\n' % message)  # save the message