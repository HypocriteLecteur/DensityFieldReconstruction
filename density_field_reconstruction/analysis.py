import os
import numpy as np
import matplotlib.pyplot as plt

def compare_final_loss_history(log_file_path, log_file_path2, name=None):
    log_data = np.load(os.path.join(log_file_path, "statistics.npz"))
    log_data2 = np.load(os.path.join(log_file_path2, "statistics.npz"))

    percentage_improvement_2d_loss = ((log_data['final_loss_history'] - log_data2['final_loss_history']) / log_data['final_loss_history'] * 100).squeeze()
    with np.printoptions(precision=2, suppress=True):
        print(f'percentage of improvement in 2d loss, mean {np.mean(percentage_improvement_2d_loss)} +- {np.std(percentage_improvement_2d_loss)}')

    percentage_improvement_3d_loss = ((log_data['ise_loss_history'] - log_data2['ise_loss_history']) / log_data['ise_loss_history'] * 100).squeeze()
    with np.printoptions(precision=2, suppress=True):
        print(f'percentage of improvement in 3d loss, mean {np.mean(percentage_improvement_3d_loss[0:])} +- {np.std(percentage_improvement_3d_loss[0:])}')

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(211)
    # ax.scatter(np.arange(log_data['train_gaussian_scale_space_time'].size), log_data['train_gaussian_scale_space_time'], label='simulate vision time')
    ax.plot(np.arange(log_data['final_loss_history'].size), log_data['final_loss_history'], label='2d loss 1')
    ax.plot(np.arange(log_data2['final_loss_history'].size), log_data2['final_loss_history'], label='2d loss 2')
    plt.legend()
    plt.title(f'Loss History {name}')

    ax2 = fig.add_subplot(212)
    ax2.plot(np.arange(log_data['ise_loss_history'].size), log_data['ise_loss_history'], label='3d loss 1')
    ax2.plot(np.arange(log_data2['ise_loss_history'].size)[0:], log_data2['ise_loss_history'][0:], label='3d loss 2')
    plt.legend()
    plt.title(f'mean 3d loss 1 {np.mean(log_data['ise_loss_history']):.4f} mean 3d loss 2 {np.mean(log_data2['ise_loss_history']):.4f}')
